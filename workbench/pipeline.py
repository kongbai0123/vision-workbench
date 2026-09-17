"""Read-only dataset import and snapshot-based, verified dataset publishing.

All runtime code lives in this application. No source project is imported or run.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import zipfile

import cv2
import numpy as np
import yaml

from composer_core.geometry import bounds, decode_rle, encode_rle, shape_polygons, validate_shape

FORMATS = {'native', 'coco', 'yolo_detection', 'yolo_segmentation', 'labelme', 'classification', 'jsonl'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
SPLITS = {'train', 'val', 'test'}
NATIVE_FORMAT = 'vision-workbench-native'
WORKING_TYPE = 'sam2_training_pseudo_labels'
VERIFIED_TYPE = 'sam2_human_verified_immutable_export'
FLAT_TYPE = 'sam2_human_verified_flat_training'
AUXILIARY = {'masks', 'mask', 'contours', 'overlays', 'overlay', 'pseudo', 'pending', 'rejected',
             '.venv', 'venv', 'node_modules', '__pycache__', '.git'}
# Per-source failures become import issues; anything else (e.g. job cancellation) propagates.
IMPORT_ERRORS = (ValueError, OSError, KeyError, TypeError, IndexError, cv2.error)


def _subrange(report, start, end):
    """Map a nested 0..1 progress fraction into [start, end] of the caller's fraction."""
    if report is None:
        return None
    return lambda fraction, detail='': report(start + (end - start) * fraction, detail)


def _tick(report, done, total, detail):
    if report is not None and total:
        report(done / total, f'{detail} {done} / {total}')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root, relative):
    root = Path(root).resolve()
    raw = Path(str(relative))
    if raw.is_absolute() or not str(relative).strip():
        raise ValueError('資料集宣告必須使用非空相對路徑')
    candidate = (root / raw).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f'資料集路徑越界：{relative}')
    if not candidate.is_file():
        raise ValueError(f'宣告的檔案不存在：{relative}')
    return candidate


def _dimensions(path):
    payload = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'無法解碼圖片：{Path(path).name}')
    return int(image.shape[1]), int(image.shape[0])


def _exif_orientation(path):
    """Read presentation metadata; pixel coordinates never apply EXIF rotation."""
    from PIL import Image

    with Image.open(path) as image:
        orientation = image.getexif().get(274, 1)
    if type(orientation) is not int or orientation not in range(1, 9):
        raise ValueError('圖片 EXIF Orientation 無效，無法確認外部顯示方向')
    return orientation


def _coordinate_metadata(path):
    return {'convention': 'raw_pixel_matrix', 'exif_orientation': _exif_orientation(path),
            'exif_orientation_applied': False}


def _identifier(*parts):
    return uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(parts, ensure_ascii=False, sort_keys=True)).hex


def _shape(kind, label, key, *, points=None, box=None, counts=None, metadata=None):
    shape = {'id': _identifier(*key), 'type': kind, 'label': label, 'hidden': False,
             'metadata': deepcopy(metadata or {})}
    if points is not None:
        shape['points'] = [[float(x), float(y)] for x, y in points]
        xs, ys = zip(*shape['points'])
        shape.update(x=min(xs), y=min(ys), width=max(xs)-min(xs), height=max(ys)-min(ys))
    if box is not None:
        shape.update(zip(('x', 'y', 'width', 'height'), box))
    if counts is not None:
        shape['counts'] = counts
    return shape


def _record(path, shapes, root, *, split='', batch=None, source=None, review='pending', expected_sha=None):
    path, root = Path(path).resolve(), Path(root).resolve()
    width, height = _dimensions(path)
    if expected_sha and _sha(path) != expected_sha:
        raise ValueError(f'{path.name}：圖片 SHA-256 不一致')
    if split is None:
        split = ''
    if split not in SPLITS | {''}:
        raise ValueError(f'{path.name}：split 無效：{split}')
    for shape in shapes:
        validate_shape(shape, width, height)
    ids = [shape['id'] for shape in shapes]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{path.name}：標註 ID 重複')
    provenance = deepcopy(source or {'path': str(path), 'format': 'image'})
    provenance['pixel_coordinates'] = _coordinate_metadata(path)
    return {'path': str(path), 'name': path.name, 'shapes': shapes, 'width': width, 'height': height,
            'batch_id': str(batch or f'{root.name}-{_identifier(str(root), split)[:10]}'),
            'split': split, 'source': provenance,
            'review_state': review}


def _native_import(root, document, report=None):
    if document.get('format') != NATIVE_FORMAT or document.get('schema_version') != 1:
        raise ValueError('不支援的 Native 資料版本')
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Native 封裝缺少 manifest.json')
    manifest = _json(manifest_path)
    declared_files = manifest.get('files', [])
    if not isinstance(declared_files, list) or 'project.json' not in {f.get('path') for f in declared_files}:
        raise ValueError('Native manifest 未驗證 project.json')
    verifying, loading = _subrange(report, 0, .5), _subrange(report, .5, 1)
    for index, item in enumerate(declared_files, 1):
        file = _safe_child(root, item['path'])
        if _sha(file) != item['sha256'] or file.stat().st_size != item['bytes']:
            raise ValueError(f'Native 封裝檔案驗證失敗：{item["path"]}')
        _tick(verifying, index, len(declared_files), '驗證 Native 檔案')
    assets = document.get('assets')
    if not isinstance(assets, list):
        raise ValueError('Native 封裝缺少 assets')
    records = []
    for asset in assets:
        if asset['image'] not in {f.get('path') for f in declared_files}:
            raise ValueError('Native manifest 缺少圖片雜湊記錄')
        path = _safe_child(root, asset['image'])
        record = _record(path, deepcopy(asset['shapes']), root, split=asset.get('split', ''),
                         batch=asset.get('batch_id'), source=asset.get('source'),
                         review=asset.get('review_state', 'pending'), expected_sha=asset['sha256'])
        if (record['width'], record['height']) != (asset['width'], asset['height']):
            raise ValueError(f'{path.name}：Native 尺寸與圖片不一致')
        record['name'] = asset.get('name', path.name)
        record['source'] = {**record['source'], 'imported_from': str(root), 'native_asset_id': asset.get('id')}
        records.append(record)
        _tick(loading, len(records), len(assets), '解析 Native 圖片')
    return records


def _asset_lines_import(root, path, report=None):
    """Image JSONL is a lossless native asset interchange, not generic text data."""
    manifest = _json(root / 'manifest.json')
    files = manifest.get('files', [])
    declared = {entry['path']: entry for entry in files}
    if path.relative_to(root).as_posix() not in declared:
        raise ValueError('JSONL 未被 manifest 宣告')
    verifying, loading = _subrange(report, 0, .5), _subrange(report, .5, 1)
    for index, (relative, entry) in enumerate(declared.items(), 1):
        file = _safe_child(root, relative)
        if _sha(file) != entry['sha256'] or file.stat().st_size != entry['bytes']:
            raise ValueError(f'JSONL 封裝雜湊不一致：{relative}')
        _tick(verifying, index, len(declared), '驗證 JSONL 檔案')
    total = 0
    if loading is not None:
        with path.open(encoding='utf-8-sig') as handle:
            total = sum(1 for line in handle if line.strip())
    records = []
    with path.open(encoding='utf-8-sig') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('type') != 'vision-workbench-asset' or row.get('schema_version') != 1:
                raise ValueError(f'JSONL 第 {number} 行不是影像資產交換格式')
            asset = row['asset']
            if asset['image'] not in declared:
                raise ValueError('JSONL 圖片未被 manifest 宣告')
            image = _safe_child(root, asset['image'])
            record = _record(image, deepcopy(asset['shapes']), root, split=asset.get('split', ''),
                             batch=asset.get('batch_id'), source=asset.get('source'),
                             review=asset.get('review_state', 'pending'), expected_sha=asset['sha256'])
            if (record['width'], record['height']) != (asset['width'], asset['height']):
                raise ValueError('JSONL 圖片尺寸不一致')
            record['name'] = asset.get('name', image.name)
            record['source'] = {**record['source'], 'imported_from': str(root), 'native_asset_id': asset.get('id')}
            records.append(record)
            _tick(loading, len(records), total, '解析 JSONL 圖片')
    return records


def _coco_import(path, root=None, flat_manifest=None, report=None):
    root = Path(root or path.parent).resolve()
    companion = root / 'manifest.json'
    if flat_manifest is None and companion.is_file():
        candidate = _json(companion)
        if candidate.get('dataset_type') == FLAT_TYPE:
            if _safe_child(root, candidate.get('annotations', 'annotations.json')) != path.resolve():
                raise ValueError('verified 封裝內有未宣告的 COCO 文件')
            flat_manifest = candidate
    document = _json(path)
    images, annotations, categories = (document.get(k) for k in ('images', 'annotations', 'categories'))
    if not all(isinstance(v, list) for v in (images, annotations, categories)):
        raise ValueError('COCO 必須有 images、annotations、categories 陣列')
    for items, label in ((images, '圖片'), (annotations, '標註'), (categories, '類別')):
        ids = [x.get('id') for x in items]
        if any(x is None for x in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'COCO {label} ID 缺失或重複')
    names = {x['id']: x['name'] for x in categories}
    image_ids = {x['id'] for x in images}
    grouped = defaultdict(list)
    for annotation in annotations:
        if annotation.get('image_id') not in image_ids or annotation.get('category_id') not in names:
            raise ValueError('COCO 標註引用不存在的圖片／類別')
        grouped[annotation['image_id']].append(annotation)
    sample_map = {}
    if flat_manifest is not None:
        manifest = flat_manifest
        if (manifest.get('image_count', manifest.get('sample_count')) != len(images)
                or manifest.get('annotation_count', manifest.get('sample_count')) != len(annotations)):
            raise ValueError('verified manifest 與 COCO 數量不一致')
        if not manifest.get('annotations_sha256') or _sha(path) != manifest['annotations_sha256']:
            raise ValueError('verified annotations SHA-256 不一致或缺失')
        samples = manifest.get('samples', [])
        if len(samples) != manifest.get('sample_count') or any(x.get('review_status') != 'human_verified' for x in samples):
            raise ValueError('verified 樣本數量或審核狀態無效')
        sample_map = {x.get('image'): x for x in samples}
        if len(sample_map) != len(samples) or set(sample_map) != {x['file_name'] for x in images}:
            raise ValueError('verified sample.image 與 COCO 不一一對應')
    records = []
    file_names = [x['file_name'].casefold() for x in images]
    if len(file_names) != len(set(file_names)):
        raise ValueError('COCO 圖片檔名重複')
    _tick(report, 0, len(images), '解析圖片')
    for image in images:
        image_path = _safe_child(root, image['file_name'])
        width, height = _dimensions(image_path)
        if (image.get('width'), image.get('height')) != (width, height):
            raise ValueError(f'{image_path.name}：COCO 圖片尺寸不一致')
        sample = sample_map.get(image['file_name'], {})
        if sample and not sample.get('image_sha256'):
            raise ValueError('verified 樣本缺少 SHA-256')
        split = image.get('split') or sample.get('split') or ''
        if image.get('split') and sample.get('split') and image['split'] != sample['split']:
            raise ValueError('verified split 與 COCO 不一致')
        batch = image.get('batch_id') or image.get('session_id') or sample.get('source', {}).get('session_id')
        shapes = []
        for annotation in grouped[image['id']]:
            if annotation.get('keypoints'):
                raise ValueError('COCO keypoints 尚未映射為獨立點標註；請使用 Native 保留此任務，不會靜默忽略')
            label = names[annotation['category_id']]
            key = (str(path.resolve()), annotation['id'])
            metadata = {'format': 'coco', 'annotation': deepcopy(annotation)}
            segment = annotation.get('segmentation')
            if isinstance(segment, dict):
                if segment.get('size') != [height, width]:
                    raise ValueError('COCO RLE size 與圖片尺寸不一致')
                mask = decode_rle(segment.get('counts'), width, height)
                shape = _shape('mask', label, key, box=[0, 0, width, height], counts=encode_rle(mask), metadata=metadata)
            elif isinstance(segment, list) and segment:
                polygons = []
                for polygon in segment:
                    if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                        raise ValueError('COCO polygon 頂點無效')
                    points = [[polygon[i], polygon[i+1]] for i in range(0, len(polygon), 2)]
                    validate_shape(_shape('polygon', label, key, points=points), width, height)
                    polygons.append(points)
                if len(polygons) == 1:
                    shape = _shape('polygon', label, key, points=polygons[0], metadata=metadata)
                else:
                    # One COCO instance may have disconnected polygons. Preserve
                    # one stable object plus the original vector components.
                    mask = np.zeros((height, width), np.uint8)
                    for points in polygons:
                        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
                    metadata['original_polygons'] = polygons
                    metadata['canonical_origin'] = 'coco_multipart_polygon'
                    shape = _shape('mask', label, key, box=[0, 0, width, height], counts=encode_rle(mask), metadata=metadata)
            elif 'bbox' in annotation:
                shape = _shape('rectangle', label, key, box=annotation['bbox'], metadata=metadata)
            else:
                raise ValueError(f'COCO 標註 {annotation["id"]} 沒有支援的幾何資料')
            shape['id'] = str(annotation.get('workbench_shape_id') or shape['id'])
            shapes.append(shape)
        if flat_manifest is not None and not shapes:
            raise ValueError('verified 圖片沒有標註')
        source = {'format': 'coco', 'path': str(image_path), 'annotation_path': str(path),
                  'image': deepcopy(image), 'sample': deepcopy(sample)}
        records.append(_record(image_path, shapes, root, split=split, batch=batch, source=source,
                               review='approved' if flat_manifest is not None else 'pending',
                               expected_sha=sample.get('image_sha256')))
        _tick(report, len(records), len(images), '解析圖片')
    return records


def _working_import(root, manifest=None, selected_path=None, report=None):
    """Import a reviewable SAM2 dataset as pending Workbench assets.

    A working manifest describes how source images, masks and QA overlays are
    related; it is useful import metadata, not a reason to reject otherwise
    valid images and COCO annotations.  Only canonical source images become
    assets.  Selecting a related mask or overlay resolves to that source image.
    """
    records, samples = _working_records(root, manifest, report)
    if selected_path is None:
        return records
    return _select_working(root, records, samples, selected_path)


def _working_records(root, manifest=None, report=None):
    """Parse one SAM2 working session; callers select from the result without re-parsing."""
    root = Path(root).resolve()
    manifest = manifest or _json(root / 'manifest.json')
    if manifest.get('dataset_type') != WORKING_TYPE:
        raise ValueError('不是支援的 SAM2 工作資料格式')
    annotations = root / 'label' / 'instances.json'
    if not annotations.is_file():
        raise ValueError('SAM2 工作資料缺少 label/instances.json')
    records = _coco_import(annotations, root, report=report)
    samples = manifest.get('samples')
    if not isinstance(samples, list):
        raise ValueError('SAM2 manifest 缺少 samples 陣列')
    by_image = {str(sample.get('image')): sample for sample in samples
                if isinstance(sample, dict) and sample.get('image')}
    if len(by_image) != len(samples):
        raise ValueError('SAM2 manifest 的圖片宣告缺失或重複')
    for record in records:
        relative = Path(record['path']).relative_to(root).as_posix()
        sample = by_image.get(relative)
        if sample is None:
            raise ValueError(f'SAM2 COCO 圖片未被 manifest 宣告：{relative}')
        record['review_state'] = 'pending'
        record['batch_id'] = str(manifest.get('session_id') or record['batch_id'])
        record['source']['sample'] = deepcopy(sample)
        record['source']['sam2_review_status'] = sample.get('review_status', manifest.get('review_status'))
    return records, samples


def _select_working(root, records, samples, selected_path):
    root = Path(root).resolve()
    annotations = root / 'label' / 'instances.json'
    selected = Path(selected_path).resolve()
    if selected in {root / 'manifest.json', annotations}:
        return records
    try:
        selected_relative = selected.relative_to(root).as_posix()
    except ValueError:
        return []
    selected_images = {
        str(sample.get('image'))
        for sample in samples
        if selected_relative in {
            str(sample.get('image', '')),
            str(sample.get('mask', '')),
            str(sample.get('qa_overlay', '')),
            str(sample.get('contours', '')),
        }
    }
    return [record for record in records
            if Path(record['path']).relative_to(root).as_posix() in selected_images]


def _labelme_import(path):
    document = _json(path)
    image_name = document.get('imagePath') or path.with_suffix('.png').name
    image_path = _safe_child(path.parent, image_name)
    width, height = _dimensions(image_path)
    if document.get('imageWidth') not in (None, width) or document.get('imageHeight') not in (None, height):
        raise ValueError(f'{path.name}：LabelMe 尺寸不一致')
    shapes = []
    for index, raw in enumerate(document.get('shapes', [])):
        kind = raw.get('shape_type', 'polygon')
        label, points = raw.get('label'), raw.get('points')
        metadata = {'format': 'labelme', 'shape': deepcopy(raw)}
        key = (str(path.resolve()), index)
        if kind == 'rectangle':
            if not isinstance(points, list) or len(points) != 2:
                raise ValueError('LabelMe rectangle 必須有 2 點')
            x0, x1 = sorted((points[0][0], points[1][0]))
            y0, y1 = sorted((points[0][1], points[1][1]))
            shape = _shape('rectangle', label, key, box=[x0, y0, x1-x0, y1-y0], metadata=metadata)
        elif kind in {'polygon', 'linestrip', 'point', 'line', 'rotation'}:
            shape = _shape({'line': 'linestrip', 'rotation': 'obb'}.get(kind, kind), label, key, points=points, metadata=metadata)
        else:
            raise ValueError(f'LabelMe 不支援 {kind}，請先轉為 polygon；未略過此標註')
        shape['id'] = str(raw.get('workbench_shape_id') or shape['id'])
        shapes.append(shape)
    meta = document.get('workbench', {})
    return [_record(image_path, shapes, path.parent, split=meta.get('split', ''), batch=meta.get('batch_id'),
                    source={'format': 'labelme', 'path': str(image_path), 'annotation_path': str(path),
                            'metadata': deepcopy(meta)})]


def _class_names(root, label_path=None):
    for name in ('data.yaml', 'data.yml', 'dataset.yaml', 'dataset.yml'):
        candidate = root / name
        if candidate.is_file():
            document = yaml.safe_load(candidate.read_text('utf-8-sig')) or {}
            names = document.get('names')
            if isinstance(names, dict):
                mapped = {int(k): v for k, v in names.items()}
                if set(mapped) != set(range(len(mapped))):
                    raise ValueError('YOLO names ID 必須是連續 0..N-1')
                names = [mapped[i] for i in range(len(mapped))]
            if not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() for n in names):
                raise ValueError('YOLO YAML 缺少有效 names')
            return names
    for candidate in (root / 'classes.txt', root / 'labels' / 'classes.txt'):
        if candidate.is_file():
            names = candidate.read_text('utf-8-sig').splitlines()
            if not names or any(not n.strip() for n in names):
                raise ValueError('classes.txt 類別為空')
            return names
    raise ValueError('YOLO 標註缺少 data.yaml names 或 classes.txt；不猜測類別名称')


def _yolo_record(image, label, root, names, report_item=None):
    width, height = _dimensions(image)
    shapes = []
    for number, line in enumerate(label.read_text('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        tokens = line.split()
        cid = int(tokens[0])
        if not 0 <= cid < len(names):
            raise ValueError(f'{label.name}:{number} 類別未在 names 定義')
        values = [float(v) for v in tokens[1:]]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError(f'{label.name}:{number} 座標超出 0..1')
        key = (str(label.resolve()), number)
        metadata = {'format': 'yolo', 'line': number, 'original': line}
        if len(values) == 4:
            x, y, w, h = values
            x0, y0, x1, y1 = (x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height
            # Decimal TXT serialization can introduce sub-micro-pixel drift.
            x0 = 0.0 if -1e-6 < x0 < 0 else x0
            y0 = 0.0 if -1e-6 < y0 < 0 else y0
            x1 = float(width) if width < x1 < width+1e-6 else x1
            y1 = float(height) if height < y1 < height+1e-6 else y1
            shape = _shape('rectangle', names[cid], key,
                           box=[x0, y0, x1-x0, y1-y0], metadata=metadata)
        elif len(values) >= 6 and len(values) % 2 == 0:
            shape = _shape('polygon', names[cid], key,
                           points=[[values[i]*width, values[i+1]*height] for i in range(0, len(values), 2)], metadata=metadata)
        else:
            raise ValueError(f'{label.name}:{number} YOLO 行長度不支援')
        shapes.append(shape)
    split = next((part for part in image.relative_to(root).parts if part in SPLITS), '')
    source = {'format': 'yolo', 'path': str(image), 'annotation_path': str(label)}
    batch = None
    if report_item is not None:
        if report_item.get('split') != split:
            raise ValueError('YOLO conversion_report split 與圖片目錄不一致')
        expected = []
        for entry in report_item.get('shapes', []):
            count = entry.get('output_count', 0)
            for component in range(count):
                expected.append((entry, component))
        if len(expected) != len(shapes):
            raise ValueError('YOLO conversion_report 標註數量與 TXT 不一致')
        for shape, (entry, component) in zip(shapes, expected):
            if shape['label'] != entry['class_name']:
                raise ValueError('YOLO conversion_report 類別與 TXT 不一致')
            shape['id'] = entry['shape_id'] if entry['output_count'] == 1 else f'{entry["shape_id"]}:{component}'
            shape['metadata']['source_shape_id'] = entry['shape_id']
        batch = report_item.get('batch_id')
        source['origin'] = deepcopy(report_item.get('source', {}))
        source['origin_asset_id'] = report_item.get('asset_id')
    return _record(image, shapes, root, split=split, batch=batch, source=source)


def _files(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and
                  not any(part.lower() in AUXILIARY or part.startswith('.') for part in p.relative_to(root).parts[:-1]))


def _folder_import(root, report=None):
    # Protected roots are identified before any ordinary image traversal.
    for name in ('manifest.json', 'export_manifest.json'):
        candidate = root / name
        if not candidate.is_file():
            continue
        document = _json(candidate)
        kind = document.get('dataset_type')
        if kind == WORKING_TYPE:
            return _working_import(root, document, report=report)
        if kind == VERIFIED_TYPE:
            declared = document.get('training_ready', {}).get('path')
            if not isinstance(declared, str) or not declared:
                raise ValueError('verified 封裝未宣告 training_ready')
            ready = (root / declared).resolve()
            if not ready.is_relative_to(root) or not ready.is_dir():
                raise ValueError('verified training_ready 越界或不存在')
            manifest = _json(ready / 'manifest.json')
            if manifest.get('dataset_type') != FLAT_TYPE:
                raise ValueError('verified training_ready 類型不符')
            return _coco_import(_safe_child(ready, manifest.get('annotations', 'annotations.json')), ready, manifest,
                                report=report)
        if kind == FLAT_TYPE:
            return _coco_import(_safe_child(root, document.get('annotations', 'annotations.json')), root, document,
                                report=report)
    project_file = root / 'project.json'
    if project_file.is_file():
        document = _json(project_file)
        if document.get('format') == NATIVE_FORMAT:
            return _native_import(root, document, report=report)
    dataset_file = root / 'dataset.json'
    if dataset_file.is_file():
        document = _json(dataset_file)
        if document.get('format') in {'vision-workbench-jsonl', 'vision-workbench-classification'}:
            return _asset_lines_import(root, _safe_child(root, document.get('annotations', 'dataset.jsonl')),
                                       report=report)
    working_manifests = []
    for candidate in root.rglob('manifest.json'):
        if any(part in {'.venv', 'node_modules', '.git'} for part in candidate.relative_to(root).parts):
            continue
        try:
            if _json(candidate).get('dataset_type') == WORKING_TYPE:
                working_manifests.append(candidate)
        except json.JSONDecodeError:
            raise ValueError(f'無法解析 manifest：{candidate.name}')
    if working_manifests:
        records = []
        for index, candidate in enumerate(working_manifests):
            nested = _subrange(report, index / len(working_manifests), (index + 1) / len(working_manifests))
            records.extend(_working_import(candidate.parent, _json(candidate), report=nested))
        return records
    files = _files(root)
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    json_files = [p for p in files if p.suffix.lower() == '.json' and p.name not in {'manifest.json', 'conversion_report.json', 'export_manifest.json'}]
    # One unit per annotation document and per image; COCO documents report inside their unit.
    total, step = len(json_files) + len(images), 0
    _tick(report, step, total, '掃描檔案')
    records, used_images, used_labels = [], set(), set()
    for file in json_files:
        document = _json(file)
        nested = _subrange(report, step / total, (step + 1) / total)
        if isinstance(document, dict) and all(isinstance(document.get(k), list) for k in ('images', 'annotations', 'categories')):
            found = _coco_import(file, file.parent, report=nested)
        elif isinstance(document, dict) and isinstance(document.get('shapes'), list) and 'imagePath' in document:
            found = _labelme_import(file)
        else:
            found = []
        step += 1
        _tick(report, step, total, '掃描檔案')
        for record in found:
            key = Path(record['path']).resolve()
            if key in used_images:
                raise ValueError(f'{key.name} 被多份標註文件同時引用；請選擇單一資料版本')
            used_images.add(key)
            # Keep one import folder as batch unless explicit metadata/split exists.
            if not record['split'] and not record['source'].get('image', {}).get('session_id') and not record['source'].get('image', {}).get('batch_id') and not record['source'].get('metadata', {}).get('batch_id'):
                record['batch_id'] = f'{root.name}-{_identifier(str(root), "")[:10]}'
            records.append(record)
    names = None
    yolo_dataset = any((root / name).is_file() for name in ('data.yaml', 'data.yml', 'dataset.yaml', 'dataset.yml', 'classes.txt'))
    report_items = {}
    conversion_path = root / 'conversion_report.json'
    if conversion_path.is_file():
        conversion = _json(conversion_path)
        if conversion.get('tool') == 'Vision Workbench' and str(conversion.get('format', '')).startswith('yolo'):
            report_items = {item['image']: item for item in conversion.get('items', [])}
    for image in images:
        step += 1
        if image.resolve() in used_images:
            _tick(report, step, total, '掃描檔案')
            continue
        relative = image.relative_to(root)
        possible = [image.with_suffix('.txt')]
        parts = list(relative.parts)
        if 'images' in parts:
            parts[parts.index('images')] = 'labels'
            possible.insert(0, root.joinpath(*parts).with_suffix('.txt'))
        possible.append(root / 'labels' / relative.with_suffix('.txt'))
        labels = list(dict.fromkeys(p.resolve() for p in possible if p.is_file()))
        if len(labels) > 1:
            raise ValueError(f'{image.name} 有多份可能的 YOLO 標註')
        if labels:
            if names is None:
                names = _class_names(root)
            records.append(_yolo_record(image, labels[0], root, names, report_items.get(relative.as_posix())))
            used_labels.add(labels[0])
        else:
            if yolo_dataset:
                raise ValueError(f'{image.name}：YOLO 資料集缺少配對 TXT；空白背景圖片也請提供空 TXT')
            records.append(_record(image, [], root))
        _tick(report, step, total, '掃描檔案')
    for label in files:
        if label.suffix.lower() == '.txt' and label.name not in {'classes.txt', 'README.txt', 'train.txt', 'val.txt', 'test.txt'} and ('labels' in label.relative_to(root).parts):
            if label.resolve() not in used_labels:
                raise ValueError(f'孤立或重複的 YOLO TXT：{label.relative_to(root)}')
    if not records:
        raise ValueError('找不到支援的圖片與標註；ZIP 請先解壓縮再匯入資料夾')
    return records


YOLO_MARKERS = ('data.yaml', 'data.yml', 'dataset.yaml', 'dataset.yml', 'classes.txt')


class _ScanProgress:
    """Monotonic scan percentages over parse units; a folder or SAM2 session counts once."""

    def __init__(self, callback, units):
        self.callback, self.units, self.done, self.last = callback, max(1, units), 0, None

    def unit(self, label):
        if self.callback is None:
            return None
        base = self.done
        return lambda fraction, detail='': self._emit(base + min(1.0, max(0.0, fraction)), label, detail)

    def finish(self, label):
        self.done += 1
        self._emit(self.done, label, '')

    def _emit(self, value, label, detail):
        if self.callback is None:
            return
        percent = round(min(100.0, value * 100 / self.units), 2)
        if self.last is not None and percent < self.last:
            return
        self.last = percent
        self.callback(f'{label}：{detail}' if detail else label, percent)


def _cached_json(path, cache):
    if path not in cache:
        cache[path] = _json(path)
    return cache[path]


def _cached(cache, key, parse):
    """Parse once per import call; a failure is reported again for every dependent path."""
    if key not in cache:
        try:
            cache[key] = (parse(), None)
        except IMPORT_ERRORS as exc:
            cache[key] = (None, exc)
    value, error = cache[key]
    if error is not None:
        raise error.with_traceback(None)
    return value


def _source_working_root(path, manifests):
    """Outermost SAM2 working root above path; verified packages only expose training_ready."""
    working_root = None
    for ancestor in (path.parent, *path.parent.parents):
        for manifest_name in ('manifest.json', 'export_manifest.json'):
            parent_manifest = ancestor / manifest_name
            if not parent_manifest.is_file():
                continue
            try:
                metadata = _cached_json(parent_manifest, manifests)
            except json.JSONDecodeError:
                raise ValueError('來源上層 manifest 無法解析')
            if metadata.get('dataset_type') == WORKING_TYPE:
                working_root = ancestor
            if metadata.get('dataset_type') == VERIFIED_TYPE:
                declared = metadata.get('training_ready', {}).get('path')
                ready = (ancestor / declared).resolve() if isinstance(declared, str) else None
                if ready is None or not ready.is_relative_to(ancestor) or not path.is_relative_to(ready):
                    raise ValueError('verified 封裝只允許匯入宣告的 training_ready；不可單獨掃描 masks/overlay 等衍生檔案')
    return working_root


def _yolo_root(path):
    root = next((parent for parent in path.parents
                 if any((parent / name).is_file() for name in YOLO_MARKERS)), None)
    if root is None:
        raise ValueError('YOLO 標註需要配對圖片及 data.yaml 或 classes.txt；請拖入完整資料集資料夾')
    if root.name == 'labels' and (root.parent / 'images').is_dir():
        root = root.parent
    return root


def _plan_source(path, manifests):
    """Name the parse unit a dropped path belongs to, without decoding any image."""
    working_root = _source_working_root(path, manifests)
    if working_root is not None and path.is_file():
        return 'session', working_root
    if path.is_dir():
        return 'folder', path
    suffix = path.suffix.lower()
    if path.is_file() and suffix in {'.yaml', '.yml', '.txt'}:
        return 'folder', _yolo_root(path)
    if path.is_file() and suffix in IMAGE_EXTENSIONS:
        parent_manifest = path.parent / 'manifest.json'
        if parent_manifest.is_file() and _cached_json(parent_manifest, manifests).get('dataset_type') == FLAT_TYPE:
            return 'folder', path.parent
    return 'source', path


def import_sources(paths: list[str | Path], progress=None) -> dict:
    """Parse dropped sources read-only into import records.

    ``progress(message, percent)`` receives monotonic scan percentages. A SAM2
    session or dataset folder is parsed once per call however many of its files
    are selected, so dropping N files costs about the same as dropping the folder.
    """
    records, issues, seen = [], [], set()
    # A multi-file drop can list images before a COCO/native annotation file.
    # Read datasets and annotations first so de-duplication retains the labels.
    ordered = sorted(paths, key=lambda raw: 0 if Path(raw).is_dir() else
                     2 if Path(raw).suffix.lower() in IMAGE_EXTENSIONS else 1)
    manifests, sessions, folders, plans = {}, {}, {}, []
    for raw in ordered:
        path = Path(raw).expanduser().resolve()
        try:
            plans.append((path, _plan_source(path, manifests), None))
        except IMPORT_ERRORS as exc:
            plans.append((path, ('failed', len(plans)), exc))
    scan = _ScanProgress(progress, len({plan for _path, plan, _error in plans}))
    finished = set()
    for path, plan, error in plans:
        kind, target = plan
        label = (f'解析 SAM2 工作資料 {target.name}' if kind == 'session' else
                 f'掃描資料夾 {target.name}' if kind == 'folder' else f'解析 {path.name}')
        report = None if plan in finished else scan.unit(label)
        def folder(root, report=report):
            return _cached(folders, root, lambda: _folder_import(root, report=report))
        try:
            if error is not None:
                raise error
            if kind == 'session':
                found = _select_working(target, *_cached(sessions, target,
                                                          lambda: _working_records(target, report=report)), path)
                if not found:
                    raise ValueError('選取的檔案未被 SAM2 manifest 宣告為圖片或標註')
            elif path.is_dir():
                found = folder(path)
            elif path.suffix.lower() == '.json' and path.is_file():
                document = _json(path)
                if document.get('dataset_type') in {WORKING_TYPE, VERIFIED_TYPE, FLAT_TYPE}:
                    found = folder(path.parent)
                elif document.get('format') == NATIVE_FORMAT:
                    found = _native_import(path.parent, document, report=report)
                elif document.get('format') in {'vision-workbench-jsonl', 'vision-workbench-classification'}:
                    found = folder(path.parent)
                elif isinstance(document.get('images'), list):
                    parent_manifest = path.parent / 'manifest.json'
                    if parent_manifest.is_file() and _json(parent_manifest).get('dataset_type') == FLAT_TYPE:
                        found = folder(path.parent)
                    else:
                        found = _coco_import(path, report=report)
                elif isinstance(document.get('shapes'), list):
                    found = _labelme_import(path)
                else:
                    raise ValueError('不支援的 JSON 結構')
            elif path.suffix.lower() == '.jsonl' and path.is_file():
                found = _asset_lines_import(path.parent, path, report=report)
            elif path.is_file() and path.suffix.lower() in {'.yaml', '.yml', '.txt'}:
                if path.suffix.lower() in {'.yaml', '.yml'} and path.name not in YOLO_MARKERS:
                    raise ValueError('不支援的 YAML；YOLO 請使用 data.yaml 或 dataset.yaml')
                found = folder(target)
                if path.name not in YOLO_MARKERS:
                    found = [r for r in found if Path(r.get('source', {}).get('annotation_path', '')).resolve() == path]
                    if not found:
                        raise ValueError('找不到此 YOLO TXT 的配對圖片，請檢查 images／labels 目錄與檔名')
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                if kind == 'folder':
                    found = [r for r in folder(target) if Path(r['path']) == path]
                    if not found:
                        raise ValueError('此圖片未被 verified manifest 宣告')
                elif path.with_suffix('.json').is_file():
                    found = _labelme_import(path.with_suffix('.json'))
                elif path.with_suffix('.txt').is_file():
                    found = [_yolo_record(path, path.with_suffix('.txt'), path.parent, _class_names(path.parent))]
                else:
                    found = [_record(path, [], path.parent)]
            else:
                raise ValueError('路徑不存在或格式不支援；ZIP 請先解壓縮')
            for record in found:
                key = str(Path(record['path']).resolve()).casefold()
                if key in seen:
                    issues.append({'level': 'warning', 'message': f'略過重複選取：{record["name"]}', 'source': str(path)})
                    continue
                seen.add(key)
                records.append(record)
                if any(s.get('metadata', {}).get('canonical_origin') == 'coco_multipart_polygon' for s in record['shapes']):
                    issues.append({'level': 'warning', 'message': 'COCO 多區塊物件以 Mask 編輯；完整原始向量保留在標註 metadata', 'source': str(path)})
        except IMPORT_ERRORS as exc:
            issues.append({'level': 'error', 'message': str(exc), 'source': str(path)})
        if plan not in finished:
            finished.add(plan)
            scan.finish(label)
    return {'records': records, 'issues': issues}


def _split_plan(assets, required=False):
    assigned = {a['id']: a.get('split') or '' for a in assets}
    explicit = [s for s in assigned.values() if s]
    if any(s not in SPLITS for s in explicit):
        raise ValueError('split 僅可為 train / val / test')
    if explicit and len(explicit) != len(assets):
        raise ValueError('split 資訊不完整：不可混用已分組與未分組圖片')
    for asset in assets:
        if not asset.get('batch_id'):
            raise ValueError('每張圖片必須有 batch_id，才能保留來源追溯資訊')
    if not explicit and required:
        from .splitting import stratified_split
        assigned, _ = stratified_split(assets, {'train': 80, 'val': 20, 'test': 0})
    if required and not {'train', 'val'}.issubset(set(assigned.values())):
        raise ValueError('YOLO 必須同時有 train 與 val')
    hashes = defaultdict(set)
    for asset in assets:
        if assigned[asset['id']]:
            hashes[asset.get('sha256')].add(assigned[asset['id']])
    if any(len(splits) > 1 for splits in hashes.values()):
        raise ValueError('相同圖片 SHA-256 不可跨訓練／驗證／測試組')
    return assigned


def _shape_losses(shape, asset, format_key, tolerance):
    kind = shape['type']
    losses = []
    if format_key in {'native', 'jsonl'}:
        return losses
    if format_key == 'classification':
        return [{'reason': 'geometry_to_image_class', 'message': '分類讀取器使用整張圖片的類別；完整幾何另保存在 dataset.jsonl',
                 'asset_id': asset['id'], 'shape_id': shape['id']}]
    if kind in {'point', 'linestrip'} and format_key != 'labelme':
        raise ValueError(f'{format_key} 不支援 {kind}；請使用 Native 或 LabelMe，不會略過物件')
    if format_key == 'yolo_detection' and kind != 'rectangle':
        losses.append({'reason': 'geometry_to_bbox', 'message': '以軸對齊矩形取代原始輪廓'})
    if format_key in {'yolo_segmentation', 'labelme'} and kind == 'mask':
        _, diagnostics = shape_polygons(shape, asset['width'], asset['height'], tolerance)
        losses.append({'reason': 'mask_to_polygon', 'message': '像素 Mask 轉為多邊形；Native／COCO 保留精確 RLE', **diagnostics})
    if kind == 'obb' and format_key in {'coco', 'yolo_segmentation', 'labelme'}:
        losses.append({'reason': 'obb_to_polygon', 'message': '保留四個頂點，旋轉框的專用類型改為多邊形'})
    if format_key == 'coco' and kind == 'mask' and shape.get('metadata', {}).get('canonical_origin') == 'coco_multipart_polygon':
        losses.append({'reason': 'multipart_polygon_to_rle', 'message': '多區塊向量標註以精確工作 Mask RLE 輸出；原始向量保留於 Native metadata'})
    return [{**item, 'asset_id': asset['id'], 'shape_id': shape['id']} for item in losses]


def validate_project(snapshot: dict, format_key='native', tolerance=0) -> dict:
    errors, warnings, losses = [], [], []
    assets = snapshot.get('assets', [])
    if format_key not in FORMATS:
        errors.append('不支援的輸出格式')
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        errors.append('輪廓誤差必須為有限非負數')
    approved = [a for a in assets if a.get('review_state') == 'approved']
    if not approved:
        errors.append('沒有已核准圖片；請先完成審核')
    asset_ids = [a.get('id') for a in assets]
    if any(not i for i in asset_ids) or len(set(asset_ids)) != len(asset_ids):
        errors.append('圖片 ID 缺失或重複')
    classes = snapshot.get('classes', [])
    if not isinstance(classes, list) or any(not isinstance(c, str) or not c.strip() for c in classes) or len(classes) != len(set(classes)):
        errors.append('類別清單無效或重複')
        classes = []
    for asset in approved:
        try:
            path = Path(asset['image_path'])
            if not path.is_file() or not asset.get('sha256') or _sha(path) != asset['sha256']:
                raise ValueError('圖片遺失或 SHA-256 不符')
            width, height = _dimensions(path)
            if (width, height) != (asset['width'], asset['height']):
                raise ValueError('圖片尺寸與專案記錄不一致')
            orientation = _exif_orientation(path)
            if orientation != 1:
                if format_key not in {'native', 'jsonl'}:
                    raise ValueError(
                        f'EXIF Orientation={orientation}：外部軟體可能自動旋轉圖片，造成標註錯位。'
                        '請先將圖片與標註一起正規化方向後再匯出此格式；'
                        'Native／JSONL 可保留原始位元組與原像素座標。'
                    )
                warnings.append(
                    f'{asset.get("name", asset["id"])}：保留 EXIF Orientation={orientation} 與原始圖片；'
                    '標註使用原像素矩陣座標，不套用 EXIF 旋轉，已記錄於輸出 metadata。'
                )
            ids = [s.get('id') for s in asset.get('shapes', [])]
            if len(ids) != len(set(ids)):
                raise ValueError('同一圖片的標註 ID 重複')
            for shape in asset.get('shapes', []):
                validate_shape(shape, width, height)
                if shape['label'] not in classes:
                    raise ValueError(f'類別「{shape["label"]}」未在專案類別清單')
                if not errors:
                    losses.extend(_shape_losses(shape, asset, format_key, tolerance))
            if format_key == 'classification' and len({s['label'] for s in asset.get('shapes', [])}) != 1:
                raise ValueError('圖片分類要求每張圖片只有一種標註類別；無類別／多類別圖片不能自動猜測')
        except (ValueError, OSError, KeyError, TypeError, cv2.error) as exc:
            errors.append(f'{asset.get("name", asset.get("id", "圖片"))}：{exc}')
    splits = {}
    try:
        splits = _split_plan(approved, format_key.startswith('yolo'))
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    excluded = Counter(a.get('review_state', 'pending') for a in assets if a.get('review_state') != 'approved')
    if excluded:
        warnings.append(f'只輸出已核准圖片；排除待審核 {excluded.get("pending", 0)} 張、已拒絕 {excluded.get("rejected", 0)} 張')
    if losses:
        warnings.append(f'{len(losses)} 項格式轉換會改變標註表示方式，匯出前必須確認')
    return {'valid': not errors, 'errors': errors, 'warnings': warnings, 'losses': losses,
            'stats': {'total': len(assets), 'approved': len(approved),
                      'pending': excluded.get('pending', 0), 'rejected': excluded.get('rejected', 0),
                      'shapes': sum(len(a.get('shapes', [])) for a in approved),
                      'classes': len(classes), 'batches': len({a.get('batch_id') for a in approved}),
                      'split_counts': dict(Counter(s for s in splits.values() if s))},
            'splits': splits}


def _export_name(asset, index):
    suffix = Path(asset.get('name') or asset['image_path']).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = Path(asset['image_path']).suffix.lower()
    return f'{index:06d}_{_identifier(asset["id"])[:16]}{suffix}'


def _verify_staging(directory, exported, format_key, classes, report):
    for asset in exported:
        path = _safe_child(directory, asset['image'])
        if _sha(path) != asset['sha256'] or _dimensions(path) != (asset['width'], asset['height']):
            raise ValueError('匯出圖片完整性驗證失敗')
    if format_key.startswith('yolo'):
        config = yaml.safe_load((directory / 'data.yaml').read_text('utf-8'))
        if config.get('names') != classes:
            raise ValueError('YOLO names 與類別 ID 不一致')
        actual_labels = set((directory / 'labels').rglob('*.txt'))
        expected_labels = set()
        count = 0
        for asset in exported:
            rel = Path(asset['image']).relative_to('images').with_suffix('.txt')
            label = directory / 'labels' / rel
            expected_labels.add(label)
            for line in label.read_text('utf-8').splitlines():
                parts = line.split()
                if (format_key == 'yolo_detection' and len(parts) != 5) or (format_key == 'yolo_segmentation' and (len(parts) < 7 or len(parts) % 2 != 1)):
                    raise ValueError('YOLO TXT 座標數量錯誤')
                if not 0 <= int(parts[0]) < len(classes) or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in parts[1:]):
                    raise ValueError('YOLO TXT 類別或座標錯誤')
                count += 1
        if expected_labels != actual_labels or count != report['output_shape_count']:
            raise ValueError('YOLO TXT 配對或物件數量不一致')
    elif format_key == 'coco':
        data = _json(directory / 'annotations.json')
        if len(data['images']) != len(exported) or len(data['annotations']) != report['source_shape_count']:
            raise ValueError('COCO 輸出數量不一致')
        for annotation in data['annotations']:
            segmentation = annotation.get('segmentation')
            if isinstance(segmentation, dict):
                image = data['images'][annotation['image_id']-1]
                decode_rle(segmentation['counts'], image['width'], image['height'])


def export_project(snapshot: dict, output_dir: str | Path, format_key='native', version='v1',
                   tolerance=0, acknowledge_loss=False) -> dict:
    snapshot = deepcopy(snapshot)
    validation = validate_project(snapshot, format_key, tolerance)
    if not validation['valid']:
        raise ValueError('驗證失敗：' + '；'.join(validation['errors']))
    if validation['losses'] and acknowledge_loss is not True:
        raise ValueError('轉換會改變標註；請先檢視驗證報告並明確確認 acknowledge_loss')
    if not isinstance(version, str) or not re.fullmatch(r'[\w.-]{1,64}', version) or version in {'.', '..'}:
        raise ValueError('版本僅可包含文字、數字、底線、連字號與點，最多 64 字元')
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    project_name = re.sub(r'[^\w.-]+', '-', str(snapshot.get('name') or 'dataset')).strip('.-')[:64] or 'dataset'
    name = f'{project_name}-{version}-{format_key}'
    destination, archive = root / name, root / f'{name}.zip'
    if destination.exists() or archive.exists():
        raise FileExistsError(f'輸出版本已存在：{name}；請使用新版本')
    stage = Path(tempfile.mkdtemp(prefix='.workbench-export-', dir=root))
    working, staged_zip = stage / name, stage / f'{name}.zip'
    working.mkdir()
    published_dir = published_zip = False
    try:
        approved = [a for a in snapshot['assets'] if a.get('review_state') == 'approved']
        classes = list(snapshot.get('classes', []))
        class_ids = {name: index for index, name in enumerate(classes)}
        exported, coco_images, coco_annotations, items = [], [], [], []
        output_shape_count = 0
        for index, asset in enumerate(approved, 1):
            filename = _export_name(asset, index)
            split = validation['splits'].get(asset['id'], asset.get('split', ''))
            relative = Path('images') / split / filename if format_key.startswith('yolo') else Path('images') / filename
            if format_key == 'classification':
                label_name = asset['shapes'][0]['label']
                class_directory = f'{class_ids[label_name]:04d}_' + (re.sub(r'[^\w.-]+', '-', label_name).strip('.-')[:60] or 'class')
                relative = Path('images') / split / class_directory / filename
            target = working / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset['image_path'], target)
            exported_asset = {k: deepcopy(v) for k, v in asset.items() if k not in {'image_path', 'url'}}
            exported_asset.update(image=relative.as_posix(), split=split)
            coordinates = _coordinate_metadata(asset['image_path'])
            exported_asset['pixel_coordinates'] = coordinates
            exported_asset['source'] = {**exported_asset.get('source', {}), 'pixel_coordinates': coordinates}
            exported.append(exported_asset)
            item = {'asset_id': asset['id'], 'source_sha256': asset['sha256'], 'image': relative.as_posix(),
                    'batch_id': asset['batch_id'], 'split': split, 'source': deepcopy(exported_asset['source']),
                    'pixel_coordinates': coordinates, 'shapes': []}
            width, height = asset['width'], asset['height']
            if format_key.startswith('yolo'):
                label = working / 'labels' / split / Path(filename).with_suffix('.txt')
                label.parent.mkdir(parents=True, exist_ok=True)
                lines = []
                for shape in asset['shapes']:
                    cid = class_ids[shape['label']]
                    if format_key == 'yolo_detection':
                        x, y, w, h = bounds(shape, width, height)
                        coords = [(x+w/2)/width, (y+h/2)/height, w/width, h/height]
                        lines.append(str(cid) + ' ' + ' '.join(f'{v:.10f}' for v in coords))
                        count = 1
                    else:
                        polygons, diagnostics = shape_polygons(shape, width, height, tolerance)
                        for points in polygons:
                            coords = [coordinate / (width if axis == 0 else height) for point in points for axis, coordinate in enumerate(point)]
                            lines.append(str(cid) + ' ' + ' '.join(f'{v:.10f}' for v in coords))
                        count = len(polygons)
                    item['shapes'].append({'shape_id': shape['id'], 'class_id': cid, 'class_name': shape['label'], 'output_count': count})
                label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
                output_shape_count += len(lines)
            elif format_key == 'coco':
                coco_images.append({'id': index, 'file_name': relative.as_posix(), 'width': width, 'height': height,
                                    'workbench_asset_id': asset['id'], 'batch_id': asset['batch_id'], 'split': split,
                                    'sha256': asset['sha256'], 'source': deepcopy(asset.get('source', {}))})
                for shape in asset['shapes']:
                    bbox = bounds(shape, width, height)
                    annotation = {'id': len(coco_annotations)+1, 'image_id': index,
                                  'category_id': class_ids[shape['label']]+1, 'bbox': bbox, 'iscrowd': 0,
                                  'workbench_shape_id': shape['id'], 'metadata': deepcopy(shape.get('metadata', {}))}
                    if shape['type'] == 'mask':
                        mask = decode_rle(shape['counts'], width, height)
                        annotation.update(segmentation={'size': [height, width], 'counts': encode_rle(mask)}, area=int(np.count_nonzero(mask)))
                    elif shape['type'] in {'polygon', 'obb'}:
                        annotation.update(segmentation=[[v for point in shape['points'] for v in point]],
                                          area=abs(float(cv2.contourArea(np.asarray(shape['points'], np.float32)))))
                    else:
                        annotation.update(segmentation=[], area=bbox[2]*bbox[3])
                    coco_annotations.append(annotation)
                    output_shape_count += 1
            elif format_key == 'labelme':
                shapes = []
                for shape in asset['shapes']:
                    common = {'label': shape['label'], 'flags': {}, 'description': '', 'group_id': shape['id'],
                              'workbench_shape_id': shape['id']}
                    if shape['type'] == 'rectangle':
                        x, y, w, h = bounds(shape, width, height)
                        shapes.append({**common, 'shape_type': 'rectangle', 'points': [[x,y],[x+w,y+h]]})
                    elif shape['type'] == 'mask':
                        polygons, _ = shape_polygons(shape, width, height, tolerance)
                        for component, points in enumerate(polygons):
                            shapes.append({**common, 'workbench_shape_id': f'{shape["id"]}:{component}', 'shape_type': 'polygon', 'points': points})
                    else:
                        shapes.append({**common, 'shape_type': 'polygon' if shape['type'] == 'obb' else shape['type'], 'points': shape['points']})
                _write_json(target.with_suffix('.json'), {'version': '5.0.1', 'flags': {}, 'imagePath': target.name,
                            'imageData': None, 'imageHeight': height, 'imageWidth': width, 'shapes': shapes,
                            'workbench': {'asset_id': asset['id'], 'batch_id': asset['batch_id'], 'split': split,
                                          'source': deepcopy(asset.get('source', {}))}})
                output_shape_count += len(shapes)
            else:
                output_shape_count += len(asset['shapes'])
            items.append(item)
        if format_key == 'native':
            _write_json(working / 'project.json', {'format': NATIVE_FORMAT, 'schema_version': 1,
                        'id': snapshot.get('id'), 'name': snapshot.get('name'), 'revision': snapshot.get('revision'),
                        'classes': classes, 'assets': exported})
        elif format_key in {'classification', 'jsonl'}:
            _write_json(working / 'dataset.json', {'format': f'vision-workbench-{format_key}', 'schema_version': 1,
                        'id': snapshot.get('id'), 'name': snapshot.get('name'), 'revision': snapshot.get('revision'),
                        'classes': classes, 'annotations': 'dataset.jsonl'})
            with (working / 'dataset.jsonl').open('w', encoding='utf-8') as handle:
                for asset in exported:
                    row = {'schema_version': 1, 'type': 'vision-workbench-asset', 'asset': asset}
                    if format_key == 'classification':
                        row['classification_label'] = asset['shapes'][0]['label']
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        elif format_key == 'coco':
            _write_json(working / 'annotations.json', {'info': {'description': 'Vision Workbench approved dataset'},
                        'images': coco_images, 'annotations': coco_annotations,
                        'categories': [{'id': i+1, 'name': name} for i, name in enumerate(classes)]})
        elif format_key.startswith('yolo'):
            split_names = set(validation['splits'].values())
            config = {split: f'images/{split}' for split in ('train', 'val', 'test') if split in split_names}
            config.update(nc=len(classes), names=classes)
            (working / 'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
            (working / 'classes.txt').write_text('\n'.join(classes) + ('\n' if classes else ''), encoding='utf-8')
        report = {'schema_version': 1, 'tool': 'Vision Workbench', 'project_id': snapshot.get('id'), 'project_revision': snapshot.get('revision'),
                  'format': format_key, 'version': version, 'created_at': datetime.now(timezone.utc).isoformat(),
                  'validation': validation, 'classes': classes, 'losses': validation['losses'],
                  'acknowledge_loss': acknowledge_loss, 'image_count': len(exported),
                  'source_shape_count': validation['stats']['shapes'], 'output_shape_count': output_shape_count,
                  'items': items}
        _verify_staging(working, exported, format_key, classes, report)
        report['verified'] = True
        _write_json(working / 'conversion_report.json', report)
        manifest = {'schema_version': 1, 'tool': 'Vision Workbench', 'format': format_key,
                    'files': [{'path': p.relative_to(working).as_posix(), 'sha256': _sha(p), 'bytes': p.stat().st_size}
                              for p in sorted(working.rglob('*')) if p.is_file()]}
        _write_json(working / 'manifest.json', manifest)
        with zipfile.ZipFile(staged_zip, 'x', compression=zipfile.ZIP_DEFLATED) as handle:
            for file in sorted(working.rglob('*')):
                if file.is_file():
                    handle.write(file, file.relative_to(working).as_posix())
        with zipfile.ZipFile(staged_zip) as handle:
            if handle.testzip() is not None:
                raise ValueError('ZIP CRC 驗證失敗')
        # Hard-link a completed file into its final name: atomic and no-overwrite
        # on NTFS and local POSIX filesystems. No links to source datasets exist.
        if os.name == 'nt':
            # Windows rename is no-overwrite and works on exFAT/network volumes.
            staged_zip.rename(archive)
        else:
            os.link(staged_zip, archive)
        published_zip = True
        if destination.exists():
            raise FileExistsError('輸出版本於作業期間已被建立')
        working.rename(destination)
        published_dir = True
        return {'path': str(destination), 'zip_path': str(archive), 'report': report,
                'format': format_key, 'version': version}
    except Exception:
        if published_zip and not published_dir:
            archive.unlink(missing_ok=True)
        raise
    finally:
        # This directory is created by this function under the chosen output root.
        if stage.resolve().is_relative_to(root):
            shutil.rmtree(stage, ignore_errors=True)
