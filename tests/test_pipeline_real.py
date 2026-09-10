"""Explicit real-data acceptance runner (not run by unittest discovery).

Usage: python tests/test_pipeline_real.py SOURCE --report REPORT.json
Inputs remain read-only. All generated images and datasets use a temporary copy.
Only the aggregate verification report is retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer_core.geometry import decode_rle
from workbench.pipeline import export_project, import_sources, validate_project
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(source, report_path):
    began = time.perf_counter()
    imported = import_sources([source])
    errors = [i for i in imported['issues'] if i['level'] == 'error']
    if errors or not imported['records']:
        raise ValueError(f'Import failed: {errors}')
    originals = {r['path']: sha(r['path']) for r in imported['records']}
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'source': str(Path(source).resolve()),
              'source_images': len(imported['records']), 'import_issues': imported['issues'],
              'formats': [], 'source_images_unchanged': False, 'success': False}
    with tempfile.TemporaryDirectory(prefix='workbench-real-acceptance-') as temporary:
        root = Path(temporary)
        images = root/'images'
        images.mkdir()
        assets = []
        for index, record in enumerate(imported['records']):
            destination = images / f'{index:06d}{Path(record["path"]).suffix}'
            shutil.copyfile(record['path'], destination)
            asset = {k: deepcopy(v) for k, v in record.items() if k != 'path'}
            asset.update(id=f'asset-{index}', revision=1, image_path=str(destination), sha256=sha(destination))
            assets.append(asset)
        snapshot = {'id': 'real-acceptance', 'name': 'verified-real-acceptance', 'revision': 1,
                    'classes': sorted({s['label'] for a in assets for s in a['shapes']}), 'assets': assets}
        source_hashes = Counter(a['sha256'] for a in assets if a['review_state'] == 'approved')
        report['approved_images'] = sum(a['review_state'] == 'approved' for a in assets)
        report['source_shapes'] = sum(len(a['shapes']) for a in assets)
        report['batches'] = len({a['batch_id'] for a in assets})
        report['splits'] = dict(Counter(a['split'] for a in assets))
        for kind in ('native', 'coco', 'yolo_detection', 'yolo_segmentation', 'labelme', 'classification', 'jsonl'):
            started = time.perf_counter()
            validation = validate_project(snapshot, kind)
            if not validation['valid']:
                raise ValueError(f'{kind}: {validation["errors"]}')
            output = export_project(snapshot, root/'exports', kind, acknowledge_loss=True)
            restored = import_sources([output['path']])
            if any(i['level'] == 'error' for i in restored['issues']):
                raise ValueError(f'{kind} reimport: {restored["issues"]}')
            if Counter(sha(r['path']) for r in restored['records']) != source_hashes:
                raise ValueError(f'{kind} image hashes/count changed')
            exact_masks = None
            if kind in {'native', 'coco', 'jsonl', 'classification'}:
                original_by_hash = {a['sha256']: a for a in assets}
                exact_masks = 0
                for record in restored['records']:
                    original = original_by_hash[sha(record['path'])]
                    shapes = {s['id']: s for s in record['shapes']}
                    if set(shapes) != {s['id'] for s in original['shapes']}:
                        raise ValueError(f'{kind} stable IDs changed')
                    for shape in original['shapes']:
                        if shape['type'] == 'mask':
                            if not np.array_equal(decode_rle(shape['counts'], original['width'], original['height']),
                                                  decode_rle(shapes[shape['id']]['counts'], record['width'], record['height'])):
                                raise ValueError(f'{kind} precise mask/hole changed')
                            exact_masks += 1
            entry = {'format': kind, 'valid': True, 'image_hashes_verified': True,
                     'images': len(restored['records']), 'output_shapes': output['report']['output_shape_count'],
                     'loss_count': len(validation['losses']), 'exact_masks_verified': exact_masks,
                     'duration_seconds': round(time.perf_counter()-started, 3), 'reimport_issues': restored['issues'],
                     'export_report': output['report']}
            report['formats'].append(entry)
            print(f'{kind}: {entry["images"]} images, {entry["output_shapes"]} shapes, {entry["duration_seconds"]} seconds', flush=True)
        report['source_images_unchanged'] = all(sha(path) == digest for path, digest in originals.items())
        if not report['source_images_unchanged']:
            raise ValueError('Original source bytes changed')
    report['success'] = True
    report['duration_seconds'] = round(time.perf_counter()-began, 3)
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'PASS. Report: {report_path}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    run(args.source, args.report)
