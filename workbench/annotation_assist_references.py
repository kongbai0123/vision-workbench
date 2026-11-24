"""Build bounded visual references from saved, human-confirmed annotations."""
from collections import defaultdict
from hashlib import file_digest, sha256
import json
from math import ceil, floor
from pathlib import Path
from uuid import uuid4

from PIL import Image

from .smart_splitting import source_groups
from .store import ConflictError, dump
from .training_engine import read_json


MIN_REFERENCE_GROUPS = 2
MAX_REFERENCES_PER_CLASS = 3
MAX_REFERENCES_PER_BATCH = 12
MAX_REFERENCE_SIDE = 512


def _annotation_hash(shapes):
    return sha256(dump(shapes).encode('utf-8')).hexdigest()


def _confirmed_by_review(db, asset):
    if asset['review_state'] != 'approved':
        return False
    shape_hash = _annotation_hash(asset['shapes'])
    rows = db.execute("SELECT action,json_extract(data,'$.review_state') FROM history "
                      "WHERE asset_id=? AND revision<=? AND annotation_hash=? AND action IN ('import','review')",
                      (asset['id'], asset['revision'], shape_hash)).fetchall()
    if any(action == 'review' and state == 'approved' for action, state in rows):
        return True
    source = asset.get('source', {})
    sample = source.get('sample')
    return (source.get('format') == 'coco' and isinstance(sample, dict) and sample.get('review_status') == 'human_verified'
            and sample.get('image_sha256') == asset['sha256']
            and any(action == 'import' and state == 'approved' for action, state in rows))


def _manually_saved_shapes(db, asset, by_id):
    previous, confirmed = {}, {}
    for editor, serialized in db.execute(
            "SELECT json_extract(h.data,'$.editor'),b.shapes FROM history h "
            "JOIN annotation_blobs b ON b.hash=h.annotation_hash "
            "WHERE h.asset_id=? AND h.revision<=? AND h.action IN ('import','edit') ORDER BY h.id",
            (asset['id'], asset['revision'])):
        current = {shape['id']: shape for shape in json.loads(serialized) if shape.get('id') in by_id}
        for shape_id, shape in current.items():
            if shape == previous.get(shape_id):
                continue
            metadata = shape.get('metadata', {})
            if not isinstance(metadata, dict):
                continue
            assist = metadata.get('annotation_assist')
            manual = (metadata.get('source') == 'manual' and isinstance(assist, dict)
                      and assist.get('human_confirmed') is True)
            if editor and (manual or editor == 'builtin' and not metadata.get('source')):
                confirmed[shape_id] = shape
            else:
                confirmed.pop(shape_id, None)
        for shape_id in previous.keys() - current.keys():
            confirmed.pop(shape_id, None)
        previous = current
    return [shape for shape_id, shape in by_id.items() if confirmed.get(shape_id) == shape]


def _confirmed_rectangles(db, asset, labels):
    if asset['review_state'] == 'rejected':
        return [], None
    by_id = {shape['id']: shape for shape in asset['shapes']
             if shape.get('type') == 'rectangle' and shape['label'] in labels}
    if not by_id:
        return [], None
    if _confirmed_by_review(db, asset):
        return list(by_id.values()), 'reviewed'
    return _manually_saved_shapes(db, asset, by_id), 'manual_edit'


def _xyxy(shape):
    return [shape['x'], shape['y'], shape['x'] + shape['width'], shape['y'] + shape['height']]


def _sample_sources(sources, count):
    if not count:
        return []
    representatives = []
    for group in sources.values():
        ordered = sorted(group.values(), key=lambda row: row['box_area_fraction'])
        representatives.append(ordered[len(ordered) // 2])
    representatives.sort(key=lambda row: row['box_area_fraction'])
    positions = (0, -1) if count == MIN_REFERENCE_GROUPS else (0, len(representatives) // 2, -1)
    return [dict(representatives[position], id=uuid4().hex) for position in positions]


def _source_group_exclusions(project, datasets, targets):
    target_ids = set(targets)
    evaluation = {asset['id'] for asset in project['assets'] if asset.get('split') in {'val', 'test'}}
    members = [{key: asset[key] for key in ('id', 'sha256', 'source', 'batch_id') if key in asset}
               for asset in project['assets']]
    overrides = (project.get('split_plan') or {}).get('options', {}).get('group_overrides', {})
    bridges = {asset['id']: overrides[asset['id']] for asset in members if overrides.get(asset['id'])}
    # An immutable dataset may still be used for evaluation after a live split is changed.
    for path in sorted(datasets.glob('D[0-9]*/manifest.json')):
        manifest = read_json(path)
        frozen_overrides = (manifest.get('split_plan') or {}).get('options', {}).get('group_overrides', {})
        for asset in manifest['assets']:
            if asset.get('split') not in {'val', 'test'}:
                continue
            frozen_id = f"{path.parent.name}:{asset['asset_id']}"
            evaluation.add(frozen_id)
            members.append(dict(id=frozen_id, sha256=asset['sha256'],
                                source=asset.get('source', {}), batch_id=asset.get('batch_id')))
            if frozen_overrides.get(asset['asset_id']):
                bridges[frozen_id] = frozen_overrides[asset['asset_id']]
    # Manual grouping adds a protection. Keep the raw capture group as well.
    manual_groups = {}
    for asset in list(members):
        if asset['id'] in bridges:
            bridge_id = f"manual:{asset['id']}"
            members.append(dict(id=bridge_id, sha256=asset['sha256']))
            manual_groups[bridge_id] = bridges[asset['id']]
    groups, excluded = {}, set()
    for group in source_groups(members, manual_groups):
        ids = set(group['asset_ids'])
        if ids & evaluation and ids & target_ids:
            raise ValueError('參考人工標註重跑不能包含 Validation／Test 影像或同來源群組，請調整影像範圍。')
        if ids & (target_ids | evaluation):
            excluded.update(ids)
        groups.update(dict.fromkeys(ids, group['id']))
    return groups, excluded


def build_reference_plan(store, project, datasets, targets, labels):
    groups, excluded = _source_group_exclusions(project, datasets, targets)
    selected_labels, target_ids = set(labels), set(targets)
    evidence = defaultdict(lambda: defaultdict(dict))
    protected = {}
    with store.connection(project['id']) as db:
        if db.execute('SELECT revision FROM project').fetchone()[0] != project['revision']:
            raise ConflictError('規劃人工參考期間專案已變動，請重新開始批次。')
        for asset in project['assets']:
            asset_id = asset['id']
            if asset_id in excluded and asset_id not in target_ids:
                continue
            shapes, confirmation = _confirmed_rectangles(db, asset, selected_labels)
            if asset_id in target_ids:
                protected[asset_id] = [{'id': shape['id'], 'label': shape['label'], 'xyxy': _xyxy(shape)}
                                       for shape in shapes]
                continue
            group = groups[asset_id]
            for shape in shapes:
                xyxy = _xyxy(shape)
                key = (asset['sha256'], tuple(xyxy))
                evidence[shape['label']][group].setdefault(key, dict(
                    asset_id=asset_id, source_group=group, shape_id=shape['id'], label=shape['label'],
                    asset_revision=asset['revision'], annotation_revision=asset['annotation_revision'], image_sha256=asset['sha256'],
                    xyxy=xyxy, confirmation=confirmation,
                    box_area_fraction=(shape['width'] * shape['height']) / (asset['width'] * asset['height'])))
    allocations, remaining = {}, MAX_REFERENCES_PER_BATCH
    for label in labels:
        if len(evidence[label]) >= MIN_REFERENCE_GROUPS and remaining >= MIN_REFERENCE_GROUPS:
            allocations[label] = MIN_REFERENCE_GROUPS
            remaining -= MIN_REFERENCE_GROUPS
    for label in allocations:
        if remaining and len(evidence[label]) >= MAX_REFERENCES_PER_CLASS:
            allocations[label] = MAX_REFERENCES_PER_CLASS
            remaining -= 1
    summary, references = {}, []
    for label in labels:
        sources = evidence[label]
        count = allocations.get(label, 0)
        summary[label] = dict(eligible=sum(map(len, sources.values())), independent_groups=len(sources),
                              selected=count, status='ready' if count else 'insufficient')
        if not count:
            summary[label]['reason'] = ('not_enough_independent_groups' if len(sources) < MIN_REFERENCE_GROUPS
                                        else 'batch_reference_limit')
        references.extend(_sample_sources(sources, count))
    return summary, references, protected


def stage_reference_crops(store, pid, staging, sources, progress):
    staged = []
    for reference in sources:
        progress('準備人工參考框', 0, phase='stage')
        asset = store.get_asset(pid, reference['asset_id'], internal=True)
        if (asset['revision'] != reference['asset_revision']
                or asset['annotation_revision'] != reference['annotation_revision']):
            raise ConflictError('人工參考標註已變動，請重新開始批次。')
        path = Path(asset['image_path'])
        with path.open('rb') as content:
            if file_digest(content, 'sha256').hexdigest() != reference['image_sha256']:
                raise ConflictError('人工參考圖片已變動，請重新開始批次。')
            content.seek(0)
            with Image.open(content) as original:
                x1, y1, x2, y2 = reference['xyxy']
                crop = original.crop((floor(x1), floor(y1), min(ceil(x2), original.width),
                                      min(ceil(y2), original.height))).convert('RGB')
        crop.thumbnail((MAX_REFERENCE_SIDE, MAX_REFERENCE_SIDE), Image.Resampling.LANCZOS)
        relative = f"references/{reference['id']}.png"
        destination = staging / relative
        destination.parent.mkdir(exist_ok=True)
        crop.save(destination, 'PNG')
        staged.append(dict(id=reference['id'], label=reference['label'], image=relative,
                           image_sha256=sha256(destination.read_bytes()).hexdigest(),
                           width=crop.width, height=crop.height))
    return staged
