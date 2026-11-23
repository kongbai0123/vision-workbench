"""Stage offline MM/Qwen candidates beside the owning Workbench project.

The worker has no access to the project's SQLite database. Its output can only
become a Workbench annotation through an explicit, revision-checked review.
"""
from hashlib import file_digest, sha256
from contextlib import suppress
import json
from math import isfinite
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import uuid

from .annotations import AnnotationService
from .jobs import JobCancelled
from .project_storage import ProjectStorage
from .store import ConflictError, dump, identifier, timestamp
from .training_engine import atomic_json, read_json


RESULT_SCHEMA = 'vision_workbench.annotation_assist.v1'
HISTORY_ACTION = 'annotation_assist_review'
REASONS = {
    'accepted': 'Qwen 建議採用，請核對類別及框界',
    'bad_extent': '目標可能存在，框界與標註規則不符',
    'different_object': '候選代表其他物件或類別',
    'uncertain_review': '證據不足或兩次覆核的結果不一致',
    'invalid_proposal': '候選分數或座標無效',
    'low_score_proposal': 'MM 分數低於候選門檻',
    'duplicate_proposal': '同一位置已有重疊候選',
    'better_supported_extent': '另一個候選獲得較強的框界支持',
    'competing_extents': '同一物件有多個框界，需確認範圍',
    'conflicting_classes': '重疊的候選有不同類別，需確認',
}


def _hash(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def _rectangle(raw, width, height):
    if not isinstance(raw, list) or len(raw) != 4 or any(
        type(value) not in (float, int) or not isfinite(value) for value in raw
    ):
        return None
    left, top, right, bottom = raw
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None
    return left, top, right - left, bottom - top


def _candidate(run_id, frame, source, labels):
    candidate_id = source.get('id')
    verdict = source.get('verdict')
    label = source.get('label')
    if (not isinstance(candidate_id, str) or not 0 < len(candidate_id) <= 128 or label not in labels
            or verdict not in {'accepted', 'rejected', 'unresolved'}):
        raise ValueError('推論候選的識別碼、類別或裁決無效')
    raw = source.get('xyxy')
    box = _rectangle(raw, frame['width'], frame['height'])
    score = source.get('score')
    if type(score) not in (float, int) or not isfinite(score) or not 0 <= score <= 1:
        score = None
    reason = source.get('reason_code') or verdict
    shape = None
    if box is not None:
        shape_id = sha256(f"{run_id}\0{frame['asset_id']}\0{candidate_id}".encode()).hexdigest()[:32]
        x, y, width, height = box
        shape = {'id': shape_id, 'type': 'rectangle', 'label': label, 'hidden': False,
                 'x': x, 'y': y, 'width': width, 'height': height,
                 'metadata': {'annotation_assist': {'run_id': run_id, 'candidate_id': candidate_id,
                                                    'generated_label': label, 'score': score}}}
    votes = source.get('votes', source.get('qwen_votes', source.get('qwen', [])))
    return {'id': candidate_id, 'label': label, 'verdict': verdict, 'reason_code': reason,
            'reason': source.get('reason') or REASONS.get(reason, reason), 'score': score,
            'xyxy': raw if box else None, 'shape': shape, 'stage': source.get('stage'), 'votes': votes}


def _counts(frames):
    return {
        'images': len(frames),
        'completed_images': sum(row['status'] in {'ready', 'review'} for row in frames),
        'review_frames': sum(bool(row['exceptions']) for row in frames),
        'pending_review_frames': sum(bool(row['exceptions']) and (row.get('review') or {}).get('status') != 'resolved'
                                     and row.get('available', True) for row in frames),
        'resolved_frames': sum((row.get('review') or {}).get('status') == 'resolved' for row in frames),
        'candidates': sum(len(row['candidates']) for row in frames),
        'accepted_boxes': sum(len(row['accepted_candidate_ids']) for row in frames),
    }


def _same_box(first, second):
    if second['type'] == 'mask':
        return False
    x, y = max(first['x'], second['x']), max(first['y'], second['y'])
    right = min(first['x'] + first['width'], second['x'] + second['width'])
    bottom = min(first['y'] + first['height'], second['y'] + second['height'])
    intersection = max(0, right - x) * max(0, bottom - y)
    union = first['width'] * first['height'] + second['width'] * second['height'] - intersection
    return union > 0 and intersection / union >= .85


def _candidate_source(shape):
    metadata = shape.get('metadata')
    if not isinstance(metadata, dict) or not isinstance(metadata.get('annotation_assist'), dict):
        return {}
    return metadata['annotation_assist']


class AnnotationAssist:
    def __init__(self, root, store, jobs, *, runner=None):
        self.root = Path(root)
        self.store = store
        self.storage = ProjectStorage(root, store)
        self.jobs = jobs
        self.runner = runner

    def _path(self, pid, run_id, *, create=False):
        return self.storage.area(pid, f'integrations/annotation-assist/{identifier(run_id)}', create=create)

    def list_runs(self, pid):
        self.store.get_project(pid, include_assets=False)
        area = self.storage.area(pid, 'integrations/annotation-assist')
        runs = []
        for file in area.glob('*/batch.json'):
            run = read_json(file)
            runs.append(self._public(run, file.parent))
        return {'runs': sorted(runs, key=lambda row: (row['created_at'], row['id']), reverse=True)}

    def _public(self, run, directory):
        row = {key: value for key, value in run.items() if key != 'frames'}
        job_file = directory / 'job.json'
        if job_file.is_file():
            row['job_id'] = read_json(job_file)['id']
        if row['status'] == 'queued' and row.get('job_id'):
            try:
                job = self.jobs.get(row['job_id'])
            except FileNotFoundError:
                row.update(status='interrupted', message='背景作業已中斷，請重新啟動批次')
            else:
                row.update(status=job['state'], message=job['message'], progress=job['progress'])
                if job.get('error'):
                    row['error'] = job['error']
        return row

    def _record(self, pid, run_id):
        directory = self._path(pid, run_id)
        record = directory / 'batch.json'
        if not record.is_file():
            raise FileNotFoundError('找不到 AI 標註批次')
        return read_json(record), directory

    def start(self, pid, asset_ids, classes):
        if not isinstance(asset_ids, list) or not asset_ids:
            raise ValueError('請選擇本次要處理的影像')
        requested = list(dict.fromkeys(identifier(value) for value in asset_ids))
        project = self.store.get_project(pid)
        frozen_classes = self._classes(classes, project['classes'])
        assets = {asset['id']: asset for asset in project['assets']}
        if any(aid not in assets for aid in requested):
            raise FileNotFoundError('有選取的影像已不在專案內')
        frames = []
        for aid in requested:
            asset = assets[aid]
            frames.append({'asset_id': aid, 'name': asset['name'], 'width': asset['width'], 'height': asset['height'],
                           'base_revision': asset['revision'], 'image_sha256': asset['sha256'],
                           'status': 'pending', 'candidates': [], 'accepted_candidate_ids': [], 'exceptions': []})
        run_id = uuid.uuid4().hex
        run = {'id': run_id, 'project_id': pid, 'status': 'queued', 'created_at': timestamp(),
               'classes': frozen_classes, 'frames': frames, 'summary': _counts(frames)}
        atomic_json(self._path(pid, run_id, create=True) / 'batch.json', run)
        try:
            job = self.jobs.submit('annotation-assist', lambda progress: self._execute(run, progress))
        except Exception as exc:
            atomic_json(self._path(pid, run_id) / 'batch.json', dict(run, status='failed', error=str(exc)))
            raise
        atomic_json(self._path(pid, run_id) / 'job.json', {'id': job['id']})
        return dict(job, run_id=run_id)

    @staticmethod
    def _classes(classes, existing):
        if not isinstance(classes, list) or not 1 <= len(classes) <= 100:
            raise ValueError('請為本次作業指定 1 至 100 個專案類別')
        seen, result = set(), []
        for value in classes:
            if not isinstance(value, dict) or not isinstance(value.get('label'), str):
                raise ValueError('類別設定無效')
            label = value['label']
            prompt = value.get('prompt', label)
            description = value.get('description', '')
            if label not in existing or label in seen or len(label) > 128:
                raise ValueError('批次類別必須是現有、互不重複的專案類別')
            if not isinstance(prompt, str) or len(prompt) > 512 or not isinstance(description, str) or len(description) > 4000:
                raise ValueError('定位提示或判斷規則過長或無效')
            seen.add(label)
            result.append({'id': label, 'label': label, 'prompt': prompt.strip() or label,
                           'description': description.strip()})
        return result

    def _execute(self, run, progress):
        run_id, pid = run['id'], run['project_id']
        with TemporaryDirectory(prefix='vw-annotation-assist-') as temporary:
            staging = Path(temporary)
            inputs = staging / 'input'
            inputs.mkdir()
            output = staging / 'inference.json'
            try:
                manifest = self._stage(run, inputs, progress)
                runner = self.runner
                if runner is None:
                    from .annotation_assist_runtime import run_batch
                    runner = run_batch
                isolation = runner(self.root, manifest, output, progress)
                source = read_json(output)
                frames = self._result_frames(run, manifest, source)
                complete = dict(run, status='complete', finished_at=timestamp(), frames=frames, summary=_counts(frames),
                                method=source.get('method', {}), timing=source.get('timing', {}), isolation=isolation)
                progress('保存候選與例外；等待人工檢視', 98, phase='save')
                atomic_json(self._path(pid, run_id) / 'batch.json', complete)
                return {'run_id': run_id, 'summary': complete['summary']}
            except Exception as exc:
                status = 'cancelled' if isinstance(exc, JobCancelled) else 'failed'
                failed = dict(run, status=status, error=str(exc), finished_at=timestamp())
                with suppress(OSError):
                    atomic_json(self._path(pid, run_id) / 'batch.json', failed)
                raise
            finally:
                with suppress(OSError):
                    folder = self._path(pid, run_id)
                    for kind in ('worker.log', 'launcher.log', 'isolation.json'):
                        diagnostic = output.with_name(f'{output.stem}-{kind}')
                        if diagnostic.is_file():
                            shutil.copyfile(diagnostic, folder / diagnostic.name)

    def _stage(self, run, staging, progress):
        image_dir = staging / 'images'
        image_dir.mkdir()
        rows = []
        for frame in run['frames']:
            aid = frame['asset_id']
            progress(f"準備原圖 {len(rows) + 1}／{len(run['frames'])}", 0, phase='stage')
            image = self.store.image_path(run['project_id'], aid)
            relative = f'images/{aid}{image.suffix}'
            copied = staging / relative
            shutil.copyfile(image, copied)
            if _hash(copied) != frame['image_sha256']:
                raise ConflictError('原圖內容與批次快照不同，請重新選取影像')
            rows.append({'id': aid, 'image': relative, 'image_sha256': frame['image_sha256'],
                         'width': frame['width'], 'height': frame['height'], 'base_revision': frame['base_revision']})
        manifest = staging / 'manifest.json'
        atomic_json(manifest, {'job_id': run['id'], 'config': {'classes': run['classes'],
                     'settings': {'candidate_limit': 5, 'retry_regions': 2}}, 'frames': rows})
        return manifest

    @staticmethod
    def _result_frames(run, manifest, result):
        if (result.get('schema') != RESULT_SCHEMA or result.get('job_id') != run['id']
                or result.get('status') != 'complete' or result.get('manifest_sha256') != _hash(manifest)):
            raise ValueError('推論結果未完成，或和批次輸入快照不一致')
        raw_frames = result['frames']
        by_id = {frame['id']: frame for frame in raw_frames}
        if len(by_id) != len(raw_frames) or set(by_id) != {frame['asset_id'] for frame in run['frames']}:
            raise ValueError('推論結果的影像清單與批次設定不一致')
        rows = []
        labels = {entry['label'] for entry in run['classes']}
        for snapshot in run['frames']:
            raw = by_id[snapshot['asset_id']]
            if raw['image_sha256'] != snapshot['image_sha256'] or raw['status'] not in {'ready', 'review'}:
                raise ValueError('推論結果的影像識別或例外狀態無效')
            candidates = [_candidate(run['id'], snapshot, value, labels) for value in raw['candidates']]
            by_candidate = {candidate['id']: candidate for candidate in candidates}
            accepted = [box['candidate_id'] for box in raw['boxes']]
            if (len(by_candidate) != len(candidates) or len(set(accepted)) != len(accepted) or any(
                cid not in by_candidate or by_candidate[cid]['verdict'] != 'accepted'
                or not by_candidate[cid]['shape'] for cid in accepted
            )):
                raise ValueError('推論結果包含不一致的候選與採用清單')
            rows.append(dict(snapshot, status=raw['status'], candidates=candidates, accepted_candidate_ids=accepted,
                             exceptions=raw['exceptions'], confidence=raw.get('confidence'),
                             scene_scan=raw.get('scene_scan'), retries=raw.get('retries', []),
                             shadow_qc=raw.get('shadow_qc')))
        return rows

    def detail(self, pid, run_id):
        run, directory = self._record(pid, run_id)
        result = self._public(run, directory)
        with self.store.connection(pid) as db:
            current = {row['id']: row for row in db.execute('SELECT id,revision,sha256,shapes FROM assets')}
            reviews = {}
            for row in db.execute("SELECT asset_id,data,created_at,revision,annotation_hash FROM history "
                                  "WHERE action=? AND json_extract(data,'$.run_id')=? ORDER BY id",
                                  (HISTORY_ACTION, run_id)):
                reviews[row['asset_id']] = row
            frames = []
            for frame in run['frames']:
                aid = frame['asset_id']
                live = current.get(aid)
                available = live is not None and live['sha256'] == frame['image_sha256']
                review = self._review_detail(reviews[aid], live) if aid in reviews and available else None
                revision = live['revision'] if available else None
                frames.append(dict(frame, available=available, revision=revision, review=review,
                                   stale=revision != frame['base_revision']))
        return dict(result, frames=frames, summary=_counts(frames))

    @staticmethod
    def _review_detail(record, live):
        value = json.loads(record['data'])
        value.pop('annotation_hash', None)
        intact = record['annotation_hash'] == sha256(live['shapes'].encode('utf-8')).hexdigest()
        return dict(value, status='resolved' if intact else 'outdated', revision=record['revision'],
                    reviewed_at=record['created_at'])

    def review(self, pid, run_id, payload):
        run, _ = self._record(pid, run_id)
        aid, revision = identifier(payload.get('asset_id')), payload.get('revision')
        frame = next((value for value in run['frames'] if value['asset_id'] == aid), None)
        if run['status'] != 'complete' or frame is None:
            raise ValueError('僅能覆核本批次已完成的影像')
        selected, decisions, note = payload.get('candidate_ids', []), payload.get('decisions', []), payload.get('note', '')
        candidates = {value['id']: value for value in frame['candidates']}
        if (not isinstance(selected, list) or any(not isinstance(cid, str) or cid not in candidates
                or not candidates[cid]['shape'] for cid in selected)):
            raise ValueError('請選擇本批次中座標有效的候選框')
        if (not isinstance(decisions, list) or any(not isinstance(entry, dict)
                or not isinstance(entry.get('candidate_id'), str) or entry['candidate_id'] not in candidates
                or not isinstance(entry.get('reason'), str) or len(entry['reason']) > 500 for entry in decisions)
                or not isinstance(note, str) or len(note) > 2000):
            raise ValueError('候選處置或備註無效')
        with self.store.connection(pid, write=True) as db:
            row = db.execute('SELECT * FROM assets WHERE id=?', (aid,)).fetchone()
            if row is None or row['sha256'] != frame['image_sha256']:
                raise FileNotFoundError('原圖已不在本專案')
            if type(revision) is not int or revision != row['revision']:
                raise ConflictError('圖片已另有修改，請載入最新版本後覆核')
            shapes, adopted, skipped = json.loads(row['shapes']), [], []
            for cid in dict.fromkeys(selected):
                shape = candidates[cid]['shape']
                if any(old['id'] == shape['id'] or _same_box(shape, old) for old in shapes):
                    skipped.append(cid)
                else:
                    shapes.append(shape)
                    adopted.append(cid)
            if adopted:
                AnnotationService(self.store).commit(pid, [({'id': aid, 'revision': revision}, shapes)],
                                                     source=HISTORY_ACTION, db=db)
                row = db.execute('SELECT * FROM assets WHERE id=?', (aid,)).fetchone()
                shapes = json.loads(row['shapes'])
            retained = [source['candidate_id'] for shape in shapes if (source := _candidate_source(shape)).get('run_id') == run_id
                        and source.get('candidate_id') in candidates]
            review = {'run_id': run_id, 'actor': 'workbench_user', 'decisions': decisions, 'note': note,
                      'adopted_candidate_ids': adopted, 'retained_candidate_ids': retained,
                      'skipped_candidate_ids': skipped}
            self.store._history(db, aid, row['revision'], HISTORY_ACTION, dict(review, shapes=shapes))
            asset = self.store._asset(row, pid, detail=True, internal=False, db=db)
        result = self.detail(pid, run_id)
        current_review = next(item['review'] for item in result['frames'] if item['asset_id'] == aid)
        return {'asset': asset, 'review': current_review, 'summary': result['summary']}
