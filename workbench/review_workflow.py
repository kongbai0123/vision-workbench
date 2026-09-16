"""Review exclusions, recoverable removal and content-based import preview."""
import base64
import io
import json
import secrets
import threading
import time

import cv2
import numpy as np
from PIL import Image

from .store import ConflictError, dump, identifier, timestamp


def image_quality(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        image.thumbnail((640, 640))
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        image.thumbnail((240, 160))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=75)
    return {'blur_score': round(score, 2), 'suspected_blur': score < 80,
            'thumbnail': 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')}


class ReviewWorkflow:
    def __init__(self, store):
        self.store = store
        self.previews = {}
        self.lock = threading.Lock()

    def preview(self, pid, paths):
        from .pipeline import import_sources, _sha
        self.store.get_project(pid, include_assets=False)
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            raise ValueError('請選擇有效路徑')
        parsed = import_sources(paths)
        items = []
        for index, record in enumerate(parsed['records']):
            quality = image_quality(record['path'])
            record['expected_import_sha'] = _sha(record['path'])
            record['source'] = {**record.get('source', {}), 'quality': {k:v for k,v in quality.items() if k != 'thumbnail'}}
            items.append({'id': str(index), 'name': record['name'], 'shape_count': len(record['shapes']), **quality})
        token = secrets.token_urlsafe(24)
        with self.lock:
            now = time.monotonic()
            self.previews = {k:v for k,v in self.previews.items() if now-v[0] < 3600}
            if len(self.previews) >= 8:
                self.previews.pop(next(iter(self.previews)))
            self.previews[token] = (now, pid, parsed['records'])
        return {'token': token, 'items': items, 'issues': parsed['issues']}

    def commit_import(self, pid, token, selected):
        from .pipeline import _sha
        with self.lock:
            entry = self.previews.get(token)
            if not entry or entry[1] != pid or time.monotonic()-entry[0] >= 3600:
                raise ValueError('預覽已過期，請重新選擇匯入來源')
            if not isinstance(selected, list) or not selected:
                raise ValueError('請至少勾選一張圖片')
            valid = {str(i): r for i,r in enumerate(entry[2])}
            if any(not isinstance(i, str) or i not in valid for i in selected):
                raise ValueError('選取項目無效')
            records = [dict(valid[i]) for i in dict.fromkeys(selected)]
            for record in records:
                if _sha(record['path']) != record.pop('expected_import_sha'):
                    raise ConflictError('預覽後來源圖片已變更，請重新預覽')
            result = self.store.add_assets(pid, records)
            self.previews.pop(token, None)
            return result

    @staticmethod
    def _trash_schema(db):
        db.execute('CREATE TABLE IF NOT EXISTS review_trash(id TEXT PRIMARY KEY, removed_at TEXT NOT NULL, data TEXT NOT NULL)')

    def trash(self, pid, ids, revisions, restore=False):
        if not isinstance(ids, list) or not ids or not isinstance(revisions, dict):
            raise ValueError('請選擇圖片並提供版本')
        with self.store.connection(pid, write=True) as db:
            self._trash_schema(db)
            for aid in dict.fromkeys(ids):
                identifier(aid)
                if restore:
                    saved = db.execute('SELECT data FROM review_trash WHERE id=?', (aid,)).fetchone()
                    if not saved:
                        raise FileNotFoundError('垃圾桶項目不存在')
                    row = json.loads(saved['data'])
                    if db.execute('SELECT 1 FROM assets WHERE id=? OR sha256=?', (aid, row['sha256'])).fetchone():
                        raise ConflictError('專案已有相同圖片，請先處理重複項目後再還原')
                    if revisions.get(aid) != row['revision']:
                        raise ConflictError('垃圾桶項目已變更')
                    row['revision'] += 1
                    row['review_state'] = 'pending'
                    row['updated_at'] = timestamp()
                    source = json.loads(row['source'])
                    source.pop('review', None)
                    row['source'] = dump(source)
                    columns = list(row)
                    db.execute(f"INSERT INTO assets ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", [row[k] for k in columns])
                    db.execute('DELETE FROM review_trash WHERE id=?', (aid,))
                else:
                    record = db.execute('SELECT * FROM assets WHERE id=?', (aid,)).fetchone()
                    if not record:
                        raise FileNotFoundError('圖片不存在')
                    row = dict(record)
                    if revisions.get(aid) != row['revision']:
                        raise ConflictError('圖片已有新版本，請重新載入')
                    db.execute('INSERT INTO review_trash VALUES(?,?,?)', (aid, timestamp(), dump(row)))
                    db.execute('DELETE FROM assets WHERE id=?', (aid,))
                self.store._history(db, aid, row['revision'], 'trash_restore' if restore else 'trash', {'review_state': row['review_state']})
            self.store._touch(db)
        return self.store.get_project(pid)

    def list_trash(self, pid):
        with self.store.connection(pid, write=True) as db:
            self._trash_schema(db)
            return [{'id': r['id'], 'removed_at': r['removed_at'], 'name': json.loads(r['data'])['name'],
                     'revision': json.loads(r['data'])['revision']} for r in db.execute('SELECT * FROM review_trash ORDER BY removed_at DESC')]

    def quality(self, pid):
        project = self.store.get_project(pid, internal=True)
        results = {a['id']: {k:v for k,v in image_quality(a['image_path']).items() if k != 'thumbnail'} for a in project['assets']}
        with self.store.connection(pid, write=True) as db:
            for aid, quality in results.items():
                row = db.execute('SELECT source FROM assets WHERE id=?', (aid,)).fetchone()
                if row:
                    source = json.loads(row['source'])
                    source['quality'] = quality
                    db.execute('UPDATE assets SET source=? WHERE id=?', (dump(source), aid))
            self.store._touch(db)
        return self.store.get_project(pid)
