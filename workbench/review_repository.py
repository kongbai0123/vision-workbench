"""Transactional trash repository owned by ProjectStore."""
import json
from .store import ConflictError, dump, identifier, timestamp


class ReviewRepository:
    def __init__(self, store):
        self.store = store

    def trash(self, pid, ids, revisions, restore=False):
        if not isinstance(ids, list) or not ids or not isinstance(revisions, dict):
            raise ValueError('請選擇圖片並提供版本')
        with self.store.connection(pid, write=True) as db:
            for aid in dict.fromkeys(ids):
                identifier(aid)
                if restore:
                    saved = db.execute('SELECT data FROM review_trash WHERE id=?', (aid,)).fetchone()
                    if not saved:
                        raise FileNotFoundError('垃圾桶項目不存在')
                    row = json.loads(saved['data'])
                    if not (self.store.directory(pid) / 'images' / row['image_file']).is_file():
                        raise FileNotFoundError('原圖檔案遺失，請先從備份復原；垃圾桶項目已保留')
                    if db.execute('SELECT 1 FROM assets WHERE id=? OR sha256=?', (aid, row['sha256'])).fetchone():
                        raise ConflictError('專案已有相同圖片，請先處理重複項目後再還原')
                    if revisions.get(aid) != row['revision']:
                        raise ConflictError('垃圾桶項目已變更')
                    row['revision'] += 1
                    row['review_state'] = 'pending'
                    row['updated_at'] = timestamp()
                    source = json.loads(row['source'])
                    source.pop('review', None)
                    db.execute('DELETE FROM asset_review WHERE asset_id=?', (aid,))
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
        with self.store.connection(pid) as db:
            return [{'id': r['id'], 'removed_at': r['removed_at'], 'name': json.loads(r['data'])['name'],
                     'revision': json.loads(r['data'])['revision']} for r in db.execute('SELECT * FROM review_trash ORDER BY removed_at DESC')]

