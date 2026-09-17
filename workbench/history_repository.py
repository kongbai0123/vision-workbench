"""Content-addressed annotation history owned by the project transaction."""
import hashlib
import json


class HistoryRepository:
    @staticmethod
    def append(db, asset_id, revision, action, data):
        from .store import dump, timestamp
        data = dict(data)
        if 'shapes' in data:
            serialized = dump(data.pop('shapes'))
            key = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
            db.execute('INSERT OR IGNORE INTO annotation_blobs VALUES(?,?)', (key, serialized))
            data['annotation_hash'] = key
        db.execute('INSERT INTO history(asset_id,revision,action,created_at,data) VALUES(?,?,?,?,?)',
                   (asset_id, revision, action, timestamp(), dump(data)))

    @staticmethod
    def list(db, asset_id):
        rows = []
        for row in db.execute('SELECT * FROM history WHERE asset_id=? ORDER BY id', (asset_id,)):
            data = json.loads(row['data'])
            if data.get('annotation_hash'):
                data['shapes'] = json.loads(db.execute('SELECT shapes FROM annotation_blobs WHERE hash=?', (data['annotation_hash'],)).fetchone()[0])
            rows.append(dict(row, data=data))
        return rows
