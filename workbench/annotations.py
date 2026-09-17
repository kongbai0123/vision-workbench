"""Shared annotation validation and atomic commit for every editor."""
import json


class AnnotationService:
    def __init__(self, store):
        self.store = store

    def commit(self, pid, updates, *, source='builtin', register_classes=False):
        from .store import ConflictError, clean_shapes, dump, identifier, timestamp, repair_saved_mask_holes
        prepared = []
        repairs = []
        with self.store.connection(pid, write=True) as db:
            classes = json.loads(db.execute('SELECT classes FROM project').fetchone()[0])
            for original, shapes in updates:
                row = db.execute('SELECT * FROM assets WHERE id=?', (identifier(original['id']),)).fetchone()
                if row is None:
                    raise FileNotFoundError('找不到圖片')
                revision_matches = row['revision'] == original['revision']
                if not revision_matches and type(original['revision']) is int and original['revision'] < row['revision']:
                    intervening = db.execute('SELECT action FROM history WHERE asset_id=? AND revision>?', (row['id'], original['revision'])).fetchall()
                    revision_matches = bool(intervening) and all(item[0] in {'assign', 'auto_split', 'smart_split'} for item in intervening)
                if type(original['revision']) is not int or not revision_matches:
                    raise ConflictError('圖片已有新版本；保留目前編輯，請重新載入後整合修改')
                cleaned = clean_shapes(shapes, row['width'], row['height'])
                cleaned, changes = repair_saved_mask_holes(cleaned, row['width'], row['height'])
                repairs.extend(changes)
                unknown = list(dict.fromkeys(shape['label'] for shape in cleaned if shape['label'] not in classes))
                if unknown and not register_classes:
                    raise ValueError(f"類別「{'、'.join(unknown[:5])}」尚未由使用者建立；請先到類別管理新增")
                classes.extend(unknown)
                if dump(cleaned) != row['shapes']:
                    prepared.append((row, cleaned))
            for row, cleaned in prepared:
                revision = row['revision'] + 1
                db.execute("UPDATE assets SET shapes=?,revision=?,review_state='pending',updated_at=? WHERE id=?",
                           (dump(cleaned), revision, timestamp(), row['id']))
                self.store._history(db, row['id'], revision, 'edit', {'shapes': cleaned, 'review_state': 'pending', 'editor': source})
            if prepared:
                db.execute('UPDATE project SET classes=?', (dump(classes),))
                self.store._touch(db)
        return {'updated': len(prepared), 'repairs': repairs}
