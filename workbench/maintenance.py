"""Recoverable image maintenance; active and trashed records both own bytes."""
import json
from functools import wraps
from .file_lock import exclusive_file_lock


def serialized_images(method):
    @wraps(method)
    def guarded(self, project_id, *args, **kwargs):
        with exclusive_file_lock(self.directory(project_id) / '.images.lock'):
            return method(self, project_id, *args, **kwargs)
    return guarded


def orphan_images(store, pid, *, quarantine=False):
    folder = store.directory(pid).resolve()
    images = (folder / 'images').resolve()
    if images.parent != folder or (folder / 'images').is_symlink():
        raise ValueError('圖片目錄無效')
    with exclusive_file_lock(folder / '.images.lock'), store.connection(pid) as db:
        used = {r[0] for r in db.execute('SELECT image_file FROM assets')}
        used.update(json.loads(r[0])['image_file'] for r in db.execute('SELECT data FROM review_trash'))
        candidates = [p for p in images.iterdir() if p.is_file() and not p.is_symlink() and p.name not in used]
        if quarantine:
            target = folder / 'orphan-quarantine'
            if target.is_symlink():
                raise ValueError('隔離目錄不得為連結')
            target.mkdir(exist_ok=True)
            for path in candidates:
                destination = target / path.name
                if destination.exists():
                    raise FileExistsError(f'隔離目錄已有 {path.name}；未覆寫既有檔案')
            for path in candidates:
                destination = target / path.name
                path.rename(destination)
        return {'files': [p.name for p in candidates], 'quarantined': quarantine,
                'recoverable': True, 'directory': str(folder / 'orphan-quarantine')}
