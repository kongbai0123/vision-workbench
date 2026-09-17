"""Review exclusions, recoverable removal and content-based import preview."""
import base64
import io
import secrets
import threading
import time

import cv2
import numpy as np
from PIL import Image

from .store import ConflictError


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

    def preview(self, pid, paths, *, progress=lambda *_, **__: None):
        from .pipeline import import_sources, _sha
        self.store.get_project(pid, include_assets=False)
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            raise ValueError('請選擇有效路徑')
        progress('掃描檔案、解碼圖片與解析標註', 0, phase='scan')
        parsed = import_sources(paths, progress=lambda message, percent: progress(message, percent))
        items = []
        progress('建立縮圖與分析清晰度', 0, phase='quality')
        for index, record in enumerate(parsed['records']):
            quality = image_quality(record['path'])
            record['expected_import_sha'] = _sha(record['path'])
            record['source'] = {**record.get('source', {}), 'quality': {k:v for k,v in quality.items() if k != 'thumbnail'}}
            items.append({'id': str(index), 'name': record['name'], 'shape_count': len(record['shapes']), **quality})
            progress(f'縮圖、清晰度與雜湊 {index+1} / {len(parsed["records"])}', (index+1)/len(parsed['records'])*100)
        token = secrets.token_urlsafe(24)
        with self.lock:
            now = time.monotonic()
            self.previews = {k:v for k,v in self.previews.items() if now-v[0] < 3600}
            if len(self.previews) >= 8:
                self.previews.pop(next(iter(self.previews)))
            self.previews[token] = (now, pid, parsed['records'])
        return {'token': token, 'items': items, 'issues': parsed['issues']}

    def commit_import(self, pid, token, selected, *, progress=lambda *_, **__: None):
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
            progress('確認來源圖片未變更', 0, phase='verify')
            for index, record in enumerate(records):
                if _sha(record['path']) != record.pop('expected_import_sha'):
                    raise ConflictError('預覽後來源圖片已變更，請重新預覽')
                progress(f'來源完整性 {index+1} / {len(records)}', (index+1)/len(records)*100)
            progress('保存原圖與標註', phase='save')
            progress('保存原圖與標註', 0)
            def saving(number, total):
                progress(f'保存原圖 {number} / {total}', number/total*100)
                if number == total:
                    progress('寫入資料庫並更新專案；正在確認完成', phase='commit')
            result = self.store.add_assets(pid, records, progress=saving)
            self.previews.pop(token, None)
            return result

    def trash(self, pid, ids, revisions, restore=False):
        return self.store.trash_assets(pid, ids, revisions, restore=restore)

    def list_trash(self, pid):
        return self.store.list_trash(pid)

    def quality(self, pid):
        project = self.store.get_project(pid, internal=True)
        results = {a['id']: {k:v for k,v in image_quality(a['image_path']).items() if k != 'thumbnail'} for a in project['assets']}
        return self.store.save_quality(pid, results)
