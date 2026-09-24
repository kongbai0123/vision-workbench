"""Project-scoped editable setup, separate from immutable datasets and runs."""
import json
from .store import ConflictError, dump, timestamp


class WorkflowDrafts:
    scope = 'preparation-training'

    def __init__(self, store):
        self.store = store

    def read(self, pid):
        with self.store.connection(pid) as db:
            row = db.execute('SELECT * FROM workflow_drafts WHERE scope=?', (self.scope,)).fetchone()
            return self._record(row)

    @staticmethod
    def _record(row):
        return {'schema_version': 1, 'revision': row['revision'] if row else 0,
                'updated_at': row['updated_at'] if row else None,
                'payload': json.loads(row['payload']) if row else {}}

    def save(self, pid, revision, payload):
        if type(revision) is not int or revision < 0 or not isinstance(payload, dict):
            raise ValueError('設定草稿需要有效版本與物件內容')
        if set(payload) - {'augmentation', 'engine', 'dataset', 'parameters', 'tab'}:
            raise ValueError('設定草稿包含未知欄位')
        for key in ('augmentation', 'parameters'):
            if key in payload and not isinstance(payload[key], dict):
                raise ValueError(f'{key} 必須是物件')
        for key in ('engine', 'dataset', 'tab'):
            if key in payload and (not isinstance(payload[key], str) or len(payload[key]) > 128):
                raise ValueError(f'{key} 必須是短字串')
        augmentation = payload.get('augmentation', {})
        if set(augmentation) - {'preset', 'expansion_count', 'brightness', 'contrast', 'fliplr', 'flipud'}:
            raise ValueError('草稿包含未知增強欄位')
        if any(type(value) not in (str, int, float) or len(str(value)) > 128 for value in augmentation.values()):
            raise ValueError('增強草稿值無效')
        parameters = payload.get('parameters', {})
        if len(parameters) > 100 or any(not isinstance(values, dict) or len(values) > 100 for values in parameters.values()):
            raise ValueError('模型參數草稿無效')
        if any(not isinstance(value, str) or len(value) > 128 for values in parameters.values() for value in values.values()):
            raise ValueError('模型參數草稿必須保存原始輸入文字')
        # Drafts preserve incomplete input; publication validates domain values.
        encoded = dump(payload)
        if len(encoded.encode('utf-8')) > 65536:
            raise ValueError('設定草稿超過 64 KiB')
        with self.store.connection(pid, write=True) as db:
            row = db.execute('SELECT * FROM workflow_drafts WHERE scope=?', (self.scope,)).fetchone()
            current = self._record(row)
            if revision != current['revision']:
                raise ConflictError('設定草稿已由另一個視窗更新；目前輸入仍保留，請記下變更後重新載入頁面或重啟程式以取得最新設定')
            if current['payload'] == payload:
                return current
            now, next_revision = timestamp(), revision + 1
            db.execute('INSERT INTO workflow_drafts VALUES(?,1,?,?,?) ON CONFLICT(scope) DO UPDATE SET revision=excluded.revision,updated_at=excluded.updated_at,payload=excluded.payload',
                       (self.scope, next_revision, now, encoded))
            db.execute('INSERT INTO workflow_draft_history VALUES(?,?,?,?)', (self.scope, next_revision, now, encoded))
        return {'schema_version': 1, 'revision': next_revision, 'updated_at': now, 'payload': payload}
