"""Versioned, dependency-free contracts at HTTP and process boundaries."""
import math
import re

SCHEMA_VERSION = 1
TERMINAL_STATES = frozenset({'completed', 'failed', 'stopped'})
ACTIVE_STATES = frozenset({'queued', 'preparing', 'running', 'stopping'})

FIELD_TYPES = {
    'asset_ids': list, 'project_ids': list, 'paths': list, 'shapes': list, 'classes': list,
    'revisions': dict, 'replacements': dict, 'config': dict, 'options': dict,
    'ratios': dict, 'settings': dict, 'revision': int, 'history_id': int, 'delta_base': int,
    'name': str, 'state': str, 'reason': str, 'note': str, 'batch_id': str,
    'split': str, 'project_id': str, 'asset_id': str, 'dataset_version_id': str,
    'model_version_id': str, 'request_id': str, 'format': str, 'version': str,
    'path': str, 'kind': str, 'mode': str, 'engine': str,
    'confirmed': bool, 'confirm': bool, 'acknowledge_loss': bool, 'quarantine': bool,
}

REQUIRED = (
    (r'/api/projects', 'POST', ('name',)),
    (r'/api/projects/[^/]+/assets/[^/]+', 'PUT', ('shapes', 'revision')),
    (r'/api/projects/[^/]+/review', 'POST', ('asset_ids', 'state', 'revisions')),
    (r'/api/projects/[^/]+/training-runs', 'POST', ('dataset_version_id',)),
)


def validate_request(path, method, payload):
    if not isinstance(payload, dict):
        raise ValueError('請求內容必須為物件')
    for pattern, verb, fields in REQUIRED:
        if verb == method and re.fullmatch(pattern, path):
            missing = [name for name in fields if name not in payload]
            if missing:
                raise ValueError('缺少必要欄位：' + ', '.join(missing))
    for name, expected in FIELD_TYPES.items():
        if name in payload and type(payload[name]) is not expected:
            raise ValueError(f'{name} 的資料型別無效')
    for name in ('asset_ids', 'project_ids', 'paths', 'classes'):
        if name in payload and any(not isinstance(item, str) for item in payload[name]):
            raise ValueError(f'{name} 必須為文字陣列')
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('數值必須為有限數')
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    finite(payload)
    return payload


def validate_run_event(event):
    if not isinstance(event, dict) or event.get('schema_version') != SCHEMA_VERSION or event.get('type') != 'state':
        raise ValueError('Invalid run event envelope')
    state = event.get('state')
    if not isinstance(state, dict) or state.get('status') not in ACTIVE_STATES | TERMINAL_STATES:
        raise ValueError('Invalid run state')
    return event
