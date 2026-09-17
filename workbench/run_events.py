"""Durable run-state events; JSON snapshots remain export-compatible projections."""
import json
import os
import time
from vision_workbench.contracts import validate_run_event

TERMINAL = {'completed', 'failed', 'stopped'}
PROGRESS_FIELDS = {'progress', 'batch', 'batches_per_epoch', 'execution', 'message',
                   'updated_at', 'device', 'initialization', 'runtime'}


def append_state(path, value):
    """Called while the writer holds .events.lock through snapshot replacement."""
    events = path.with_name('events.jsonl')
    previous = latest_state(path)
    if previous and previous.get('status') in TERMINAL:
        return previous
    try:
        snapshot = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        snapshot = {}
    value = dict(value, sequence=max((previous or {}).get('sequence', 0), snapshot.get('sequence', 0)) + 1)
    keys = ('status', 'phase', 'epoch', 'metrics')
    if previous is None or any(previous.get(k) != value.get(k) for k in keys):
        event = {'schema_version': 1, 'type': 'state', 'time': time.time(), 'pid': os.getpid(), 'state': value}
        validate_run_event(event)
        # Isolate a crash-truncated tail so it cannot swallow the next record.
        if events.exists() and events.stat().st_size:
            with events.open('rb') as stream:
                stream.seek(-1, 2)
                incomplete = stream.read(1) != b'\n'
            if incomplete:
                with events.open('ab') as stream:
                    stream.write(b'\n')
        with events.open('ab') as stream:
            stream.write((json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
    return value


def read_state(path):
    from .training_engine import read_json
    event = latest_state(path)
    if event and event.get('status') in TERMINAL:
        return event
    try:
        snapshot = read_json(path)
    except (OSError, ValueError):
        if event is None:
            raise
        return event
    if event is None:
        return snapshot
    if (snapshot.get('sequence', 0) > event.get('sequence', 0)
            and snapshot.get('status') == event.get('status')
            and snapshot.get('epoch') == event.get('epoch')):
        event.update({k: snapshot[k] for k in PROGRESS_FIELDS if k in snapshot})
        event['sequence'] = snapshot['sequence']
    return event


def latest_state(path):
    events = path.with_name('events.jsonl')
    if not events.exists():
        return None
    # Read backwards; memory is bounded by the newest record, not run length.
    with events.open('rb') as stream:
        stream.seek(0, 2)
        position = stream.tell()
        data = b''
        while position:
            size = min(position, 65536)
            position -= size
            stream.seek(position)
            data = stream.read(size) + data
            lines = data.split(b'\n')
            for raw in reversed(lines[1:] if position else lines):
                if not raw.strip():
                    continue
                try:
                    event = validate_run_event(json.loads(raw))
                except (ValueError, UnicodeDecodeError):
                    continue
                if event.get('type') == 'state':
                    return event['state']
    return None
