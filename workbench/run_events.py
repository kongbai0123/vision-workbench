"""Durable run-state events; JSON snapshots remain export-compatible projections."""
import json
import os
import time
from .file_lock import exclusive_file_lock
from vision_workbench.contracts import validate_run_event


def append_state(path, value):
    events = path.with_name('events.jsonl')
    with exclusive_file_lock(path.with_name('.events.lock')):
        event = {'schema_version': 1, 'type': 'state', 'time': time.time(), 'pid': os.getpid(), 'state': value}
        validate_run_event(event)
        with events.open('ab') as stream:
            stream.write((json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())


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
