"""Conservative shared accelerator lease for training and interactive inference."""
from contextlib import contextmanager
from pathlib import Path
import os
import errno
import time
import json
import uuid


def accelerator_waiters(root):
    from .process_identity import alive
    for path in (Path(root) / '.gpu-requests').glob('*.json'):
        try:
            request = json.loads(path.read_text(encoding='utf-8'))
            if alive(request['pid'], request.get('created')) is not False:
                return True
        except (OSError, ValueError, KeyError):
            continue
    return False


@contextmanager
def _request_marker(root):
    from .process_identity import identity
    from .training_engine import atomic_json
    requests = Path(root) / '.gpu-requests'
    requests.mkdir(parents=True, exist_ok=True)
    marker = requests / f'{os.getpid()}-{uuid.uuid4().hex}.json'
    atomic_json(marker, {'pid': os.getpid(), 'created': identity(os.getpid())['created']})
    try:
        yield marker
    finally:
        marker.unlink(missing_ok=True)


@contextmanager
def accelerator_lease(root, *, device='auto', checkpoint=lambda: None):
    if str(device).lower() == 'cpu':
        yield
        return
    path = Path(root) / '.gpu-lease.lock'
    with _request_marker(root) as marker, path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        acquired = False
        try:
            while not acquired:
                checkpoint()
                stream.seek(0)
                try:
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    marker.unlink(missing_ok=True)
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    time.sleep(.2)
            yield
        finally:
            if acquired:
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
