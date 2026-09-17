"""Worker-owned liveness records, independent of application lifetime."""
from contextlib import contextmanager
import os
import threading
import time


@contextmanager
def worker_lifecycle(run_dir, run):
    from .training_engine import atomic_json
    stop = threading.Event()
    def beat():
        while not stop.is_set():
            atomic_json(run_dir / 'heartbeat.json', {'pid': os.getpid(), 'time': time.time()})
            stop.wait(2)
    thread = threading.Thread(target=beat, name='worker-heartbeat', daemon=True)
    thread.start()
    try:
        from .resources import accelerator_lease
        from pathlib import Path
        root = Path(os.environ.get('VISION_WORKBENCH_RESOURCE_ROOT', str(run_dir.parent.parent.parent.parent)))
        def checkpoint():
            from .training_engine import stop_requested
            if stop_requested(run_dir):
                raise InterruptedError('等待運算資源時已取消訓練')
        with accelerator_lease(root, device=run.get('config', {}).get('device', 'auto'), checkpoint=checkpoint):
            yield
    finally:
        stop.set()
        thread.join(timeout=5)
