"""Keep the SAM runtime and its GPU lease together until idle or requested."""
import logging
import threading
import time
from .resources import accelerator_lease, accelerator_waiters


class InteractiveRuntime:
    def __init__(self, root, unload, *, idle_seconds=60, interval=.2):
        self.root, self.unload = root, unload
        self.idle_seconds = idle_seconds
        self.lock = threading.RLock()
        self.lease = None
        self.last_used = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._maintenance, args=(interval,), daemon=True, name='sam-runtime-retention')
        self.thread.start()

    def _release(self):
        if self.lease is not None:
            lease, self.lease = self.lease, None
            try:
                self.unload()
            finally:
                lease.__exit__(None, None, None)

    def run(self, action, checkpoint):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError('AI 執行環境已關閉')
            if self.lease is not None and accelerator_waiters(self.root):
                self._release()
                # Let a waiting trainer take the released lease first.
                while accelerator_waiters(self.root):
                    checkpoint()
                    time.sleep(.1)
            if self.lease is None:
                lease = accelerator_lease(self.root, checkpoint=checkpoint)
                lease.__enter__()
                self.lease = lease
            try:
                result = action()
                checkpoint()
                return result
            except BaseException:
                self._release()
                raise
            finally:
                self.last_used = time.monotonic()
                if accelerator_waiters(self.root):
                    self._release()

    def _maintenance(self, interval):
        while not self.stop.wait(interval):
            if self.lock.acquire(blocking=False):
                try:
                    if self.lease is not None and (accelerator_waiters(self.root)
                            or time.monotonic() - self.last_used >= self.idle_seconds):
                        self._release()
                except Exception:
                    logging.exception('Could not release idle SAM runtime')
                finally:
                    self.lock.release()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        with self.lock:
            self._release()
