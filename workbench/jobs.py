"""Visible, bounded background jobs. Results never imply a successful commit early."""
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
import uuid

class JobCancelled(Exception):
    pass


class JobManager:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="workbench")
        self.install_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='installation')
        self.interactive_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='interactive')
        self.cpu_interactive_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='interactive-cpu')
        self.lock = threading.RLock()
        self.jobs = {}
        self.controls = {}
        self.closed = False
        self._cleanup_stop = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, name='job-retention', daemon=True)
        self._cleanup_thread.start()

    def _cleanup_loop(self):
        while not self._cleanup_stop.wait(60):
            with self.lock:
                self._prune()

    def submit(self, kind, action):
        with self.lock:
            self._prune()
            if self.closed:
                raise ValueError("軟體正在關閉")
            if len(self.active()) >= 8:
                raise ValueError("背景工作已滿，請等待目前工作完成")
            jid = uuid.uuid4().hex
            job = dict(id=jid, kind=kind, state="queued", message="等待處理", progress=None, created_at=time.time(), pauseable=True, stoppable=True)
            self.jobs[jid] = job
            self.controls[jid] = dict(paused=threading.Event(), cancelled=threading.Event())
            pool = (self.install_pool if kind == 'model-install' else self.interactive_pool if kind == 'ai'
                    else self.cpu_interactive_pool if kind == 'ai-cpu' else self.pool)
            pool.submit(self._run, jid, action)
            return dict(job)

    def _run(self, jid, action):
        control = self.controls[jid]
        def checkpoint():
            if control["cancelled"].is_set():
                raise JobCancelled("工作已停止")
            while control["paused"].is_set():
                with self.lock:
                    self.jobs[jid].update(state="paused", message="已暫停")
                if control["cancelled"].wait(.15):
                    raise JobCancelled("工作已停止")
            with self.lock:
                if self.jobs[jid]["state"] == "paused":
                    self.jobs[jid].update(state="running", message="繼續處理")
        def progress(message, percent=None):
            checkpoint()
            with self.lock:
                self.jobs[jid].update(message=str(message), progress=percent)
        with self.lock:
            self.jobs[jid].update(state="running", message="處理中")
        try:
            checkpoint()
            result = action(progress)
            with self.lock:
                self.jobs[jid].update(state="succeeded", message="已完成", progress=100, result=result)
        except JobCancelled as exc:
            with self.lock:
                self.jobs[jid].update(state="cancelled", message=str(exc), progress=None)
        except Exception as exc:
            logging.exception("Background job %s failed", jid)
            with self.lock:
                self.jobs[jid].update(state="failed", message=str(exc), error=str(exc), progress=None)
        finally:
            with self.lock:
                self.jobs[jid]['finished_at'] = time.time()

    def _prune(self):
        completed = sorted((job for job in self.jobs.values() if 'finished_at' in job), key=lambda job: job['finished_at'])
        for index, job in enumerate(completed):
            if time.time() - job['finished_at'] > 3600 or index < len(completed) - 100:
                self.jobs.pop(job['id'], None)
                self.controls.pop(job['id'], None)

    def get(self, jid):
        with self.lock:
            self._prune()
            if jid not in self.jobs:
                raise FileNotFoundError("找不到背景工作")
            return dict(self.jobs[jid])

    def active(self):
        with self.lock:
            self._prune()
            return [dict(job) for job in self.jobs.values() if job["state"] in {"running", "queued", "paused", "stopping"}]

    def control(self, jid, action):
        with self.lock:
            if jid not in self.jobs:
                raise FileNotFoundError("找不到背景工作")
            job, control = self.jobs[jid], self.controls[jid]
            if job["state"] in {"succeeded", "failed", "cancelled"}:
                return dict(job)
            if action == "pause":
                control["paused"].set(); job.update(state="paused", message="已暫停")
            elif action == "resume":
                control["paused"].clear(); job.update(state="running", message="繼續處理")
            elif action == "cancel":
                control["cancelled"].set(); control["paused"].clear(); job.update(state="stopping", message="正在安全停止…")
            else:
                raise ValueError("不支援的工作控制")
            return dict(job)

    def close(self):
        self._cleanup_stop.set()
        self._cleanup_thread.join(timeout=2)
        with self.lock:
            self.closed = True
            for control in self.controls.values():
                control["cancelled"].set()
                control["paused"].clear()
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.install_pool.shutdown(wait=True, cancel_futures=False)
        self.interactive_pool.shutdown(wait=True, cancel_futures=False)
        self.cpu_interactive_pool.shutdown(wait=True, cancel_futures=False)
