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
        self.lock = threading.RLock()
        self.jobs = {}
        self.controls = {}
        self.closed = False

    def submit(self, kind, action):
        with self.lock:
            if self.closed:
                raise ValueError("軟體正在關閉")
            if len(self.active()) >= 8:
                raise ValueError("背景工作已滿，請等待目前工作完成")
            jid = uuid.uuid4().hex
            job = dict(id=jid, kind=kind, state="queued", message="等待處理", progress=None, created_at=time.time(), pauseable=True, stoppable=True)
            self.jobs[jid] = job
            self.controls[jid] = dict(paused=threading.Event(), cancelled=threading.Event())
            self.pool.submit(self._run, jid, action)
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

    def get(self, jid):
        with self.lock:
            if jid not in self.jobs:
                raise FileNotFoundError("找不到背景工作")
            return dict(self.jobs[jid])

    def active(self):
        with self.lock:
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
        with self.lock:
            self.closed = True
            for control in self.controls.values():
                control["cancelled"].set()
                control["paused"].clear()
        self.pool.shutdown(wait=True, cancel_futures=False)
