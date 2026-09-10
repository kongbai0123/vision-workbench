import threading
import time
import unittest

from workbench.jobs import JobManager


class JobControlTests(unittest.TestCase):
    def setUp(self):
        self.jobs = JobManager()

    def tearDown(self):
        self.jobs.close()

    def wait_for(self, job_id, states, timeout=3):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.jobs.get(job_id)
            if job["state"] in states:
                return job
            time.sleep(.01)
        self.fail(f"job did not enter {states}: {self.jobs.get(job_id)}")

    def test_pause_resume_and_complete(self):
        entered = threading.Event()

        def action(progress):
            for index in range(30):
                entered.set()
                progress(f"step {index}", index)
                time.sleep(.01)
            return "done"

        job_id = self.jobs.submit("test", action)["id"]
        self.assertTrue(entered.wait(1))
        paused = self.jobs.control(job_id, "pause")
        self.assertEqual(paused["state"], "paused")
        time.sleep(.05)
        self.assertEqual(self.jobs.get(job_id)["state"], "paused")
        self.jobs.control(job_id, "resume")
        result = self.wait_for(job_id, {"succeeded"})
        self.assertEqual(result["result"], "done")
        self.assertEqual(result["progress"], 100)

    def test_cancel_paused_job_at_checkpoint(self):
        entered = threading.Event()

        def action(progress):
            for index in range(100):
                entered.set()
                progress(f"step {index}", index)
                time.sleep(.01)

        job_id = self.jobs.submit("test", action)["id"]
        self.assertTrue(entered.wait(1))
        self.jobs.control(job_id, "pause")
        self.jobs.control(job_id, "cancel")
        result = self.wait_for(job_id, {"cancelled"})
        self.assertEqual(result["message"], "工作已停止")


if __name__ == "__main__":
    unittest.main()
