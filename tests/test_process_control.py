import sys
import tempfile
import unittest

from workbench.jobs import JobCancelled
from workbench.process_control import run_controlled


class ProcessControlTests(unittest.TestCase):
    def test_silent_process_can_be_cancelled(self):
        def cancel(*args):
            raise JobCancelled('cancel')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(JobCancelled):
                run_controlled([sys.executable, '-c', 'import time; time.sleep(60)'], cancel, cwd=folder)

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(TimeoutError):
                run_controlled([sys.executable, '-c', 'import time; time.sleep(60)'], lambda *a: None,
                               cwd=folder, timeout=.1)
