import tempfile
import subprocess
import sys
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

from workbench.file_lock import exclusive_file_lock
from workbench.jobs import JobManager
from workbench.training import TrainingWorkspace


class HardeningTests(unittest.TestCase):
    def test_readiness_cache_invalidates_by_revision(self):
        workspace = object.__new__(TrainingWorkspace)
        workspace.lock = threading.RLock()
        revision = [1]
        @contextmanager
        def connection(_pid):
            db = Mock()
            db.execute.return_value.fetchone.return_value = [revision[0]]
            yield db
        workspace.store = Mock(connection=connection)
        workspace._calculate_readiness = Mock(side_effect=lambda pid: {'project_revision': revision[0], 'warnings': []})
        workspace.readiness('p')['warnings'].append('external mutation')
        self.assertEqual(workspace.readiness('p')['warnings'], [])
        self.assertEqual(workspace._calculate_readiness.call_count, 1)
        revision[0] = 2
        self.assertEqual(workspace.readiness('p')['project_revision'], 2)
        self.assertEqual(workspace._calculate_readiness.call_count, 2)

    def test_lock_excludes_another_process(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'admission.lock'
            marker = Path(folder) / 'acquired'
            code = ('from pathlib import Path; from workbench.file_lock import exclusive_file_lock; '
                    'import sys\nwith exclusive_file_lock(Path(sys.argv[1])):\n Path(sys.argv[2]).touch()')
            with exclusive_file_lock(path):
                process = subprocess.Popen([sys.executable, '-c', code, str(path), str(marker)],
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    time.sleep(.3)
                    self.assertFalse(marker.exists())
                    self.assertIsNone(process.poll())
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
            self.assertEqual(process.wait(timeout=15), 0)
            self.assertTrue(marker.exists())

    def test_status_does_not_load_overview(self):
        workspace = object.__new__(TrainingWorkspace)
        workspace.list_runs = Mock(return_value=[{'run_id': 'R001'}])
        workspace.overview = Mock(side_effect=AssertionError('expensive overview'))
        self.assertEqual(workspace.status('p'), {'runs': [{'run_id': 'R001'}]})

    def test_request_retry_returns_existing_run(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = object.__new__(TrainingWorkspace)
            workspace.lock = threading.RLock()
            workspace.store = Mock()
            workspace.store.directory.return_value = Path(folder)
            previous = {'run_id': 'R001', 'request_id': 'retry', 'request_payload': {'dataset_id': 'D001', 'config': {}}}
            workspace.list_runs = Mock(return_value=[previous])
            workspace._start_run_locked = Mock(side_effect=AssertionError('duplicate start'))
            self.assertEqual(workspace.start_run('p', 'D001', {}, 'retry'), previous)
            with self.assertRaises(ValueError):
                workspace.start_run('p', 'D002', {}, 'retry')

    def test_completed_jobs_expire_without_removing_active_jobs(self):
        jobs = JobManager()
        try:
            jobs.jobs = {'old': {'id': 'old', 'state': 'succeeded', 'finished_at': time.time()-3601},
                         'running': {'id': 'running', 'state': 'running'}}
            jobs.controls = {'old': {}}
            jobs._prune()
            self.assertEqual(set(jobs.jobs), {'running'})
        finally:
            jobs.controls.clear()
            jobs.close()
