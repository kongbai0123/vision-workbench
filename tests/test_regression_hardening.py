"""Upgrade/lifecycle regressions. Every database and worker belongs to a temp dir."""
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image
from workbench.independence import confirmation, fingerprint, repair, valid
from workbench.interactive_runtime import InteractiveRuntime
from workbench.jobs import JobCancelled, JobManager
from workbench.migrations import migrate
from workbench.process_control import installation_options, run_controlled
from workbench.process_identity import alive, identity
from workbench.repair_independence import repair_files
from workbench.run_events import latest_state, read_state
from workbench.store import ProjectStore, dump
from workbench.training import TrainingWorkspace
from workbench.training_engine import atomic_json, _status


class RegressionHardeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='regression-hardening-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def legacy(self, *, good=True):
        path = self.root / 'legacy.db'
        db = sqlite3.connect(path)
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript((Path(__file__).parents[1] / 'workbench/migrations/0000_core.sql').read_text())
        db.execute('CREATE TABLE independence_reviews(id INTEGER PRIMARY KEY,data TEXT NOT NULL)')
        db.execute("INSERT INTO project VALUES('p','old',10,'then','then','[]')")
        source = dump({'kind': 'import', 'review': {'reason': 'approved'}, 'z': 'source', 'quality': {'score': 8}})
        db.execute('INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   ('a', 'a.png', 10, 10, 'hash', 'a.png', 'batch', 'train', source, 2, 'approved', '[]', 'then', 'then'))
        signature, count = fingerprint(db)
        record = {'asset_fingerprint': signature if good else 'tampered', 'asset_count': count, 'confirmed_at': 'original-time'}
        db.execute('INSERT INTO independence_reviews VALUES(1,?)', (dump(record),))
        db.execute("INSERT INTO history VALUES(1,'a',2,'save','then',?)", (dump({'shapes': [], 'review_state': 'approved'}),))
        db.commit()
        return path, db, record

    def test_legacy_wal_upgrade_preserves_valid_confirmation_and_history(self):
        path, db, original = self.legacy()
        with closing(db):
            self.assertGreater(Path(str(path) + '-wal').stat().st_size, 0)
            migrate(db)
            self.assertTrue(valid(confirmation(db), fingerprint(db)))
            self.assertEqual(confirmation(db)['confirmed_at'], original['confirmed_at'])
            self.assertEqual(confirmation(db)['migration_audit'][0]['migration'], '0002')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM annotation_blobs').fetchone()[0], 1)
            self.assertIsNotNone(db.execute('SELECT annotation_hash FROM history').fetchone()[0])
            saved = path.with_suffix('.before-migration-0005.backup')
            with closing(sqlite3.connect(saved)) as backup:
                self.assertEqual(backup.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
                self.assertEqual(confirmation(backup), original)
                self.assertTrue(valid(original, fingerprint(backup)))
            before = confirmation(db)
            migrate(db)
            self.assertEqual(before, confirmation(db))

    def test_upgrade_does_not_validate_tampered_confirmation(self):
        _, db, original = self.legacy(good=False)
        with closing(db):
            migrate(db)
            self.assertEqual(confirmation(db), original)
            self.assertFalse(valid(confirmation(db), fingerprint(db)))

    def test_later_migrations_add_a_new_backup_without_replacing_old_backup(self):
        path, db, _ = self.legacy()
        with closing(db):
            migrations = Path(__file__).parents[1] / 'workbench/migrations'
            original_glob = Path.glob
            def pre_0004(folder, pattern):
                return (p for p in original_glob(folder, pattern)
                        if folder != migrations or int(p.name.split("_")[0]) < 4)
            with patch('workbench.migrations.Path.glob', pre_0004):
                migrate(db)
            older = path.with_suffix('.before-migration-0003.backup')
            original_bytes = older.read_bytes()
            db.execute("INSERT INTO asset_review VALUES('orphan','{}')")
            db.execute("INSERT INTO annotation_blobs VALUES('orphan','[]')")
            db.commit()
            migrate(db)
            self.assertEqual(older.read_bytes(), original_bytes)
            self.assertTrue(path.with_suffix('.before-migration-0005.backup').exists())
            self.assertEqual(db.execute("SELECT COUNT(*) FROM asset_review WHERE asset_id='orphan'").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM annotation_blobs WHERE hash='orphan'").fetchone()[0], 0)

    def test_repair_requires_all_evidence_and_is_dry_run_idempotent(self):
        path, db, original = self.legacy()
        with closing(db):
            migrate(db)
            backup = path.with_suffix('.before-migration-0005.backup')
            db.execute('UPDATE independence_reviews SET data=?', (dump(original),))
            db.commit()  # Simulate the previously shipped 0002 migration.
        before = path.read_bytes()
        self.assertEqual(repair_files(path, backup)['reason'], 'migration_only')
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            repair_files(path, backup, apply=True)
        self.assertTrue(repair_files(path, backup, apply=True, offline=True)['changed'])
        self.assertEqual(repair_files(path, backup, apply=True, offline=True)['reason'], 'already_current')
        with closing(sqlite3.connect(path)) as current, closing(sqlite3.connect(backup)) as evidence:
            current.execute('UPDATE independence_reviews SET data=?', (dump(dict(original, confirmed_at='changed')),))
            self.assertEqual(repair(current, evidence)['reason'], 'confirmation_changed')
            current.execute('UPDATE independence_reviews SET data=?', (dump(original),))
            current.execute("UPDATE assets SET shapes='[{}]'")
            self.assertEqual(repair(current, evidence)['reason'], 'approved_content_changed')
            evidence.execute('UPDATE independence_reviews SET data=?', (dump(dict(original, asset_fingerprint='wrong')),))
            self.assertEqual(repair(current, evidence)['reason'], 'backup_confirmation_invalid')

    def workspace(self):
        store = ProjectStore(self.root / 'data/projects')
        pid = store.create_project('reattach')['id']
        with patch.object(TrainingWorkspace, '_probe_maskrcnn', return_value=False):
            workspace = TrainingWorkspace(self.root / 'data', store)
        return store, pid, workspace

    def test_reopened_workspace_detects_later_worker_death(self):
        store, pid, first = self.workspace()
        directory = first.runs_dir(pid, create=True) / 'R001'
        path = directory / 'run.json'
        atomic_json(path, {'run_id': 'R001', 'status': 'running', 'epoch': 1})
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'],
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            atomic_json(directory / 'worker.json', {'worker_pid': process.pid, 'worker_created': identity(process.pid)['created']})
            atomic_json(directory / 'heartbeat.json', {'pid': process.pid, 'time': time.time() - 30})
            first.close()
            with patch.object(TrainingWorkspace, '_probe_maskrcnn', return_value=False):
                second = TrainingWorkspace(self.root / 'data', store)
            try:
                current = second.run(pid, 'R001')
                self.assertEqual(current['status'], 'running')
                self.assertIn('health_warning', current)
                process.kill()
                process.wait(timeout=5)
                self.assertFalse(alive(process.pid))
                self.assertEqual(second.run(pid, 'R001')['status'], 'failed')
                self.assertFalse(any(r['status'] in {'queued', 'preparing', 'running', 'stopping'}
                                     for r in second.status(pid)['runs']))
            finally:
                second.close()
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            first.close()

    def test_pid_identity_and_final_event_race(self):
        _, pid, workspace = self.workspace()
        try:
            path = workspace.runs_dir(pid, create=True) / 'R001/run.json'
            atomic_json(path, {'run_id': 'R001', 'status': 'running'})
            atomic_json(path.parent / 'worker.json', {'worker_pid': os.getpid(), 'worker_created': 'wrong'})
            self.assertFalse(alive(os.getpid(), 'wrong'))
            self.assertEqual(workspace.run(pid, 'R001')['status'], 'failed')
            path.parent.joinpath('control/terminal.json').unlink()
            def dies(*_):
                atomic_json(path, {'run_id': 'R001', 'status': 'completed'})
                return False
            with patch.object(workspace, '_pid_alive', side_effect=dies):
                self.assertEqual(workspace.run(pid, 'R001')['status'], 'completed')
            self.assertFalse(path.parent.joinpath('control/terminal.json').exists())
        finally:
            workspace.close()

    def test_actual_training_worker_reopens_dies_and_allows_new_training(self):
        from workbench.resources import accelerator_lease
        store, pid, first = self.workspace()
        second = None
        processes = []
        try:
            shape = {'id': 's', 'type': 'rectangle', 'label': 'part', 'x': 2, 'y': 2, 'width': 5, 'height': 5}
            for index, split in enumerate(('train', 'val', 'test')):
                image = self.root / f'{split}.png'
                Image.new('RGB', (12, 12), (80 + index, 20, 30)).save(image)
                store.add_assets(pid, [{'path': image, 'shapes': [shape], 'review_state': 'approved',
                                       'split': split, 'batch_id': split}])
            dataset = first.create_dataset_version(pid)
            with accelerator_lease(self.root / 'data'):
                run = first.start_run(pid, dataset['id'], {'engine': 'pixel_prototype_v1', 'epochs': 2})
                worker = first.processes[(pid, run['run_id'])]
                processes.append(worker)
                first.close()
                with patch.object(TrainingWorkspace, '_probe_maskrcnn', return_value=False):
                    second = TrainingWorkspace(self.root / 'data', store)
                self.assertIn(second.run(pid, run['run_id'])['status'], {'queued', 'preparing', 'running'})
                worker.kill()
                worker.wait(timeout=5)
                self.assertEqual(second.run(pid, run['run_id'])['status'], 'failed')
                replacement = second.start_run(pid, dataset['id'], {'engine': 'pixel_prototype_v1', 'epochs': 2})
                processes.append(second.processes[(pid, replacement['run_id'])])
            processes[-1].wait(timeout=20)
            self.assertEqual(second.run(pid, replacement['run_id'])['status'], 'completed')
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
            first.close()
            if second:
                second.close()

    def test_31300_batches_have_bounded_events_and_terminal_precedence(self):
        path = self.root / 'run.json'
        run = {'run_id': 'R001', 'status': 'running', 'config': {'padding': 'x' * 7000}}
        for epoch in range(1, 101):
            for batch in range(1, 314):
                _status(self.root, run, epoch=epoch, batch=batch, phase='training', progress=batch / 313)
            _status(self.root, run, metrics={'epoch': epoch, 'loss': 1 / epoch})
        self.assertEqual(read_state(path)['batch'], 313)
        self.assertLessEqual(len(path.with_name('events.jsonl').read_text().splitlines()), 200)
        self.assertLess(path.with_name('events.jsonl').stat().st_size, 2_000_000)
        sequence = read_state(path)['sequence']
        self.assertEqual(sequence, 31400)
        atomic_json(path, dict(run, status='completed'))
        path.write_text(dump(dict(run, status='running', sequence=999999)), encoding='utf-8')
        self.assertEqual(read_state(path)['status'], 'completed')
        atomic_json(path, dict(run, status='running'))
        self.assertEqual(read_state(path)['status'], 'completed')

    def test_snapshot_progress_and_truncated_event_recovery(self):
        path = self.root / 'run.json'
        atomic_json(path, {'status': 'running', 'epoch': 1, 'batch': 1})
        atomic_json(path, {'status': 'running', 'epoch': 1, 'batch': 2})
        self.assertEqual(latest_state(path)['batch'], 1)
        self.assertEqual(read_state(path)['batch'], 2)
        path.write_text(dump({'status': 'running', 'epoch': 1, 'batch': 99, 'sequence': 0}))
        self.assertEqual(read_state(path)['batch'], 1)
        with path.with_name('events.jsonl').open('ab') as out:
            out.write(b'{"partial":')
        atomic_json(path, {'status': 'completed', 'epoch': 1})
        self.assertEqual(read_state(path)['status'], 'completed')

    def test_install_streams_cr_invalid_utf8_and_inherits_pip_raw(self):
        messages = []
        code = "import os,time; print(os.environ['PIP_PROGRESS_BAR'],flush=True); [ (os.write(1, ('Progress '+str(i)+'\\r').encode()+b'\\xff'),time.sleep(.2)) for i in range(4)]"
        options = installation_options()
        self.assertIsNone(options['timeout'])
        run_controlled([sys.executable, '-c', code], lambda m, *_: messages.append(m),
                       cwd=self.root, **dict(options, idle_timeout=1))
        self.assertTrue(any('Progress 0' in m for m in messages))
        self.assertTrue(any('Progress 2' in m for m in messages))

    def test_install_idle_timeout_failure_tail_and_tree_cancellation(self):
        with self.assertRaisesRegex(TimeoutError, 'last-output'):
            run_controlled([sys.executable, '-c', "import time;print('last-output',flush=True);time.sleep(10)"],
                           lambda *_: None, cwd=self.root, timeout=None, idle_timeout=.5)
        with self.assertRaisesRegex(RuntimeError, 'failure-detail'):
            run_controlled([sys.executable, '-c', "print('failure-detail');raise SystemExit(2)"],
                           lambda *_: None, cwd=self.root)
        pid_file = self.root / 'grandchild.pid'
        code = f"import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"
        def cancel(*_):
            if pid_file.exists():
                raise JobCancelled('cancel tree')
        with self.assertRaises(JobCancelled):
            run_controlled([sys.executable, '-c', code], cancel, cwd=self.root, timeout=10)
        self.assertFalse(alive(int(pid_file.read_text())))

    def test_runtime_reuses_model_and_releases_for_waiting_process(self):
        runtime = {'loaded': False, 'loads': 0, 'unloads': 0}
        def infer():
            if not runtime['loaded']:
                runtime.update(loaded=True, loads=runtime['loads'] + 1)
            return 'mask'
        def unload():
            runtime.update(loaded=False, unloads=runtime['unloads'] + 1)
        owner = InteractiveRuntime(self.root, unload, idle_seconds=10, interval=.02)
        try:
            owner.run(infer, lambda: None)
            owner.run(infer, lambda: None)
            self.assertEqual(runtime['loads'], 1)
            code = f"from workbench.resources import accelerator_lease; from pathlib import Path\nwith accelerator_lease({str(self.root)!r}): Path({str(self.root / 'acquired')!r}).touch()"
            child = subprocess.Popen([sys.executable, '-c', code], cwd=Path(__file__).parents[1],
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                child.wait(timeout=10)
                self.assertEqual(child.returncode, 0)
                self.assertTrue((self.root / 'acquired').exists())
                self.assertFalse(runtime['loaded'])
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
            owner.idle_seconds = .05
            owner.run(infer, lambda: None)
            deadline = time.monotonic() + 2
            while runtime['loaded'] and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertFalse(runtime['loaded'])
        finally:
            owner.close()

    def test_cpu_job_does_not_wait_for_gpu_queue(self):
        jobs = JobManager()
        entered, release = threading.Event(), threading.Event()
        try:
            def wait_gpu(progress):
                entered.set()
                release.wait(5)
            jobs.submit('ai', wait_gpu)
            self.assertTrue(entered.wait(2))
            cpu = jobs.submit('ai-cpu', lambda _: 'grabcut')
            deadline = time.monotonic() + 2
            while jobs.get(cpu['id'])['state'] not in {'succeeded', 'failed'} and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(jobs.get(cpu['id'])['state'], 'succeeded')
        finally:
            release.set()
            jobs.close()

    def test_service_grabcut_finishes_while_sam_waits_for_external_gpu(self):
        from workbench.server import WorkbenchService
        with patch.object(TrainingWorkspace, '_probe_maskrcnn', return_value=False):
            service = WorkbenchService(self.root / 'data')
        child = None
        try:
            store = service.store
            pid = store.create_project('CPU during GPU wait')['id']
            image = self.root / 'image.png'
            from PIL import ImageDraw
            picture = Image.new('RGB', (40, 40), 'black')
            ImageDraw.Draw(picture).rectangle((10, 10, 30, 30), fill='white')
            picture.save(image)
            shape = {'id': 's', 'type': 'rectangle', 'label': 'part', 'x': 8, 'y': 8, 'width': 25, 'height': 25}
            aid = store.add_assets(pid, [{'path': image, 'shapes': [shape]}])['asset_ids'][0]
            payload = {'asset_id': aid, 'revision': store.get_asset(pid, aid)['revision'], 'label': 'part', 'box': [8, 8, 33, 33]}
            marker = self.root / 'gpu-acquired'
            code = f"import time;from pathlib import Path;from workbench.resources import accelerator_lease\nwith accelerator_lease({str(service.data_root)!r}):\n Path({str(marker)!r}).touch()\n time.sleep(30)"
            child = subprocess.Popen([sys.executable, '-c', code], cwd=Path(__file__).parents[1],
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(marker.exists())
            sam = service.ai_job(pid, dict(payload, engine='sam2'))
            cpu = service.ai_job(pid, dict(payload, engine='grabcut'))
            deadline = time.monotonic() + 5
            while service.jobs.get(cpu['id'])['state'] not in {'succeeded', 'failed'} and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(service.jobs.get(cpu['id'])['state'], 'succeeded')
            self.assertIn(service.jobs.get(sam['id'])['state'], {'queued', 'running'})
            service.jobs.control(sam['id'], 'cancel')
        finally:
            service.close()
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait()

    def test_delete_collects_unreferenced_only_and_trash_restore_error(self):
        store = ProjectStore(self.root / 'projects')
        pid = store.create_project('cleanup')['id']
        shape = {'id': 's', 'type': 'rectangle', 'label': 'p', 'x': 1, 'y': 1, 'width': 4, 'height': 4}
        ids = []
        for color in ('red', 'blue'):
            image = self.root / f'{color}.png'
            Image.new('RGB', (10, 10), color).save(image)
            ids.append(store.add_assets(pid, [{'path': image, 'shapes': [shape]}])['asset_ids'][0])
        a, b = ids
        from workbench.review_repository import ReviewRepository
        repository = ReviewRepository(store)
        repository.trash(pid, [b], {b: store.get_asset(pid, b)['revision']})
        trash_history = store.history(pid, b)[-1]
        with self.assertRaisesRegex(ValueError, '不含標註'):
            store.restore(pid, b, trash_history['id'], 1)
        with patch.object(store, '_touch', side_effect=RuntimeError('rollback')):
            with self.assertRaises(RuntimeError):
                store.delete_asset(pid, a, store.get_asset(pid, a)['revision'])
        self.assertTrue(store.get_asset(pid, a))
        store.delete_asset(pid, a, store.get_asset(pid, a)['revision'])
        with store.connection(pid) as db:
            for table in ('asset_summaries', 'annotation_revisions', 'asset_review', 'asset_quality'):
                self.assertEqual(db.execute(f'SELECT COUNT(*) FROM {table} WHERE asset_id=?', (a,)).fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM annotation_blobs').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM annotation_revisions WHERE asset_id=?', (b,)).fetchone()[0], 1)
        revision = repository.list_trash(pid)[0]['revision']
        repository.trash(pid, [b], {b: revision}, restore=True)
        store.delete_asset(pid, b, store.get_asset(pid, b)['revision'])
        with store.connection(pid) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM annotation_blobs').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
