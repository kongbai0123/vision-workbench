import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

from PIL import Image
from vision_workbench.contracts import validate_request, validate_run_event
from workbench.artifact_repository import ArtifactRepository
from workbench.editor_sync import commit_updates
from workbench.maintenance import orphan_images
from workbench.routes import validate_method, MethodNotAllowed
from workbench.resources import accelerator_lease
from workbench.run_events import latest_state
from workbench.store import ProjectStore
from workbench.training_engine import atomic_json, stop_requested


class ArchitectureContractsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = ProjectStore(self.root / 'projects')
        self.pid = self.store.create_project('migration')['id']
        self.source = self.root / 'image.png'
        Image.new('RGB', (32, 32), 'red').save(self.source)
        self.shape = {'id': 'box', 'type': 'rectangle', 'label': 'part', 'x': 2, 'y': 2, 'width': 10, 'height': 10}
        self.aid = self.store.add_assets(self.pid, [{'path': self.source, 'shapes': [self.shape]}])['asset_ids'][0]

    def tearDown(self):
        self.temporary.cleanup()

    def asset(self):
        return self.store.get_asset(self.pid, self.aid)

    def test_migrations_are_idempotent_and_read_only_requests_do_not_write(self):
        folder = self.store.directory(self.pid)
        ProjectStore(self.store.root)
        ProjectStore(self.store.root)
        with self.store.connection(self.pid) as db:
            versions = list(db.execute('SELECT version FROM schema_migrations'))
            self.assertEqual(len(versions), 4)
        before = (folder / 'project.sqlite3').stat().st_mtime_ns
        self.store.list_projects()
        self.store.get_project(self.pid)
        self.assertEqual(before, (folder / 'project.sqlite3').stat().st_mtime_ns)

    def test_history_deduplicates_geometry_and_restores_compatibility(self):
        for state in ['approved', 'pending', 'approved']:
            self.store.review(self.pid, [self.aid], state, {self.aid: self.asset()['revision']})
        with self.store.connection(self.pid) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM annotation_blobs').fetchone()[0], 1)
            rows = [json.loads(row[0]) for row in db.execute('SELECT data FROM history')]
            self.assertTrue(all('shapes' not in row for row in rows))
        self.assertEqual(self.store.history(self.pid, self.aid)[0]['data']['shapes'], self.asset()['shapes'])

    def test_review_quality_not_persisted_as_source_and_annotation_revision_stable(self):
        before = self.asset()['annotation_revision']
        self.store.save_quality(self.pid, {self.aid: {'suspected_blur': True}})
        self.store.review(self.pid, [self.aid], 'rejected', {self.aid: self.asset()['revision']}, reason='模糊')
        asset = self.asset()
        self.assertEqual(asset['annotation_revision'], before)
        self.assertTrue(asset['source']['quality']['suspected_blur'])
        with self.store.connection(self.pid) as db:
            source = json.loads(db.execute('SELECT source FROM assets WHERE id=?', (self.aid,)).fetchone()[0])
            self.assertNotIn('quality', source)
            self.assertNotIn('review', source)

    def test_external_editor_obeys_same_class_policy(self):
        before = self.asset()
        with self.assertRaises(ValueError):
            commit_updates(self.store, self.pid, [(before, [dict(self.shape, label='unknown')])], source='cvat')
        self.assertEqual(self.asset(), before)

    def test_split_metadata_does_not_reject_unchanged_annotation_revision(self):
        before = self.asset()
        self.store.assign(self.pid, [self.aid], split='train')
        self.store.save_asset(self.pid, self.aid, [dict(self.shape, x=3)], before['revision'])
        self.assertEqual(self.asset()['shapes'][0]['x'], 3)
        self.assertEqual(self.asset()['annotation_revision'], before['annotation_revision'] + 1)

    def test_delta_only_contains_changed_assets_and_stale_base_falls_back(self):
        second = self.root / 'second.png'
        Image.new('RGB', (32, 32), 'blue').save(second)
        self.store.add_assets(self.pid, [{'path': second}])
        before = self.store.get_project(self.pid)
        delta = self.store.review(self.pid, [self.aid], 'approved', {self.aid: self.asset()['revision']}, delta_base=before['revision'])
        self.assertEqual(delta['delta']['base_revision'], before['revision'])
        self.assertEqual([a['id'] for a in delta['assets']], [self.aid])
        full = self.store.assign(self.pid, [self.aid], split='train', delta_base=before['revision'])
        self.assertNotIn('delta', full)
        self.assertEqual(len(full['assets']), 2)

    def test_failed_delete_transaction_preserves_image(self):
        path = self.store.image_path(self.pid, self.aid)
        with patch.object(self.store, '_touch', side_effect=RuntimeError('commit failed')):
            with self.assertRaises(RuntimeError):
                self.store.delete_asset(self.pid, self.aid, self.asset()['revision'])
        self.assertTrue(path.is_file())
        self.assertEqual(self.asset()['id'], self.aid)

    def test_orphan_maintenance_is_recoverable_and_retains_trash(self):
        image = self.store.image_path(self.pid, self.aid)
        self.store.trash_assets(self.pid, [self.aid], {self.aid: self.asset()['revision']})
        orphan = image.parent / 'orphan.png'
        orphan.write_bytes(b'fixture')
        report = orphan_images(self.store, self.pid)
        self.assertEqual(report['files'], ['orphan.png'])
        self.assertTrue(orphan.exists())
        report = orphan_images(self.store, self.pid, quarantine=True)
        self.assertTrue(image.exists())
        self.assertEqual((Path(report['directory']) / orphan.name).read_bytes(), b'fixture')

    def test_events_ignore_partial_or_invalid_tail(self):
        path = self.root / 'run.json'
        atomic_json(path, {'status': 'running', 'epoch': 1})
        atomic_json(path, {'status': 'completed', 'epoch': 2})
        with path.with_name('events.jsonl').open('ab') as output:
            output.write(b'{"schema_version":1,"type":"state"}\n{"unfinished":')
        self.assertEqual(latest_state(path)['epoch'], 2)

    def test_dataset_summary_cache_avoids_reopening_manifest_and_defends_mutation(self):
        path = self.root / 'manifest.json'
        atomic_json(path, {'value': 1})
        repository = ArtifactRepository()
        summary = repository.dataset_summary(path, lambda data: {'values': [data['value']]}, persist=True)
        summary['values'].append(2)
        with patch('workbench.artifact_repository.read_json', side_effect=AssertionError('cache miss')):
            self.assertEqual(repository.dataset_summary(path, lambda _: None), {'values': [1]})
        reopened = ArtifactRepository()
        self.assertEqual(reopened.dataset_summary(path, lambda _: self.fail('recomputed')), {'values': [1]})

    def test_migration_backup_includes_wal_and_original_history(self):
        database = self.store.directory(self.pid) / 'project.sqlite3'
        with closing(sqlite3.connect(database)) as db, db:
            db.execute('DELETE FROM schema_migrations WHERE version=3')
            db.execute('DROP TRIGGER asset_summary_insert')
            db.execute('DROP TRIGGER asset_summary_update')
            db.execute('DROP TABLE asset_summaries')
        ProjectStore(self.store.root)
        backups = list(database.parent.glob('*.backup'))
        self.assertEqual(len(backups), 1)
        with closing(sqlite3.connect(backups[0])) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM assets').fetchone()[0], 1)

    def test_stop_control_and_legacy_marker(self):
        self.assertFalse(stop_requested(self.root))
        (self.root / 'control').mkdir()
        (self.root / 'control' / 'stop.requested').touch()
        self.assertTrue(stop_requested(self.root))

    def test_request_and_event_contracts(self):
        for payload in [{'name': []}, {'name': 'x', 'revision': True}, {'name': 'x', 'config': {'lr': float('nan')}}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_request('/api/projects', 'POST', payload)
        with self.assertRaises(ValueError):
            validate_run_event({'schema_version': 1, 'type': 'state', 'state': {'status': 'other'}})
        with self.assertRaises(MethodNotAllowed):
            validate_method('/api/projects/p/review', 'DELETE')

    def test_gpu_admission_is_exclusive_cancellable_and_cpu_is_independent(self):
        attempts = []
        def cancel():
            attempts.append(True)
            if len(attempts) > 1:
                raise InterruptedError('cancelled while waiting')
        with accelerator_lease(self.root):
            with accelerator_lease(self.root, device='cpu'):
                pass
            with self.assertRaises(InterruptedError):
                with accelerator_lease(self.root, checkpoint=cancel):
                    self.fail('GPU lease admitted two owners')
        with accelerator_lease(self.root):
            pass

    def test_cancel_before_training_does_not_become_failure(self):
        from workbench.training_worker import main
        run_dir = self.root / 'run'
        run_dir.mkdir()
        atomic_json(run_dir / 'run.json', {'status': 'queued', 'engine': 'pixel_prototype_v1', 'config': {'device': 'auto'}})
        (run_dir / 'control').mkdir()
        (run_dir / 'control' / 'stop.requested').touch()
        with patch.dict('os.environ', {'VISION_WORKBENCH_RESOURCE_ROOT': str(self.root)}):
            self.assertEqual(main(['--dataset', str(self.root / 'unused.json'), '--run-dir', str(run_dir),
                                   '--model-dir', str(self.root / 'unused-model')]), 0)
        self.assertEqual(latest_state(run_dir / 'run.json')['status'], 'stopped')
        self.assertTrue((run_dir / 'heartbeat.json').is_file())


if __name__ == '__main__':
    unittest.main()
