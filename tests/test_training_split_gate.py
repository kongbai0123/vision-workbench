"""Split safety at the launch boundary, without starting training workers."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from workbench.split_quality import split_class_coverage
from workbench.store import ProjectStore
from workbench.training import TrainingWorkspace


class TrainingSplitGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = TrainingWorkspace.__new__(TrainingWorkspace)
        self.workspace.datasets = self.root / 'datasets'
        self.workspace.runs = self.root / 'runs'
        self.workspace.models = self.root / 'models'
        self.workspace.lock = threading.RLock()
        self.workspace.processes = {}
        self.workspace.list_runs = Mock(return_value=[])
        self.workspace.store = Mock()
        self.workspace.registry = Mock()
        self.workspace.registry.component_python.return_value = sys.executable
        self.workspace.registry.model.side_effect = lambda key: {
            'key': key, 'name': key, 'train': True, 'component': 'builtin',
            'task': 'instance_segmentation'}

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self, dataset_id, assets, *, copy_images=False):
        target = self.workspace.datasets / 'pid' / dataset_id
        target.mkdir(parents=True)
        rows = json.loads(json.dumps(assets))
        if copy_images:
            (target / 'images').mkdir()
            for row in rows:
                payload = row['asset_id'].encode()
                row['image_file'] = f"images/{row['asset_id']}.png"
                row['sha256'] = hashlib.sha256(payload).hexdigest()
                (target / row['image_file']).write_bytes(payload)
        manifest = {'dataset_version_id': dataset_id, 'project_revision': 1,
                    'created_at': '2026-09-15T00:00:00', 'manifest_sha256': 'original-manifest',
                    'classes': sorted({s['label'] for a in rows for s in a['shapes']}), 'assets': rows}
        (target / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return target

    def start_without_worker(self, dataset_id):
        with patch('workbench.training.subprocess.Popen') as popen, \
                patch('workbench.training.threading.Thread') as reaper:
            popen.return_value.pid = 12345
            run = self.workspace.start_run('pid', dataset_id, {'engine': 'pixel_prototype_v1', 'epochs': 2})
            popen.assert_called_once()
            reaper.return_value.start.assert_called_once()
        return run

    def test_old_missing_class_dataset_is_rejected_before_directories_or_process(self):
        rows = [{'asset_id': str(i), 'split': split, 'shapes': [{'label': label}]}
                for i, (split, label) in enumerate((('train', 'a'), ('train', 'b'), ('val', 'c'), ('test', 'd')))]
        source = self.write_manifest('D004', rows)
        original = (source / 'manifest.json').read_bytes()
        for engine in ('pixel_prototype_v1', 'maskrcnn_resnet50_fpn', 'yolo26n_seg'):
            with self.subTest(engine=engine), patch('workbench.training.subprocess.Popen') as popen, \
                    patch('workbench.training.threading.Thread') as reaper:
                with self.assertRaisesRegex(ValueError, 'Train.*c'):
                    self.workspace.start_run('pid', 'D004', {'engine': engine, 'epochs': 2})
                popen.assert_not_called()
                reaper.assert_not_called()
                self.assertFalse(self.workspace.runs.exists())
                self.assertFalse(self.workspace.models.exists())
                self.assertEqual(self.workspace.processes, {})
        self.assertEqual((source / 'manifest.json').read_bytes(), original)

    def test_diagnostic_new_version_preserves_source_and_carries_warning_to_run(self):
        rows = [{'asset_id': f'{label}-{i}', 'name': f'sample_{i:06d}.png', 'batch_id': label,
                 'split': 'train' if label in ('a', 'b') else 'val' if label == 'c' else 'test',
                 'shapes': [{'label': label}]}
                for label in ('a', 'b', 'c', 'd') for i in range(13)]
        source = self.write_manifest('D004', rows, copy_images=True)
        original = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob('*') if path.is_file()}
        created = self.workspace.create_diagnostic_dataset_version('pid', 'D004')
        self.assertEqual(created['id'], 'D005')
        target = source.parent / 'D005'
        manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
        coverage = split_class_coverage(manifest['assets'])
        self.assertTrue(coverage['ready'])
        self.assertEqual(coverage['warnings'], [])
        self.assertEqual(coverage['image_counts'], {'train': 28, 'val': 8, 'test': 8})
        for split in ('train', 'val', 'test'):
            self.assertTrue(all(coverage['class_counts'][split][label] for label in ('a', 'b', 'c', 'd')))
        for asset in manifest['assets']:
            self.assertEqual(hashlib.sha256((target / asset['image_file']).read_bytes()).hexdigest(), asset['sha256'])
        run = self.start_without_worker('D005')
        self.assertEqual(run['data_quality'], manifest['data_quality'])
        self.assertEqual(run['data_quality']['purpose'], 'diagnostic')
        self.assertFalse(run['data_quality']['independent_sources'])
        self.assertEqual(run['data_quality']['source_dataset_version_id'], 'D004')
        self.assertEqual(len(run['data_quality']['excluded_assets']), 8)
        self.assertTrue(any('不代表' in warning for warning in run['data_quality']['warnings']))
        persisted = json.loads((self.workspace.runs / 'pid' / run['run_id'] / 'run.json').read_text(encoding='utf-8'))
        self.assertEqual(persisted['data_quality'], run['data_quality'])
        self.assertEqual({str(path.relative_to(source)): path.read_bytes() for path in source.rglob('*') if path.is_file()}, original)
        self.workspace.store.snapshot.assert_not_called()
        self.workspace.store.assign.assert_not_called()

    def test_incomplete_validation_coverage_blocks_launch_and_test_gap_is_visible(self):
        rows = [{'asset_id': str(i), 'split': split, 'shapes': [{'label': label}]}
                for i, (split, label) in enumerate((('train', 'a'), ('train', 'b'), ('val', 'a'), ('test', 'b')))]
        self.write_manifest('D001', rows)
        readiness = self.workspace.dataset('pid', 'D001')['readiness']
        self.assertFalse(readiness['ready'])
        self.assertTrue(any(item['code'] == 'validation_class_missing' and item['label'] == 'b'
                            for item in readiness['blockers']))
        self.assertTrue(any(item['code'] == 'evaluation_class_missing' and item['split'] == 'test'
                            and item['label'] == 'a' for item in readiness['warnings']))
        with self.assertRaisesRegex(ValueError, 'Validation'):
            self.workspace.start_run('pid', 'D001', {'engine': 'pixel_prototype_v1', 'epochs': 2})

    def test_test_cannot_replace_validation_at_launch(self):
        rows = [{'asset_id': 'train', 'split': 'train', 'shapes': [{'label': 'part'}]},
                {'asset_id': 'test', 'split': 'test', 'shapes': [{'label': 'part'}]}]
        self.write_manifest('D006', rows)
        readiness = self.workspace.dataset('pid', 'D006')['readiness']
        self.assertFalse(readiness['ready'])
        self.assertTrue(any(item['code'] == 'no_validation_split' for item in readiness['blockers']))
        with patch('workbench.training.subprocess.Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'Validation'):
                self.workspace.start_run('pid', 'D006', {'engine': 'pixel_prototype_v1', 'epochs': 2})
            popen.assert_not_called()

    def test_explicit_loose_plan_launches_and_preserves_warnings(self):
        rows = [{'asset_id': str(i), 'split': split, 'shapes': [{'label': label}]}
                for i, (split, label) in enumerate((('train', 'a'), ('val', 'b'), ('test', 'c')))]
        source = self.write_manifest('D007', rows)
        path = source / 'manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['split_plan'] = {'strategy': 'random_loose',
                                  'assignments': {a['asset_id']: a['split'] for a in rows}}
        path.write_text(json.dumps(manifest), encoding='utf-8')
        run = self.start_without_worker('D007')
        self.assertIn('loose_split', [item['code'] for item in run['split_warnings']])

    def test_blocked_smart_plan_cannot_mutate_project_even_with_valid_fingerprint(self):
        store = ProjectStore(self.root / 'projects')
        project_id = store.create_project('split safety')['id']
        records = []
        for index in range(4):
            source = self.root / f'capture-{index}.png'
            Image.new('RGB', (10, 10), (index * 30, 20, 60)).save(source)
            records.append({'path': str(source), 'batch_id': str(index), 'review_state': 'approved',
                            'split': 'train', 'shapes': [{'type': 'rectangle', 'label': f'class-{index}',
                                                        'x': 1, 'y': 1, 'width': 4, 'height': 4}]})
        store.add_assets(project_id, records)
        before = store.snapshot(project_id)
        plan = store.preview_split(project_id, {})
        self.assertFalse(plan['ready'])
        with self.assertRaisesRegex(ValueError, 'Train'):
            store.apply_split(project_id, {}, plan['project_revision'], plan['fingerprint'])
        after = store.snapshot(project_id)
        self.assertEqual({k: v for k, v in before.items() if k != 'snapshot_at'},
                         {k: v for k, v in after.items() if k != 'snapshot_at'})


if __name__ == '__main__':
    unittest.main()
