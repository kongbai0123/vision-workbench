"""04–06 contract checks. No training worker may be started by this suite."""
import json
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase
from unittest.mock import patch, Mock
from tests import test_training as fixtures
from workbench.store import ConflictError, ProjectStore
from workbench.workflow_drafts import WorkflowDrafts
from workbench.augmentation import normalize_augmentation
from workbench.training_parameters import create_optimizer


class SetupWorkflowTests(TestCase):
    setUp = fixtures.TrainingWorkflowTests.setUp
    tearDown = fixtures.TrainingWorkflowTests.tearDown

    def test_project_drafts_persist_isolate_and_reject_lost_updates(self):
        drafts = WorkflowDrafts(self.store)
        other = self.store.create_project('other')['id']
        revision = self.store.get_project(self.pid)['revision']
        payload = {'augmentation': {'preset': 'custom', 'expansion_count': '2', 'brightness': ''},
                   'parameters': {'yolo26n_detect': {'batch_size': '4', 'learning_rate': ''}}}
        saved = drafts.save(self.pid, 0, payload)
        self.assertEqual(saved['revision'], 1)
        self.assertEqual(WorkflowDrafts(ProjectStore(self.store.root)).read(self.pid)['payload'], payload)
        self.assertEqual(drafts.read(other)['payload'], {})
        with self.assertRaises(ConflictError):
            drafts.save(self.pid, 0, {})
        self.assertEqual(drafts.save(self.pid, 1, payload)['revision'], 1)
        drafts.save(self.pid, 1, {})
        with self.store.connection(self.pid) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM workflow_draft_history').fetchone()[0], 2)
            self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(list(db.execute('PRAGMA foreign_key_check')), [])
        self.assertEqual(self.store.get_project(self.pid)['revision'], revision)

    def test_dataset_readiness_is_calculated_from_exact_published_snapshot(self):
        revision = self.store.get_project(self.pid)['revision']
        with patch.object(self.workspace, 'readiness', side_effect=AssertionError('must not reread')):
            dataset = self.workspace.create_dataset_version(self.pid, {'preset': 'light', 'expansion_count': 2}, revision)
        manifest = json.loads((self.workspace.datasets_dir(self.pid) / dataset['id'] / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['readiness']['project_revision'], manifest['project_revision'])
        self.assertEqual(dataset['training_events']['events'], 6)
        with self.assertRaises(ConflictError):
            self.workspace.create_dataset_version(self.pid, expected_revision=revision-1)
        with self.assertRaises(ValueError):
            self.workspace.create_dataset_version(self.pid, expected_revision=True)

    def test_concurrent_publications_have_unique_complete_versions(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(lambda _: self.workspace.create_dataset_version(self.pid), range(2)))
        self.assertEqual({row['id'] for row in rows}, {'D001', 'D002'})
        for row in rows:
            self.assertEqual(row['asset_count'], 6)

    def test_preflight_matches_expansion_and_preserves_evaluation_and_creates_no_run(self):
        dataset = self.workspace.create_dataset_version(self.pid, {'preset': 'light', 'expansion_count': 2})
        from workbench.model_registry import MODELS
        model = lambda key: {**next(item for item in MODELS if item['key'] == key), 'train': False}
        with patch.object(self.workspace.registry, 'model', side_effect=model), patch('workbench.training.subprocess.Popen', side_effect=AssertionError('training forbidden')):
            report = self.workspace.training_preflight(self.pid, dataset['id'],
                {'engine': 'yolo26n_detect', 'batch_size': 4, 'gradient_accumulation': 3})
            baseline = self.workspace.training_preflight(self.pid, dataset['id'], {'engine': 'pixel_prototype_v1'})
        self.assertEqual(report['config']['augmentation']['expansion_count'], 2)
        self.assertEqual(report['summary']['training_events']['events'], 6)
        self.assertEqual(report['summary']['batches_per_epoch'], 2)
        self.assertEqual(report['summary']['effective_batch_size'], 12)
        self.assertEqual(report['summary']['splits'], {'train': 2, 'val': 2, 'test': 2})
        self.assertIn('contrast', report['summary']['ignored_augmentation_fields'])
        self.assertFalse(report['training_started'])
        self.assertEqual(baseline['summary']['training_events']['events'], 2)
        self.assertIsNone(baseline['summary']['batches_per_epoch'])
        self.assertFalse(baseline['summary']['augmentation_applied'])
        self.assertEqual(self.workspace.list_runs(self.pid), [])
        self.assertEqual(list(self.workspace.models_dir(self.pid).glob('M*')), [])

    def test_preflight_rejects_corruption_and_invalid_parameters(self):
        dataset = self.workspace.create_dataset_version(self.pid)
        with self.assertRaises(ValueError):
            self.workspace.training_preflight(self.pid, dataset['id'], {'engine': 'yolo26n_detect', 'image_size': 641})
        path = self.workspace.datasets_dir(self.pid) / dataset['id'] / 'manifest.json'
        original = path.read_bytes()
        content = json.loads(original)
        image = path.parent / content['assets'][0]['image_file']
        raw = image.read_bytes()
        image.write_bytes(b'corrupted fixture')
        with self.assertRaisesRegex(ValueError, '圖片雜湊'):
            self.workspace.training_preflight(self.pid, dataset['id'], {'engine': 'pixel_prototype_v1'})
        image.write_bytes(raw)
        from workbench.training import _canonical_hash
        content['assets'][0]['image_file'] = '../outside.png'
        content['manifest_sha256'] = _canonical_hash({k: v for k, v in content.items() if k != 'manifest_sha256'})
        path.write_text(json.dumps(content), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '路徑無效'):
            self.workspace.training_preflight(self.pid, dataset['id'], {'engine': 'pixel_prototype_v1'})
        path.write_bytes(original)
        content = json.loads(path.read_text(encoding='utf-8'));content['classes'].append('tampered')
        path.write_text(json.dumps(content), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'manifest 雜湊'):
            self.workspace.training_preflight(self.pid, dataset['id'], {'engine': 'pixel_prototype_v1'})

    def test_adapter_lengths_and_materialized_files_match_frozen_recipe_without_training(self):
        from workbench.maskrcnn_engine import NativeMaskDataset
        from workbench.torchvision_engines import SemanticDataset
        from workbench.classification_engine import ClassificationDataset
        from workbench.ultralytics_engine import prepare_yolo_dataset
        dataset = self.workspace.create_dataset_version(self.pid, {'preset': 'light', 'expansion_count': 2})
        root = self.workspace.datasets_dir(self.pid) / dataset['id']
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        profile = manifest['augmentation']
        for cls, args in ((NativeMaskDataset, ()), (SemanticDataset, (640,)), (ClassificationDataset, (224,))):
            for split, expected in [('train', 6), ('val', 2), ('test', 2)]:
                with self.subTest(adapter=cls.__name__, split=split):
                    adapter = cls(manifest, root, split, None, *args, augmentation=profile)
                    self.assertEqual(len(adapter), expected)
                    if split != 'train':
                        self.assertIsNone(adapter.augmentation)
        output = self.root / 'prepared-only'
        prepare_yolo_dataset(root / 'manifest.json', output, 'object_detection', {'augmentation': {'preset': 'off'}})
        for split, expected in [('train', 6), ('val', 2), ('test', 2)]:
            self.assertEqual(len(list((output / 'images' / split).iterdir())), expected)
            self.assertEqual(len(list((output / 'labels' / split).iterdir())), expected)
        self.assertEqual(self.workspace.list_runs(self.pid), [])

    def test_review_changes_invalidate_approval_but_not_fixed_version(self):
        dataset = self.workspace.create_dataset_version(self.pid)
        path = self.workspace.datasets_dir(self.pid) / dataset['id'] / 'manifest.json'
        before = path.read_bytes()
        asset = self.store.snapshot(self.pid)['assets'][0]
        self.store.save_asset(self.pid, asset['id'], [], asset['revision'])
        self.assertEqual(self.store.get_asset(self.pid, asset['id'])['review_state'], 'pending')
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.workspace.readiness(self.pid)['stats']['approved'], 5)

    def test_off_and_scope_cannot_silently_enable_augmentation(self):
        for profile in ({'preset': 'off', 'fliplr': .5}, {'preset': 'light', 'apply_to': 'test'},
                        {'preset': 'light', 'mode': 'offline'}, {'schema_version': 999}):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                normalize_augmentation(profile)

    def test_sgd_configuration_uses_documented_momentum_without_training(self):
        torch = Mock()
        create_optimizer(torch, [], {'optimizer': 'SGD'})
        self.assertEqual(torch.optim.SGD.call_args.kwargs['momentum'], .9)

    def test_all_implemented_training_schemas_validate_without_loading_models(self):
        from workbench.model_registry import MODELS
        from workbench.engine_specs import engine_spec
        from workbench.training import _canonical_hash
        dataset = self.workspace.create_dataset_version(self.pid, {'preset': 'standard', 'expansion_count': 2})
        path = self.workspace.datasets_dir(self.pid) / dataset['id'] / 'manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['classes'] = ['handlebar', 'other']
        # Two singly labelled examples in each split support classification too.
        seen = {}
        for asset in manifest['assets']:
            index = seen.get(asset['split'], 0);seen[asset['split']] = index + 1
            for shape in asset['shapes']:
                shape['label'] = manifest['classes'][index % 2]
        manifest['manifest_sha256'] = _canonical_hash({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        path.write_text(json.dumps(manifest), encoding='utf-8')
        verified = []
        for definition in MODELS:
            if definition['integration'] != 'ready' or definition.get('inference_only'):
                continue
            with self.subTest(engine=definition['key']), patch.object(self.workspace.registry, 'model', return_value={**definition, 'train': False}), patch('workbench.training.subprocess.Popen', side_effect=AssertionError('no process')):
                fields = engine_spec(definition['key']).parameter_schema()
                config = {field['key']: field['default'] for field in fields}
                report = self.workspace.training_preflight(self.pid, dataset['id'], {'engine': definition['key'], **config})
                self.assertTrue(report['validated'])
                self.assertFalse(report['training_started'])
                verified.append(definition['key'])
        self.assertGreaterEqual(len(verified), 8)
