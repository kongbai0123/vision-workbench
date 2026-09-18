from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile
import sys
from types import SimpleNamespace
from unittest.mock import patch

from workbench.store import ProjectStore
from workbench.training import TrainingWorkspace
from workbench.training_engine import atomic_json, read_json


class ModelImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ProjectStore(self.root / 'projects')
        self.project = self.store.create_project('外部模型')
        self.pid = self.project['id']
        self.workspace = TrainingWorkspace(self.root, self.store)
        self.checkpoint = self.root / 'best.pt'
        self.checkpoint.write_bytes(b'synthetic checkpoint; worker mocked')
        self.workspace.registry.component_status = lambda *args, **kwargs: {'ultralytics': {'state': 'ready'}}
        self.workspace.registry.model = lambda key, **kwargs: {'key': key, 'name': '外部 YOLO', 'inference_only': True,
                                                               'task': 'object_detection', 'predict': True, 'component': 'ultralytics'}

    def tearDown(self):
        self.workspace.close()
        self.temp.cleanup()

    @staticmethod
    def inspect(command, *_args, **_kwargs):
        atomic_json(Path(command[command.index('--output') + 1]), {
            'engine': 'external_yolo_detect', 'task': 'object_detection', 'classes': ['part'], 'architecture': 'DetectionModel'})

    def test_import_is_owned_by_project_without_fabricated_training_or_scores(self):
        with patch('workbench.process_control.run_controlled', self.inspect):
            result = self.workspace.import_model(self.pid, str(self.checkpoint), '零件偵測', True)
            second = self.workspace.import_model(self.pid, str(self.checkpoint), '另一模型', True)
        self.assertNotEqual(result['model_version_id'], second['model_version_id'])
        self.checkpoint.unlink()
        model = self.workspace.model(self.pid, result['model_version_id'])
        self.assertIsNone(model['run_id'])
        self.assertIsNone(model['dataset_version_id'])
        self.assertIsNone(model['validation'])
        self.assertEqual(model['classes'], ['part'])
        self.assertEqual(len(model['source']['sha256']), 64)
        self.assertTrue((self.workspace.models_dir(self.pid) / result['model_version_id'] / 'checkpoint.pt').is_file())
        self.assertEqual(self.workspace.catalog_issues[self.pid], [])
        exported = self.workspace.export_model(self.pid, model['model_version_id'])
        with zipfile.ZipFile(exported['path']) as archive:
            self.assertIn('model/checkpoint.pt', archive.namelist())
            self.assertNotIn('run/run.json', archive.namelist())
        self.assertEqual(self.workspace.catalog_issues[self.pid], [])
        with self.assertRaisesRegex(ValueError, '固定資料版本'):
            self.workspace.create_model_comparison(self.pid, model['model_version_id'])
        with self.assertRaisesRegex(ValueError, '加入模型類別'):
            self.workspace.create_predictions(self.pid, model['model_version_id'])
        from PIL import Image
        image = self.root / 'sample.png'
        Image.new('RGB', (10, 10)).save(image)
        self.store.update_project(self.pid, classes=['part'])
        self.store.add_assets(self.pid, [{'path': str(image), 'name': image.name}])
        def predict(command, *_args, **_kwargs):
            request = read_json(Path(command[command.index('--request')+1]))
            atomic_json(Path(command[command.index('--output')+1]), {'assets': [
                {'asset_id': asset['asset_id'], 'shapes': []} for asset in request['assets']]})
        with patch('workbench.process_control.run_controlled', predict):
            candidates = self.workspace.create_predictions(self.pid, model['model_version_id'])
        self.assertIsNone(candidates['run_id'])
        self.assertEqual(self.workspace.catalog_issues[self.pid], [])

    def test_failed_worker_leaves_no_model_or_partial_import(self):
        with patch('workbench.process_control.run_controlled', side_effect=RuntimeError('invalid checkpoint')):
            with self.assertRaisesRegex(RuntimeError, 'invalid checkpoint'):
                self.workspace.import_model(self.pid, str(self.checkpoint), trusted=True)
        self.assertEqual(self.workspace.list_models(self.pid), [])
        self.assertEqual(list(self.workspace.models_dir(self.pid).iterdir()), [])

    def test_failed_catalog_publication_removes_new_model(self):
        with patch('workbench.process_control.run_controlled', self.inspect), patch.object(self.workspace, '_sync_catalog', side_effect=RuntimeError('catalog failure')):
            with self.assertRaisesRegex(RuntimeError, 'catalog failure'):
                self.workspace.import_model(self.pid, str(self.checkpoint), trusted=True)
        self.assertEqual(self.workspace.list_models(self.pid), [])
        self.assertEqual(list(self.workspace.models_dir(self.pid).iterdir()), [])

    def test_trust_and_runtime_checks_happen_before_loading(self):
        with patch('workbench.process_control.run_controlled') as worker:
            with self.assertRaisesRegex(ValueError, '可信任'):
                self.workspace.import_model(self.pid, str(self.checkpoint))
            self.workspace.registry.component_status = lambda: {'ultralytics': {'state': 'not_installed'}}
            with self.assertRaisesRegex(ValueError, '安裝或修復'):
                self.workspace.import_model(self.pid, str(self.checkpoint), trusted=True)
            worker.assert_not_called()

    def test_migration_keeps_existing_foreign_keys_and_accepts_external_lineage(self):
        migrations = Path(__file__).parents[1] / 'workbench/migrations'
        db = sqlite3.connect(':memory:')
        try:
            db.execute('PRAGMA foreign_keys=ON')
            db.executescript((migrations / '0000_core.sql').read_text())
            db.execute('INSERT INTO dataset_versions VALUES(?,?,?,?,?,?,?)', ('D001', 1, 'now', 'a'*64, 1, 'd', None))
            db.execute('INSERT INTO training_runs VALUES(?,?,?,?,?,?,?,?)', ('R001', 'D001', 'M001', 'engine', 'completed', 1, 1, 'r'))
            db.execute('INSERT INTO model_versions VALUES(?,?,?,?,?,?)', ('M001', 'R001', 'D001', 'engine', 1, 'm'))
            db.execute('INSERT INTO model_exports VALUES(?,?,?,?,?,?,?,?)', ('E001', 'M001', 'R001', 'D001', 'now', 'b'*64, 1, 'e'))
            db.execute('INSERT INTO prediction_candidates VALUES(?,?,?,?,?,?,?)', ('c'*32, 'M001', 'R001', 'now', 0, 0, 'p'))
            db.executescript((migrations / '0005_external_models.sql').read_text())
            self.assertEqual(db.execute('SELECT run_id FROM model_versions').fetchone()[0], 'R001')
            db.execute('INSERT INTO model_versions VALUES(?,?,?,?,?,?)', ('M002', None, None, 'external_yolo_detect', 2, 'm2'))
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
        finally:
            db.close()


class ModelInspectionTests(unittest.TestCase):
    def test_detection_segmentation_and_rtdetr_use_correct_adapter(self):
        from workbench.model_import_worker import inspect_checkpoint
        class RTDETRDetectionModel:
            pass
        for task, rtdetr, expected in [('detect', False, 'external_yolo_detect'),
                                      ('segment', False, 'external_yolo_segment'),
                                      ('detect', True, 'rt_detr_external')]:
            calls=[]
            model=SimpleNamespace(task=task, names={0:'part'}, model=RTDETRDetectionModel() if rtdetr else object(),
                                  predict=lambda **kwargs:calls.append(kwargs))
            module=SimpleNamespace(YOLO=lambda path:model, RTDETR=lambda path:model)
            with patch.dict(sys.modules, {'ultralytics':module, 'ultralytics.nn.tasks':SimpleNamespace(RTDETRDetectionModel=RTDETRDetectionModel)}):
                result=inspect_checkpoint(Path('synthetic.pt'))
            self.assertEqual(result['engine'],expected)
            self.assertEqual(calls[0]['device'],'cpu')
            self.assertEqual(calls[0]['source'].shape,(640,640,3))

    def test_unsupported_tasks_are_rejected_before_inference(self):
        from workbench.model_import_worker import inspect_checkpoint
        model=SimpleNamespace(task='pose',model=object())
        module=SimpleNamespace(YOLO=lambda path:model,RTDETR=lambda path:model)
        with patch.dict(sys.modules, {'ultralytics':module, 'ultralytics.nn.tasks':SimpleNamespace(RTDETRDetectionModel=type('RTModel',(),{}))}):
            with self.assertRaisesRegex(ValueError,'不支援'):
                inspect_checkpoint(Path('synthetic.pt'))
