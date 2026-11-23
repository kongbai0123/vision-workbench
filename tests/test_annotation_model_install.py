from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from workbench import annotation_model_install as installer
from workbench.annotation_assist_runtime import MODEL_ENV, model_home
from workbench.jobs import JobCancelled


class AnnotationModelInstallTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory())).resolve()
        self.app = self.root / 'target'
        self.app.mkdir()
        self.source_app = self.root / 'source'
        self.source = self.source_app / installer.INSTALL_PATH
        self.home = self.app / installer.INSTALL_PATH
        self.payload = b'verified model'
        pins = {'weight.bin': sha256(self.payload).hexdigest()}
        self.runtime = installer.Component('runtime', 'CUDA Python', 'runtime', {'python.exe': None}, required=True)
        self.primary = installer.Component('test-mm', 'MM Tiny', 'MM-Tiny', pins, required=True)
        self.optional = installer.Component('test-opt', 'Optional', '_runtimes/tools', pins)
        self.components = (self.runtime, self.primary, self.optional)
        self.enterContext(patch.object(installer, 'BATCH_COMPONENTS', self.components))
        self.enterContext(patch.dict(os.environ, {MODEL_ENV: ''}))
        self.write(self.source / 'runtime/python.exe', b'portable runtime')
        self.write(self.source / 'runtime/Lib/empty.py', b'')
        self.write(self.source / 'runtime/Scripts/pip.exe', b'absolute Windows launcher')
        self.write(self.source / 'runtime/Lib/__pycache__/cache.pyc', b'cache')
        self.write(self.source / 'MM-Tiny/weight.bin', self.payload)
        self.write(self.source / 'inactive-model/weight.bin', b'inactive')

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def install(self, source=None, progress=lambda *_: None):
        return installer.install_annotation_models(self.app, source or self.source_app, progress)

    def test_portable_install_copies_active_files_and_keeps_ready_groups(self):
        self.write(self.source / '_runtimes/tools/weight.bin', self.payload)
        result = self.install()
        self.assertEqual(result['installed'], ['runtime', 'test-mm', 'test-opt'])
        self.assertEqual((self.home / 'runtime/Lib/empty.py').read_bytes(), b'')
        self.assertFalse((self.home / 'runtime/Scripts').exists())
        self.assertFalse((self.home / 'runtime/Lib/__pycache__').exists())
        self.assertFalse((self.home / 'inactive-model').exists())
        self.assertEqual(result['install_path'], installer.INSTALL_PATH)
        self.assertEqual([row['state'] for row in result['components']],
                         ['available', 'available', 'available', 'missing'])
        custom = self.home / 'MM-Tiny/user-note.txt'
        self.write(custom, b'keep me')
        result = self.install(self.source)
        self.assertEqual(result['installed'], [])
        self.assertEqual(custom.read_bytes(), b'keep me')

    def test_incomplete_component_is_preserved_in_a_backup(self):
        self.write(self.home / 'MM-Tiny/user-note.txt', b'keep original')
        result = self.install()
        backup, = result['backups']
        self.assertEqual((self.app / backup / 'user-note.txt').read_bytes(), b'keep original')
        self.assertEqual((self.home / 'MM-Tiny/weight.bin').read_bytes(), self.payload)

    def test_bad_hash_or_cancellation_leaves_no_partially_installed_group(self):
        self.write(self.source / 'MM-Tiny/weight.bin', b'bad weight!')
        with self.assertRaisesRegex(ValueError, '雜湊不符'):
            self.install()
        self.assertFalse((self.home / 'runtime').exists())
        self.assertEqual(list(self.home.glob('.annotation-install-*')), [])
        self.write(self.source / 'MM-Tiny/weight.bin', self.payload)

        def stop(message, percent):
            if percent == 95:
                raise JobCancelled('cancel')

        with self.assertRaises(JobCancelled):
            self.install(progress=stop)
        self.assertFalse((self.home / 'runtime').exists())
        self.assertEqual(self.install()['installed'], ['runtime', 'test-mm'])

    def test_failed_publish_restores_incomplete_original(self):
        original = self.home / 'MM-Tiny/user-note.txt'
        self.write(original, b'keep original')
        rename = Path.rename

        def fail_publish(path, target):
            if target == original.parent and path.parent.name.startswith('.annotation-install-'):
                raise OSError('blocked')
            return rename(path, target)

        with patch.object(Path, 'rename', fail_publish), self.assertRaisesRegex(OSError, 'blocked'):
            self.install()
        self.assertEqual(original.read_bytes(), b'keep original')
        self.assertEqual(list(self.home.glob('.annotation-install-*')), [])
        self.assertEqual(self.install()['installed'], ['test-mm'])

    def test_relative_override_and_inaccessible_catalog(self):
        with patch.dict(os.environ, {MODEL_ENV: 'shared/models'}):
            self.assertEqual(model_home(self.app), self.app / 'shared/models')
            catalog = installer.annotation_model_catalog(self.app)
            self.assertTrue(catalog['custom_location'])
            self.assertFalse(catalog['installable'])
            with self.assertRaisesRegex(ValueError, '自訂模型位置'):
                self.install()
        with patch.object(installer, 'model_home', side_effect=PermissionError):
            catalog = installer.annotation_model_catalog(self.app)
            self.assertFalse(catalog['installable'])
            self.assertEqual(catalog['components'][0]['state'], 'incomplete')

    def test_qwen_requires_the_exact_source_manifest_and_its_model_files(self):
        qwen = self.source / installer.QWEN.directory
        name = 'weight.safetensors'
        self.write(qwen / name, self.payload)
        manifest = json.dumps({'files': {name: {'bytes': len(self.payload), 'sha256': sha256(self.payload).hexdigest()}}}).encode()
        self.write(qwen / installer.QWEN_MANIFEST, manifest)
        with patch.object(installer, 'BATCH_COMPONENTS', (installer.QWEN,)), \
             patch.object(installer, 'QWEN_PROVENANCE_SHA256', sha256(manifest).hexdigest()):
            self.assertEqual(self.install()['installed'], ['qwen-4b'])
            self.assertEqual((self.home / installer.QWEN.directory / installer.QWEN_MANIFEST).read_bytes(), manifest)
            self.write(self.home / installer.QWEN.directory / installer.QWEN_MANIFEST, manifest + b'\n')
            self.assertEqual(installer.annotation_model_catalog(self.app)['components'][0]['state'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
