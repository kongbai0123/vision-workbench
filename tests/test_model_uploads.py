from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from workbench.model_uploads import ModelUploadStore


class ModelUploadStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.uploads = ModelUploadStore(self.root)

    def tearDown(self):
        self.uploads.close()
        self.temporary.cleanup()

    def test_streamed_upload_is_single_use_and_preserves_browser_filename(self):
        payload = b'synthetic checkpoint'
        result = self.uploads.save(BytesIO(payload), len(payload), 'trained part model.pt')
        path, filename = self.uploads.claim(result['upload_token'])
        self.assertEqual(filename, 'trained part model.pt')
        self.assertEqual(path.read_bytes(), payload)
        with self.assertRaisesRegex(ValueError, '失效|使用'):
            self.uploads.claim(result['upload_token'])
        self.uploads.discard(path)
        self.assertFalse(path.exists())

    def test_invalid_or_incomplete_upload_leaves_no_partial_file(self):
        with self.assertRaisesRegex(ValueError, '.pt'):
            self.uploads.save(BytesIO(b'bad'), 3, 'weights.onnx')
        with self.assertRaisesRegex(ValueError, '不完整'):
            self.uploads.save(BytesIO(b'short'), 20, 'weights.pt')
        self.assertEqual(list(self.uploads.root.iterdir()), [])
