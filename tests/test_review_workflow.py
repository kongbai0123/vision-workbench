import tempfile
import unittest
from pathlib import Path

from PIL import Image

from workbench.store import ProjectStore, ConflictError
from workbench.review_workflow import ReviewWorkflow


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ProjectStore(self.root / 'projects')
        self.pid = self.store.create_project('review')['id']
        self.flow = ReviewWorkflow(self.store)
        self.images = []
        for i in range(2):
            path = self.root / f'{i}.png'
            Image.new('RGB', (32, 32), (50+i*100, 20, 30)).save(path)
            self.images.append(path)

    def tearDown(self):
        self.temp.cleanup()

    def test_preview_selection_and_source_change(self):
        preview = self.flow.preview(self.pid, [str(p) for p in self.images])
        self.assertEqual(self.store.get_project(self.pid)['total'], 0)
        self.assertTrue(preview['items'][0]['suspected_blur'])
        result = self.flow.commit_import(self.pid, preview['token'], ['1'])
        asset = self.store.get_asset(self.pid, result['asset_ids'][0])
        self.assertEqual(asset['name'], '1.png')
        self.assertTrue(asset['source']['quality']['suspected_blur'])
        preview = self.flow.preview(self.pid, [str(self.images[0])])
        Image.new('RGB', (32, 32), 'white').save(self.images[0])
        with self.assertRaises(ConflictError):
            self.flow.commit_import(self.pid, preview['token'], ['0'])

    def test_import_reports_measured_phases_and_commit_wait(self):
        updates, phases, phase = [], [], ['none']
        def callback(message, percent=None, **metadata):
            phase[0] = metadata.get('phase', phase[0])
            updates.append((message, percent))
            phases.append(phase[0])
        preview = self.flow.preview(self.pid, [str(p) for p in self.images], progress=callback)
        scan = [p for (_, p), name in zip(updates, phases) if name == 'scan' and p is not None]
        self.assertEqual(scan, sorted(scan))
        self.assertEqual((scan[0], scan[-1]), (0, 100))
        self.assertEqual([p for (_, p), name in zip(updates, phases) if name == 'quality' and p is not None], [0, 50, 100])
        updates.clear()
        phases.clear()
        result = self.flow.commit_import(self.pid, preview['token'], ['0', '1'], progress=callback)
        self.assertEqual(result['added'], 2)
        self.assertEqual([p for _, p in updates if p is not None], [0, 50, 100, 0, 50, 100])
        self.assertIsNone(updates[-1][1])
        self.assertIn('寫入資料庫', updates[-1][0])

    def test_trash_restore_keeps_image_history_and_resets_approval(self):
        result = self.store.add_assets(self.pid, [{'path': self.images[0], 'shapes': []}])
        aid = result['asset_ids'][0]
        project = self.store.review(self.pid, [aid], 'rejected', {aid: 1}, '模糊', '失焦')
        asset = project['assets'][0]
        self.assertEqual(asset['source']['review']['reason'], '模糊')
        path = self.store.image_path(self.pid, aid)
        original = path.read_bytes()
        with self.assertRaises(ConflictError):
            self.flow.trash(self.pid, [aid], {aid: 1})
        self.flow.trash(self.pid, [aid], {aid: asset['revision']})
        self.assertEqual(self.store.get_project(self.pid)['total'], 0)
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(self.images[0].exists())
        entry = self.flow.list_trash(self.pid)[0]
        project = self.flow.trash(self.pid, [aid], {aid: entry['revision']}, restore=True)
        self.assertEqual(project['assets'][0]['review_state'], 'pending')
        self.assertGreater(project['assets'][0]['revision'], asset['revision'])
        self.assertEqual(self.flow.list_trash(self.pid), [])
        self.assertTrue(self.store.history(self.pid, aid))

    def test_batch_trash_rolls_back_on_stale_revision(self):
        result = self.store.add_assets(self.pid, [{'path': p, 'shapes': []} for p in self.images])
        ids = result['asset_ids']
        with self.assertRaises(ConflictError):
            self.flow.trash(self.pid, ids, {ids[0]: 1, ids[1]: 0})
        self.assertEqual(self.store.get_project(self.pid)['total'], 2)
        self.assertEqual(self.flow.list_trash(self.pid), [])

    def test_delete_reimport_preserves_trashed_image(self):
        record = {'path': self.images[0], 'shapes': []}
        aid = self.store.add_assets(self.pid, [record])['asset_ids'][0]
        original = self.store.image_path(self.pid, aid).read_bytes()
        self.flow.trash(self.pid, [aid], {aid: 1})
        duplicate = self.store.add_assets(self.pid, [record])['asset_ids'][0]
        self.store.delete_asset(self.pid, duplicate, 1)
        self.flow.trash(self.pid, [aid], {aid: 1}, restore=True)
        self.assertEqual(self.store.image_path(self.pid, aid).read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
