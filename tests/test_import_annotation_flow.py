"""Import preview and commit retain labels from a portable YOLO export."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from workbench.materials import MaterialCatalog
from workbench.pipeline import export_project
from workbench.project_layout import project_area
from workbench.review_workflow import ReviewWorkflow
from workbench.store import ConflictError, ProjectStore


class ImportAnnotationFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vision-import-annotations-')
        self.root = Path(self.temp.name)
        self.image = self.root / 'original.png'
        Image.new('RGB', (80, 60), (120, 130, 140)).save(self.image)
        validation_image = self.root / 'validation.png'
        Image.new('RGB', (80, 60), (140, 130, 120)).save(validation_image)
        shape = {'id': 'box-1', 'type': 'rectangle', 'label': 'grip_done',
                 'x': 10, 'y': 12, 'width': 30, 'height': 20}
        assets = []
        for index, (path, split) in enumerate(((self.image, 'train'), (validation_image, 'val')), 1):
            assets.append({'id': f'asset-{index}', 'name': path.name, 'width': 80, 'height': 60,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'image_path': str(path), 'batch_id': f'source-batch-{index}',
                           'split': split, 'source': {}, 'revision': 1,
                           'review_state': 'approved', 'shapes': [shape]})
        snapshot = {'id': 'source', 'name': 'portable', 'revision': 1,
                    'classes': ['grip_done'], 'assets': assets}
        exported = export_project(snapshot, self.root / 'exports', 'yolo_detection')
        self.exported_folder = Path(exported['path'])
        self.exported_image = next((self.exported_folder / 'images' / 'train').glob('*.png'))
        self.store = ProjectStore(self.root / 'projects')
        self.pid = self.store.create_project('destination')['id']
        self.workflow = ReviewWorkflow(self.store)
        self.catalog = MaterialCatalog(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_selected_export_image_imports_its_annotation(self):
        preview = self.workflow.preview(self.pid, [str(self.exported_image)])
        self.assertEqual(preview['issues'], [])
        self.assertEqual(preview['items'][0]['shape_count'], 1)
        result = self.workflow.commit_import(self.pid, preview['token'], ['0'])
        self.assertEqual((result['added'], result['updated']), (1, 0))
        asset = self.store.get_asset(self.pid, result['asset_ids'][0])
        self.assertEqual(asset['shapes'][0]['label'], 'grip_done')
        self.assertEqual(asset['review_state'], 'pending')
        annotations = self.catalog.list_materials(self.pid, 'annotation')
        self.assertEqual({row['metadata']['role'] for row in annotations}, {'labels', 'class_mapping'})
        folder = self.store.directory(self.pid)
        for row in annotations:
            source = Path(row['metadata']['original_path'])
            self.assertTrue(row['relative_path'].startswith('inputs/annotations/'))
            self.assertEqual((folder / row['relative_path']).read_bytes(), source.read_bytes())
            self.assertEqual(row['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            links = row['metadata']['linked_images']
            self.assertEqual([(link['image_id'], link['annotation_revision']) for link in links], [(asset['id'], 1)])

    def test_existing_unlabeled_image_receives_annotation(self):
        blank = self.store.add_assets(self.pid, [{'path': self.image, 'shapes': []}])
        asset_id = blank['asset_ids'][0]
        preview = self.workflow.preview(self.pid, [str(self.exported_image)])
        self.assertEqual(preview['items'][0]['shape_count'], 1)
        result = self.workflow.commit_import(self.pid, preview['token'], ['0'])
        self.assertEqual((result['added'], result['updated'], result['duplicates']), (0, 1, 0))
        self.assertEqual(result['updated_asset_ids'], [asset_id])
        self.assertEqual(self.store.get_asset(self.pid, asset_id)['shapes'][0]['label'], 'grip_done')
        for annotation in self.catalog.list_materials(self.pid, 'annotation'):
            links = annotation['metadata']['linked_images']
            self.assertEqual([(link['image_id'], link['annotation_revision']) for link in links], [(asset_id, 2)])

    def test_modified_label_after_preview_rejects_batch_and_cleans_staged_annotations(self):
        preview = self.workflow.preview(self.pid, [str(self.exported_folder)])
        self.assertEqual(len(preview['items']), 2)
        label = next((self.exported_folder / 'labels' / 'val').glob('*.txt'))
        label.write_bytes(label.read_bytes() + b'\n')

        with self.assertRaisesRegex(ConflictError, '來源標註'):
            self.workflow.commit_import(self.pid, preview['token'], ['0', '1'])

        self.assertEqual(self.catalog.list_materials(self.pid), [])
        folder = self.store.directory(self.pid)
        self.assertEqual(list(project_area(folder, 'imported-annotations').rglob('*')), [])

    def test_shared_coco_source_links_images_and_preserves_trash_history(self):
        folder = self.root / 'coco'
        folder.mkdir()
        images = []
        for index in (1, 2):
            name = f'input-{index}.png'
            Image.new('RGB', (80, 60), (index, 80, 80)).save(folder / name)
            images.append({'id': index, 'file_name': name, 'width': 80, 'height': 60})
        annotation = folder / 'annotations.json'
        annotation.write_text(json.dumps({
            'images': images, 'categories': [{'id': 1, 'name': 'grip_done'}],
            'annotations': [{'id': index, 'image_id': index, 'category_id': 1, 'bbox': [5, 5, 18, 20]}
                            for index in (1, 2)],
        }), encoding='utf-8')
        preview = self.workflow.preview(self.pid, [str(annotation)])
        result = self.workflow.commit_import(self.pid, preview['token'], ['0', '1'])
        ids = set(result['asset_ids'])

        saved, = self.catalog.list_materials(self.pid, 'annotation')
        self.assertEqual({link['image_id'] for link in saved['metadata']['linked_images']}, ids)
        self.assertEqual((self.store.directory(self.pid) / saved['relative_path']).read_bytes(),
                         annotation.read_bytes())
        preview = self.workflow.preview(self.pid, [str(annotation)])
        repeated = self.workflow.commit_import(self.pid, preview['token'], ['0', '1'])
        self.assertEqual(repeated['duplicates'], 2)
        self.assertEqual(self.catalog.list_materials(self.pid, 'annotation'), [saved])

        image_id = result['asset_ids'][0]
        asset = self.store.get_asset(self.pid, image_id)
        self.store.trash_assets(self.pid, [image_id], {image_id: asset['revision']})
        self.assertEqual(self.catalog.list_materials(self.pid, 'annotation'), [saved])
        archived = next(item for item in self.catalog.list_materials(self.pid, 'image') if item['id'] == image_id)
        self.assertEqual(archived['state'], 'trashed')
        trash = self.store.list_trash(self.pid)[0]
        self.store.trash_assets(self.pid, [image_id], {image_id: trash['revision']}, restore=True)
        restored = self.store.get_asset(self.pid, image_id)
        self.store.delete_asset(self.pid, image_id, restored['revision'])
        retained, = self.catalog.list_materials(self.pid, 'annotation')
        self.assertEqual({link['image_id'] for link in retained['metadata']['linked_images']}, ids - {image_id})


if __name__ == '__main__':
    unittest.main()
