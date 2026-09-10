from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

from composer_core.geometry import decode_rle, encode_rle
from workbench.pipeline import import_sources, validate_project, export_project


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / 'source'
        self.images.mkdir()
        for index in range(3):
            image = np.zeros((20, 24, 3), np.uint8)
            image[:, :, index] = 90 + index
            self.write_image(self.images / f'{index}.png', image)
        mask = np.zeros((20, 24), np.uint8)
        mask[2:17, 2:21] = 255
        mask[7:12, 8:15] = 0
        self.mask = mask
        self.mask_shape = {'id': 'mask-stable', 'label': 'workpiece', 'type': 'mask', 'hidden': False,
                           'x': 0, 'y': 0, 'width': 24, 'height': 20, 'counts': encode_rle(mask),
                           'metadata': {'operator_note': 'hole must remain'}}

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def write_image(self, path, image):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode('.png', image)[1].tofile(str(path))

    def snapshot(self, count=2, mask=True):
        assets = []
        for index in range(count):
            path = self.images / f'{index}.png'
            shape = deepcopy(self.mask_shape) if mask else {'id': 'rect-stable', 'label': 'workpiece',
                    'type': 'rectangle', 'hidden': False, 'x': 2.5, 'y': 3.5, 'width': 10, 'height': 9}
            assets.append({'id': f'asset-{index}', 'name': path.name, 'width': 24, 'height': 20,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'image_path': str(path),
                           'batch_id': f'batch-{index}', 'split': '', 'source': {'camera': 0},
                           'revision': 3, 'review_state': 'approved', 'shapes': [shape]})
        return {'id': 'project', 'name': '中文 Project', 'revision': 8, 'classes': ['workpiece'], 'assets': assets}

    def test_native_roundtrip_preserves_holes_ids_metadata_bytes(self):
        snapshot = self.snapshot()
        before = {p.name: p.read_bytes() for p in self.images.glob('*.png')}
        result = export_project(snapshot, self.root / 'out')
        self.assertTrue(Path(result['zip_path']).is_file())
        with zipfile.ZipFile(result['zip_path']) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn('project.json', archive.namelist())
        imported = import_sources([result['path']])
        self.assertEqual(imported['issues'], [])
        self.assertEqual(len(imported['records']), 2)
        for record in imported['records']:
            self.assertEqual(record['shapes'][0], self.mask_shape)
            np.testing.assert_array_equal(decode_rle(record['shapes'][0]['counts'], 24, 20), self.mask)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.images.glob('*.png')})
        with self.assertRaises(FileExistsError):
            export_project(snapshot, self.root / 'out')

    def test_coco_roundtrip_precise_mask(self):
        result = export_project(self.snapshot(), self.root / 'out', 'coco')
        imported = import_sources([result['path']])
        self.assertEqual(imported['issues'], [])
        self.assertEqual(len(imported['records']), 2)
        self.assertEqual(imported['records'][0]['shapes'][0]['id'], 'mask-stable')
        np.testing.assert_array_equal(decode_rle(imported['records'][0]['shapes'][0]['counts'], 24, 20), self.mask)

    def test_pending_rejected_excluded_and_edited_snapshot_blocks_review(self):
        snapshot = self.snapshot(3)
        snapshot['assets'][1]['review_state'] = 'pending'
        snapshot['assets'][2]['review_state'] = 'rejected'
        result = export_project(snapshot, self.root / 'out')
        self.assertEqual(result['report']['image_count'], 1)
        self.assertEqual(result['report']['validation']['stats']['pending'], 1)
        self.assertEqual(result['report']['validation']['stats']['rejected'], 1)
        snapshot['assets'][0]['review_state'] = 'pending'
        self.assertFalse(validate_project(snapshot)['valid'])

    def test_yolo_loss_requires_ack_and_report_and_reimport(self):
        snapshot = self.snapshot()
        checked = validate_project(snapshot, 'yolo_segmentation')
        self.assertTrue(checked['valid'])
        self.assertEqual(checked['losses'][0]['holes_omitted'], 1)
        with self.assertRaisesRegex(ValueError, 'acknowledge_loss'):
            export_project(snapshot, self.root / 'out', 'yolo_segmentation')
        result = export_project(snapshot, self.root / 'out', 'yolo_segmentation', acknowledge_loss=True)
        imported = import_sources([result['path']])
        self.assertEqual(imported['issues'], [])
        self.assertEqual(len(imported['records']), 2)
        self.assertEqual({r['split'] for r in imported['records']}, {'train', 'val'})
        self.assertEqual({r['batch_id'] for r in imported['records']}, {'batch-0', 'batch-1'})
        self.assertEqual({r['shapes'][0]['id'] for r in imported['records']}, {'mask-stable'})

    def test_detection_bbox_recomputed_from_mask(self):
        snapshot = self.snapshot()
        snapshot['assets'][0]['shapes'][0].update(x=0, y=0, width=1, height=1)
        result = export_project(snapshot, self.root / 'out', 'yolo_detection', acknowledge_loss=True)
        rows = [p.read_text().split() for p in Path(result['path']).rglob('labels/*/*.txt')]
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0][3]), 19/24, places=8)
        self.assertAlmostEqual(float(rows[0][4]), 15/20, places=8)

    def test_split_mixed_same_batch_and_duplicate_content_are_rejected(self):
        snapshot = self.snapshot()
        snapshot['assets'][0]['split'] = 'train'
        self.assertFalse(validate_project(snapshot)['valid'])
        snapshot['assets'][1]['split'] = 'val'
        snapshot['assets'][1]['batch_id'] = 'batch-0'
        self.assertFalse(validate_project(snapshot)['valid'])
        snapshot['assets'][1]['batch_id'] = 'batch-1'
        snapshot['assets'][1]['image_path'] = snapshot['assets'][0]['image_path']
        snapshot['assets'][1]['sha256'] = snapshot['assets'][0]['sha256']
        self.assertFalse(validate_project(snapshot)['valid'])

    def test_single_batch_yolo_blocks_but_native_works(self):
        snapshot = self.snapshot()
        snapshot['assets'][1]['batch_id'] = 'batch-0'
        self.assertFalse(validate_project(snapshot, 'yolo_detection')['valid'])
        self.assertTrue(validate_project(snapshot)['valid'])

    def test_invalid_hash_dimensions_class_shape_block(self):
        for mutate in (lambda a: a.update(sha256='0'*64), lambda a: a.update(width=99),
                       lambda a: a['shapes'][0].update(label='missing'),
                       lambda a: a['shapes'][0].update(counts=[1, 2])):
            snapshot = self.snapshot()
            mutate(snapshot['assets'][0])
            self.assertFalse(validate_project(snapshot)['valid'])

    def test_labelme_all_vector_shapes_roundtrip(self):
        snapshot = self.snapshot(1, mask=False)
        snapshot['assets'][0]['shapes'] += [
            {'id': 'point', 'label': 'workpiece', 'type': 'point', 'points': [[5,5]]},
            {'id': 'line', 'label': 'workpiece', 'type': 'linestrip', 'points': [[1,1],[8,8]]},
            {'id': 'poly', 'label': 'workpiece', 'type': 'polygon', 'points': [[2,2],[10,2],[5,10]]}]
        result = export_project(snapshot, self.root / 'out', 'labelme')
        imported = import_sources([result['path']])
        self.assertEqual(imported['issues'], [])
        self.assertEqual([s['id'] for s in imported['records'][0]['shapes']], ['rect-stable', 'point', 'line', 'poly'])
        self.assertFalse(validate_project(snapshot, 'coco')['valid'])

    def test_raw_import_copies_nothing_and_unknown_labelme_shape_errors(self):
        result = import_sources([self.images])
        self.assertEqual(len(result['records']), 3)
        self.assertTrue(all(r['review_state'] == 'pending' for r in result['records']))
        self.write_json(self.images / '0.json', {'imagePath': '0.png', 'shapes': [{'shape_type': 'circle', 'label': 'x', 'points': [[1,1],[2,2]]}]})
        result = import_sources([self.images])
        self.assertEqual(result['records'], [])
        self.assertEqual(result['issues'][0]['level'], 'error')

    def test_protected_working_folder_and_individual_image_rejected(self):
        self.write_json(self.images / 'manifest.json', {'dataset_type': 'sam2_training_pseudo_labels'})
        for path in (self.images, self.images/'0.png', self.images.parent):
            result = import_sources([path])
            self.assertEqual(result['records'], [])
            self.assertEqual(result['issues'][0]['level'], 'error')

    def test_graph_verified_hash_and_no_auxiliary_scan(self):
        package = self.root / 'verified'
        ready = package / 'training_ready'
        ready.mkdir(parents=True)
        image = ready/'frame.png'
        image.write_bytes((self.images/'0.png').read_bytes())
        annotation = {'images': [{'id': 1, 'file_name': 'frame.png', 'width': 24, 'height': 20, 'session_id': 'session-1'}],
                      'categories': [{'id': 1, 'name': 'workpiece'}],
                      'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'segmentation': {'size': [20,24], 'counts': encode_rle(self.mask)}}]}
        self.write_json(ready/'annotations.json', annotation)
        manifest = {'dataset_type': 'sam2_human_verified_flat_training', 'sample_count': 1, 'image_count': 1, 'annotation_count': 1,
                    'annotations_sha256': hashlib.sha256((ready/'annotations.json').read_bytes()).hexdigest(),
                    'samples': [{'id': 'sample', 'image': 'frame.png', 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'review_status': 'human_verified'}]}
        self.write_json(ready/'manifest.json', manifest)
        self.write_json(package/'export_manifest.json', {'dataset_type': 'sam2_human_verified_immutable_export', 'training_ready': {'path': 'training_ready'}})
        self.write_image(package/'masks'/'mask.png', self.mask)
        result = import_sources([package])
        self.assertEqual(result['issues'], [])
        self.assertEqual(len(result['records']), 1)
        self.assertEqual(result['records'][0]['review_state'], 'approved')
        self.assertEqual(result['records'][0]['batch_id'], 'session-1')
        self.assertEqual(import_sources([package/'masks'/'mask.png'])['records'], [])
        self.assertEqual(import_sources([package/'masks'])['records'], [])
        manifest['annotations_sha256'] = '0'*64
        self.write_json(ready/'manifest.json', manifest)
        for path in (package, ready/'annotations.json', image):
            result = import_sources([path])
            self.assertEqual(result['records'], [])

    def test_native_manifest_tampering_rejected(self):
        exported = export_project(self.snapshot(), self.root / 'out')
        project = Path(exported['path'])/'project.json'
        project.write_text(project.read_text(encoding='utf-8')+' ', encoding='utf-8')
        self.assertEqual(import_sources([exported['path']])['records'], [])

    def test_failed_archive_keeps_no_published_version_or_partial(self):
        with patch('workbench.pipeline.zipfile.ZipFile', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                export_project(self.snapshot(), self.root/'out')
        self.assertEqual(list((self.root/'out').iterdir()), [])

    def test_compressed_coco_rle_decoder(self):
        def compress(counts):
            output = []
            for i, raw in enumerate(counts):
                value = raw - counts[i-2] if i > 2 else raw
                while True:
                    code = value & 31
                    value >>= 5
                    more = value != -1 if code & 16 else value != 0
                    if more:
                        code |= 32
                    output.append(chr(code+48))
                    if not more:
                        break
            return ''.join(output)
        encoded = compress(encode_rle(self.mask))
        np.testing.assert_array_equal(decode_rle(encoded, 24, 20), self.mask)

    def test_runtime_has_no_imports_from_original_projects(self):
        import inspect
        import workbench.pipeline as pipeline
        source = inspect.getsource(pipeline)
        self.assertNotIn('sys.path', source)
        self.assertNotIn('subprocess', source)
        self.assertNotIn('C:\\workspace\\program\\test', source)

    def test_jsonl_preserves_precise_shapes_and_pairs(self):
        snapshot = self.snapshot()
        result = export_project(snapshot, self.root/'out', 'jsonl')
        imported = import_sources([result['path']])
        self.assertEqual(imported['issues'], [])
        self.assertEqual(len(imported['records']), 2)
        self.assertEqual(imported['records'][0]['shapes'][0], self.mask_shape)
        lines = Path(result['path'])/'dataset.jsonl'
        self.assertEqual(len(import_sources([lines])['records']), 2)

    def test_classification_is_single_label_and_keeps_shape_sidecar(self):
        snapshot = self.snapshot()
        result = export_project(snapshot, self.root/'out', 'classification', acknowledge_loss=True)
        imported = import_sources([result['path']])
        self.assertEqual(imported['issues'], [])
        self.assertEqual(len(imported['records']), 2)
        self.assertEqual(imported['records'][0]['shapes'][0], self.mask_shape)
        self.assertEqual(len(list(Path(result['path']).glob('images/0000_workpiece/*.png'))), 2)
        snapshot['classes'].append('other')
        shape = deepcopy(self.mask_shape)
        shape.update(id='second', label='other')
        snapshot['assets'][0]['shapes'].append(shape)
        self.assertFalse(validate_project(snapshot, 'classification')['valid'])

    def test_missing_yolo_pair_rejected(self):
        (self.images/'classes.txt').write_text('workpiece\n', encoding='utf-8')
        (self.images/'0.txt').write_text('0 0.5 0.5 0.5 0.5\n', encoding='utf-8')
        result = import_sources([self.images])
        self.assertEqual(result['records'], [])
        self.assertEqual(result['issues'][0]['level'], 'error')

    def test_split_preserved_in_labelme_roundtrip(self):
        snapshot = self.snapshot(mask=False)
        snapshot['assets'][0]['split'] = 'train'
        snapshot['assets'][1]['split'] = 'val'
        result = export_project(snapshot, self.root/'out', 'labelme')
        imported = import_sources([result['path']])
        self.assertEqual(imported['issues'], [])
        self.assertEqual({r['batch_id'] for r in imported['records']}, {'batch-0', 'batch-1'})
        self.assertEqual({r['split'] for r in imported['records']}, {'train', 'val'})

    def test_exif6_blocks_external_formats_and_native_preserves_raw_coordinates(self):
        from PIL import Image
        jpeg = self.images/'rotated-metadata.jpg'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (24,20), (25,75,125)).save(jpeg, exif=exif)
        original = jpeg.read_bytes()
        imported = import_sources([jpeg])
        self.assertEqual(imported['issues'], [])
        self.assertEqual((imported['records'][0]['width'], imported['records'][0]['height']), (24,20))
        self.assertEqual(imported['records'][0]['source']['pixel_coordinates']['exif_orientation'], 6)
        snapshot = self.snapshot(1, mask=False)
        snapshot['assets'][0].update(image_path=str(jpeg), name=jpeg.name,
                                    sha256=hashlib.sha256(original).hexdigest())
        for kind in ('coco', 'yolo_detection', 'yolo_segmentation', 'labelme', 'classification'):
            result = validate_project(snapshot, kind)
            self.assertFalse(result['valid'])
            self.assertTrue(any('EXIF Orientation=6' in message for message in result['errors']))
            with self.assertRaisesRegex(ValueError, 'EXIF Orientation=6'):
                export_project(snapshot, self.root/'out', kind, acknowledge_loss=True)
        for kind in ('native', 'jsonl'):
            result = validate_project(snapshot, kind)
            self.assertTrue(result['valid'])
            self.assertTrue(any('EXIF Orientation=6' in message for message in result['warnings']))
            output = export_project(snapshot, self.root/'out', kind)
            restored = import_sources([output['path']])
            self.assertEqual(restored['issues'], [])
            asset = restored['records'][0]
            self.assertEqual(Path(asset['path']).read_bytes(), original)
            self.assertEqual(asset['shapes'][0], snapshot['assets'][0]['shapes'][0])
            self.assertEqual(asset['source']['pixel_coordinates'], {
                'convention': 'raw_pixel_matrix', 'exif_orientation': 6, 'exif_orientation_applied': False})
            self.assertEqual(output['report']['items'][0]['pixel_coordinates']['exif_orientation'], 6)
        self.assertEqual(jpeg.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
