"""Scan-phase progress and once-per-dataset parsing for multi-file drops."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from composer_core.geometry import encode_rle
from workbench import pipeline
from workbench.pipeline import import_sources


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', image)[1].tofile(str(path))


class ImportScanTests(unittest.TestCase):
    COUNT = 6

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.session = self.root / 'session'
        mask = np.zeros((20, 24), np.uint8)
        mask[2:17, 2:21] = 255
        samples, images, annotations = [], [], []
        for index in range(self.COUNT):
            name = f'sample_{index:03d}.png'
            frame = np.full((20, 24, 3), 40 + index, np.uint8)
            write_image(self.session / 'normal_output' / name, frame)
            write_image(self.session / 'algorithm_output' / name, frame)
            write_image(self.session / 'label' / 'masks' / name, mask)
            samples.append({'id': f'sample-{index}', 'image': f'normal_output/{name}',
                            'mask': f'label/masks/{name}', 'qa_overlay': f'algorithm_output/{name}',
                            'review_status': 'pending'})
            images.append({'id': index + 1, 'file_name': f'normal_output/{name}', 'width': 24, 'height': 20})
            annotations.append({'id': index + 1, 'image_id': index + 1, 'category_id': 1,
                                'segmentation': {'size': [20, 24], 'counts': encode_rle(mask)}})
        write_json(self.session / 'manifest.json', {
            'dataset_type': 'sam2_training_pseudo_labels', 'session_id': 'capture',
            'review_status': 'pending', 'samples': samples})
        write_json(self.session / 'label' / 'instances.json', {
            'images': images, 'annotations': annotations, 'categories': [{'id': 1, 'name': 'workpiece'}]})
        self.images = sorted((self.session / 'normal_output').glob('*.png'))

    def tearDown(self):
        self.temp.cleanup()

    def test_selecting_many_session_files_parses_the_session_once(self):
        with patch.object(pipeline, '_coco_import', wraps=pipeline._coco_import) as coco, \
                patch.object(pipeline, '_dimensions', wraps=pipeline._dimensions) as dimensions:
            result = import_sources([str(path) for path in self.images])
        self.assertEqual(result['issues'], [])
        self.assertEqual(sorted(Path(r['path']) for r in result['records']), self.images)
        self.assertEqual(coco.call_count, 1)
        # Two dimension probes per session image, not per image for every selected file.
        self.assertEqual(dimensions.call_count, 2 * self.COUNT)

    def test_overlay_and_mask_selections_match_the_folder_import(self):
        folder = {r['path']: r for r in import_sources([self.session])['records']}
        name = self.images[0].name
        result = import_sources([self.session / 'algorithm_output' / name,
                                 self.session / 'label' / 'masks' / name, self.images[1]])
        self.assertEqual([Path(r['path']).name for r in result['records']], [name, self.images[1].name])
        self.assertEqual([issue['level'] for issue in result['issues']], ['warning'])
        for record in result['records']:
            self.assertEqual(record['shapes'], folder[record['path']]['shapes'])
            self.assertEqual((record['batch_id'], record['review_state']), ('capture', 'pending'))

    def test_broken_session_reports_every_selection_but_parses_once(self):
        (self.session / 'label' / 'instances.json').write_text('{"images": [', encoding='utf-8')
        with patch.object(pipeline, '_working_records', wraps=pipeline._working_records) as parse:
            result = import_sources([str(path) for path in self.images])
        self.assertEqual(parse.call_count, 1)
        self.assertEqual(result['records'], [])
        self.assertEqual(len(result['issues']), self.COUNT)
        self.assertEqual({issue['level'] for issue in result['issues']}, {'error'})
        self.assertEqual(len({issue['message'] for issue in result['issues']}), 1)

    def test_scan_progress_is_monotonic_and_advances_per_session_image(self):
        updates = []
        import_sources([str(path) for path in self.images],
                       progress=lambda message, percent: updates.append((message, percent)))
        percents = [percent for _message, percent in updates]
        self.assertEqual(percents, sorted(percents))
        self.assertEqual((percents[0], percents[-1]), (0, 100))
        # All selections share one session unit, so each parsed image moves the bar.
        self.assertGreaterEqual(len(set(percents)), self.COUNT + 1)
        self.assertTrue(all(message.strip() for message, _percent in updates))
        self.assertTrue(any('解析 SAM2 工作資料' in message for message, _percent in updates))

    def test_mixed_drop_progress_covers_every_unit(self):
        plain = self.root / 'plain'
        for index in range(3):
            write_image(plain / f'{index}.png', np.full((10, 12, 3), index, np.uint8))
        percents = []
        result = import_sources([plain, self.images[0], self.root / 'missing.png'],
                                progress=lambda message, percent: percents.append(percent))
        self.assertEqual(len(result['records']), 4)
        self.assertEqual([issue['level'] for issue in result['issues']], ['error'])
        self.assertEqual(percents, sorted(percents))
        self.assertEqual(percents[-1], 100)
        self.assertIn(round(100 / 3, 2), percents)

    def test_cancellation_raised_by_progress_is_not_reported_as_an_issue(self):
        class Cancelled(Exception):
            pass

        def cancel(message, percent):
            if percent > 0:
                raise Cancelled()

        with self.assertRaises(Cancelled):
            import_sources([str(path) for path in self.images], progress=cancel)

    def test_multiple_yolo_labels_scan_the_dataset_once(self):
        dataset = self.root / 'yolo'
        for index in range(4):
            write_image(dataset / 'images' / 'train' / f'{index}.png', np.zeros((10, 12, 3), np.uint8))
            label = dataset / 'labels' / 'train' / f'{index}.txt'
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text('0 0.5 0.5 0.5 0.5\n', encoding='utf-8')
        (dataset / 'data.yaml').write_text('names: [part]\n', encoding='utf-8')
        labels = sorted((dataset / 'labels' / 'train').glob('*.txt'))
        with patch.object(pipeline, '_folder_import', wraps=pipeline._folder_import) as scan:
            result = import_sources(labels[:3])
        self.assertEqual(result['issues'], [])
        self.assertEqual(sorted(Path(r['path']).stem for r in result['records']), ['0', '1', '2'])
        self.assertEqual(scan.call_count, 1)


if __name__ == '__main__':
    unittest.main()
