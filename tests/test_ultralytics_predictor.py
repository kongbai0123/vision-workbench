from types import SimpleNamespace
import unittest

import numpy as np

from composer_core.geometry import decode_rle
from workbench.ultralytics_engine import Predictor, ULTRALYTICS_ENGINES


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


class UltralyticsPredictorTests(unittest.TestCase):
    def _predictor(self, masks=None, labels=(0,), scores=(.95,), *, kind="segment", threshold=.5):
        predictor = Predictor.__new__(Predictor)
        engine = "yolo26n_seg" if kind == "segment" else "rt_detr_r50"
        predictor.definition = ULTRALYTICS_ENGINES[engine]
        predictor.record = {"engine": engine, "image_size": 960, "model_version_id": "M007",
                            "classes": ["part", "second"], "score_threshold": threshold}
        predictor.device = "cpu"
        result = SimpleNamespace(boxes=SimpleNamespace(cls=_Array(labels), conf=_Array(scores),
                                 xyxy=_Array([[2, 1, 6, 4] for _ in labels])),
                                 masks=None if masks is None else SimpleNamespace(data=_Array(masks)))
        calls = []

        def predict(**options):
            calls.append(options)
            return [result]

        predictor.model = SimpleNamespace(predict=predict)
        return predictor, calls

    def test_rgb_is_converted_to_bgr_without_changing_the_original_image(self):
        rgb = np.zeros((6, 10, 3), np.uint8)
        rgb[:, :5] = (240, 20, 10)
        rgb[:, 5:] = (5, 40, 220)
        original = rgb.copy()
        mask = np.zeros((6, 10), np.uint8)
        mask[1:3, 6:9] = 1
        predictor, calls = self._predictor([mask])
        shape = predictor.predict(rgb, 10, 6)[0]
        options = calls[0]
        np.testing.assert_array_equal(options["source"], original[..., ::-1])
        np.testing.assert_array_equal(rgb, original)
        self.assertTrue(options["source"].flags.c_contiguous)
        self.assertTrue(options["retina_masks"])
        self.assertEqual(options["imgsz"], 960)
        self.assertEqual(options["conf"], .5)
        self.assertEqual(options["device"], "cpu")
        np.testing.assert_array_equal(decode_rle(shape["counts"], 10, 6) > 0, mask > 0)
        self.assertEqual(shape["metadata"]["bounds"], [6, 1, 9, 3])
        self.assertEqual(shape["metadata"]["model_version_id"], "M007")
        self.assertEqual(shape["metadata"]["confidence"], .95)
        self.assertEqual(shape["label"], "part")

    def test_padded_masks_are_rejected_instead_of_stretched_over_the_image(self):
        padded = np.zeros((8, 10), np.uint8)
        padded[2:4, 6:9] = 1
        predictor, _ = self._predictor([padded])
        with self.assertRaisesRegex(ValueError, "原圖尺寸.*遮罩"):
            predictor.predict(np.zeros((6, 10, 3), np.uint8), 10, 6)

    def test_wrong_image_size_or_channel_count_fails_before_inference(self):
        predictor, calls = self._predictor()
        for pixels in (np.zeros((6, 10), np.uint8), np.zeros((6, 10, 4), np.uint8),
                       np.zeros((10, 6, 3), np.uint8)):
            with self.subTest(shape=pixels.shape), self.assertRaisesRegex(ValueError, "RGB"):
                predictor.predict(pixels, 10, 6)
        self.assertFalse(calls)

    def test_configured_threshold_and_valid_class_filter_are_preserved(self):
        masks = np.ones((4, 6, 10), np.uint8)
        predictor, calls = self._predictor(masks, labels=(0, 1, 8, 0), scores=(.7, .81, .99, .2), threshold=.8)
        shapes = predictor.predict(np.zeros((6, 10, 3), np.uint8), 10, 6)
        self.assertEqual(calls[0]["conf"], .8)
        self.assertEqual([shape["label"] for shape in shapes], ["second"])
        self.assertEqual(shapes[0]["metadata"]["confidence"], .81)

    def test_detection_keeps_original_box_coordinates_and_disables_native_masks(self):
        predictor, calls = self._predictor(kind="detect")
        shapes = predictor.predict(np.zeros((6, 10, 3), np.uint8), 10, 6)
        self.assertFalse(calls[0]["retina_masks"])
        self.assertEqual({key: shapes[0][key] for key in ("type", "x", "y", "width", "height")},
                         {"type": "rectangle", "x": 2., "y": 1., "width": 4., "height": 3.})


if __name__ == "__main__":
    unittest.main()
