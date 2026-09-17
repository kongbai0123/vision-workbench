import unittest

import numpy as np

from workbench.evaluation_metrics import PixelMetrics, detection_summary, evaluation_protocol


class PixelMetricTests(unittest.TestCase):
    def test_empty_images_do_not_reward_a_completely_missed_positive(self):
        metrics = PixelMetrics(["part"])
        positive = np.ones((2, 2), dtype=bool)
        metrics.update("part", np.zeros_like(positive), positive)
        for _ in range(8):
            empty = np.zeros((2, 2), dtype=bool)
            metrics.update("part", empty, empty)
        result = metrics.summary("val", 9)
        self.assertEqual(result["mean_iou"], 0.0)
        self.assertEqual(result["micro_iou"], 0.0)
        self.assertEqual(result["per_class"]["part"]["fn"], 4)

    def test_undefined_class_is_na_but_false_positive_class_is_zero(self):
        metrics = PixelMetrics(["unused", "false-positive"])
        empty = np.zeros((2, 2), dtype=bool)
        metrics.update("unused", empty, empty)
        predicted = empty.copy(); predicted[0, 0] = True
        metrics.update("false-positive", predicted, empty)
        result = metrics.summary("test", 1)
        self.assertIsNone(result["per_class_iou"]["unused"])
        self.assertEqual(result["per_class_iou"]["false-positive"], 0.0)
        self.assertEqual(result["mean_iou"], 0.0)


class DetectionMetricTests(unittest.TestCase):
    def test_duplicate_and_background_boxes_are_false_positives(self):
        samples = [{"expected": {"part": [[0, 0, 10, 10]]},
                    "predicted": {"part": [(0.9, [0, 0, 10, 10]),
                                             (0.8, [0, 0, 10, 10]),
                                             (0.7, [20, 20, 30, 30])]}}]
        result = detection_summary(["part"], samples, "test", 1)
        self.assertEqual(result["totals"], {"tp": 1, "fp": 2, "fn": 0})
        self.assertEqual(result["precision_50"], 0.333333)
        self.assertEqual(result["recall_50"], 1.0)

    def test_missing_detection_is_false_negative(self):
        samples = [{"expected": {"part": [[0, 0, 10, 10]]}, "predicted": {"part": []}}]
        result = detection_summary(["part"], samples, "val", 1)
        self.assertEqual(result["per_class"]["part"]["fn"], 1)
        self.assertEqual(result["recall_50"], 0.0)
        self.assertEqual(result["box_map50"], 0.0)


class EvaluationProtocolTests(unittest.TestCase):
    def test_source_overlap_is_recorded_as_non_independent(self):
        manifest = {"assets": [
            {"split": "train", "batch_id": "session-a"},
            {"split": "val", "batch_id": "session-b"},
            {"split": "test", "batch_id": "session-a"},
        ]}
        protocol = evaluation_protocol(checkpoint="best_validation", has_test=True, manifest=manifest)
        self.assertFalse(protocol["test_independent_sources"])
        self.assertEqual(protocol["test_source_overlap_groups"], ["session-a"])
        self.assertTrue(protocol["test_source_tracking_complete"])

    def test_untracked_test_sources_are_unknown_not_independent(self):
        manifest = {"assets": [{"split": "train"}, {"split": "val"}, {"split": "test"}]}
        protocol = evaluation_protocol(checkpoint="final_epoch", has_test=True, manifest=manifest)
        self.assertIsNone(protocol["test_independent_sources"])
        self.assertFalse(protocol["test_source_tracking_complete"])


if __name__ == "__main__":
    unittest.main()
