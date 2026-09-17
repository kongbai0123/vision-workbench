import unittest

import numpy as np

from classical_segmentation import SegmentationPrompts
from composer_core.mask_cleanup import repair_tiny_holes
from sam2_segmentation.backend import Sam2Segmenter
from sam2_segmentation.models import Sam2Config, Sam2RawOutput


class MaskCleanupTests(unittest.TestCase):
    def test_fills_diagonal_pixel_gap_that_polygon_would_close(self):
        mask = np.zeros((32, 32), np.uint8)
        mask[2:30, 2:30] = 1
        mask[10, 10] = 0
        mask[11:, 11:] = 0
        repaired, report = repair_tiny_holes(mask, max_ratio=.1)
        self.assertEqual(report["holes_filled"], 1)
        self.assertEqual(report["pixels_filled"], 1)
        self.assertTrue(repaired[10, 10])
        self.assertFalse(repaired[11, 11])

    def test_preserves_explicit_background_prompt_and_large_holes(self):
        mask = np.zeros((64, 64), np.uint8)
        mask[2:62, 2:62] = 1
        mask[10, 10] = 0
        mask[20:26, 20:26] = 0
        repaired, report = repair_tiny_holes(
            mask,
            protected_background_points=[[10, 10]],
            max_ratio=.1,
        )
        self.assertFalse(repaired[10, 10])
        self.assertFalse(repaired[22, 22])
        self.assertEqual(report["holes_filled"], 0)
        self.assertEqual(
            {item["reason"] for item in report["retained"]},
            {"protected_background_prompt", "above_individual_limit"},
        )

    def test_standalone_sam2_returns_cleaned_mask_and_matching_diagnostics(self):
        candidate = np.zeros((128, 128), bool)
        candidate[2:126, 2:126] = True
        candidate[50, 50] = False

        class Runtime:
            is_loaded = True
            device_name = "cpu"
            dtype_name = "float32"
            model_load_time_ms = 0.0
            def __init__(self, config):
                pass
            def load(self):
                pass
            def infer(self, image, **kwargs):
                return Sam2RawOutput(candidate[None], np.array([.99]))
            def close(self):
                pass

        segmenter = Sam2Segmenter(
            Sam2Config(tiny_hole_max_ratio=.1), runtime_factory=Runtime,
        )
        result = segmenter.segment(
            np.zeros((128, 128, 3), np.uint8),
            SegmentationPrompts(foreground_points=((.2, .2),)),
        )
        self.assertTrue(result.mask[50, 50])
        self.assertEqual(result.diagnostics.holes_filled, 1)
        self.assertEqual(result.diagnostics.hole_pixels_filled, 1)
        self.assertFalse(Sam2Config().smart_boundary_smoothing)


if __name__ == "__main__":
    unittest.main()
