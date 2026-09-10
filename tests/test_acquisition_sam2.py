"""Opt-in real offline SAM2 test; never runs against a user's camera."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from workbench import acquisition as a
from sam2_segmentation.training_dataset import decode_coco_uncompressed_rle


@unittest.skipUnless(os.environ.get("WORKBENCH_TEST_SAM2_REAL") == "1", "Set WORKBENCH_TEST_SAM2_REAL=1 for installed offline model validation")
class RealSam2Tests(unittest.TestCase):
    def test_local_model_inference_preserves_prompted_hole(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.full((240, 320, 3), 238, np.uint8)
            cv2.rectangle(image, (50, 35), (270, 205), (32, 78, 145), -1)
            cv2.circle(image, (160, 120), 37, (238, 238, 238), -1)
            path = Path(directory) / "合成有孔工件.png"
            a._save_png(path, image)
            try:
                result = a.segment_image(path, "sam2", points=[[75, 120], [240, 120]],
                                         negative_points=[[160, 120], [10, 10]],
                                         box=[40, 25, 280, 215], label="工件")
                shape = result["shape"]
                mask = decode_coco_uncompressed_rle({"size": [240, 320], "counts": shape["counts"]})
                self.assertEqual(mask[120, 75], 255)
                self.assertEqual(mask[120, 240], 255)
                self.assertEqual(mask[120, 160], 0)
                self.assertEqual(mask[10, 10], 0)
                self.assertTrue(result["diagnostics"]["local_files_only"])
                self.assertEqual(shape["metadata"]["review_state"], "pending")
                print("REAL_SAM2_RESULT=" + json.dumps(result["diagnostics"], ensure_ascii=False))
                warmed = a.segment_image(path, "sam2", points=[[75, 120], [240, 120]],
                                         negative_points=[[160, 120], [10, 10]], label="工件")
                warm_mask = decode_coco_uncompressed_rle({"size": [240, 320], "counts": warmed["shape"]["counts"]})
                self.assertEqual(warm_mask[120, 75], 255)
                self.assertEqual(warm_mask[120, 160], 0)
                print("WARM_SAM2_RESULT=" + json.dumps(warmed["diagnostics"], ensure_ascii=False))
            finally:
                a.close_ai()


if __name__ == "__main__":
    unittest.main()
