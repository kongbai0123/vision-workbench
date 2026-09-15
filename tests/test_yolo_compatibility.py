import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from composer_core.geometry import encode_rle
from workbench.training_engine import atomic_json
from workbench.ultralytics_engine import prepare_yolo_dataset
from workbench.yolo_compatibility import analyze_manifest


class YoloCompatibilityTests(unittest.TestCase):
    def manifest(self, root, mask):
        (root / "images").mkdir()
        image = root / "images/A001.png"
        Image.new("RGB", (mask.shape[1], mask.shape[0]), "white").save(image)
        shape = {"id": "s1", "type": "mask", "label": "part", "x": 0, "y": 0,
                 "width": mask.shape[1], "height": mask.shape[0], "counts": encode_rle(mask)}
        asset = {"asset_id": "A001", "name": "tiny-holes.png", "width": mask.shape[1],
                 "height": mask.shape[0], "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                 "image_file": "images/A001.png", "split": "train", "shapes": [shape],
                 "objects": [{"shape_id": "s1", "bbox_xywh": [2, 2, mask.shape[1]-4, mask.shape[0]-4]}]}
        value = {"dataset_version_id": "D001", "classes": ["part"], "assets": [asset]}
        path = root / "manifest.json"; atomic_json(path, value)
        return path, value

    def test_tiny_holes_are_repaired_only_in_run_copy_with_audit_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((256, 256), np.uint8); mask[2:254, 2:254] = 1
            mask[100, 120] = 0; mask[130, 140:143] = 0
            manifest_path, manifest = self.manifest(root, mask)
            original = manifest_path.read_bytes()
            report = analyze_manifest(manifest, {})
            self.assertTrue(report["compatible"])
            self.assertEqual(report["summary"]["holes_repaired"], 2)
            self.assertEqual(report["summary"]["pixels_repaired"], 4)
            self.assertEqual([hole["bbox"] for hole in report["repairs"][0]["holes"]],
                             [[120, 100, 1, 1], [140, 130, 3, 1]])
            prepare_yolo_dataset(manifest_path, root / "run-copy", "instance_segmentation", {})
            self.assertEqual(manifest_path.read_bytes(), original)
            persisted = json.loads((root / "run-copy/yolo-compatibility.json").read_text(encoding="utf-8"))
            self.assertTrue(persisted["source_annotations_unchanged"])
            self.assertEqual(persisted["metric_label_space"], "yolo_compatible_copy")
            self.assertTrue((root / "run-copy/labels/train/A001.txt").read_text().strip().startswith("0 "))

    def test_strict_mode_blocks_even_one_pixel_hole(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mask = np.zeros((64, 64), np.uint8); mask[2:62, 2:62] = 1; mask[20, 20] = 0
            _path, manifest = self.manifest(root, mask)
            report = analyze_manifest(manifest, {"yolo_mask_policy": "strict"})
            self.assertFalse(report["compatible"])
            self.assertEqual(report["blockers"][0]["code"], "holes_strict")

    def test_default_policy_repairs_human_invisible_holes_up_to_sixteen_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((512, 512), np.uint8); mask[2:510, 2:510] = 1
            mask[100:105, 120] = 0; mask[130:133, 140:143] = 0
            _path, manifest = self.manifest(root, mask)
            report = analyze_manifest(manifest, {})
            self.assertTrue(report["compatible"])
            self.assertEqual(report["summary"]["holes_repaired"], 2)
            self.assertEqual(report["summary"]["pixels_repaired"], 14)

    def test_large_hole_and_separate_components_stay_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mask = np.zeros((64, 64), np.uint8); mask[2:62, 2:62] = 1; mask[20:26, 20:26] = 0
            _path, manifest = self.manifest(root, mask)
            report = analyze_manifest(manifest, {"tiny_hole_max_ratio": .1})
            self.assertFalse(report["compatible"])
            self.assertEqual(report["blockers"][0]["code"], "holes_above_limit")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mask = np.zeros((64, 64), np.uint8); mask[2:20, 2:20] = 1; mask[35:55, 35:55] = 1
            _path, manifest = self.manifest(root, mask)
            report = analyze_manifest(manifest, {})
            self.assertFalse(report["compatible"])
            self.assertEqual(report["blockers"][0]["code"], "multiple_components")

    def test_diagonal_one_pixel_gap_is_repaired_before_worker_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((512, 512), np.uint8)
            mask[2:510, 2:510] = 1
            mask[100:, 100:] = 0
            mask[99, 99] = 0
            manifest_path, manifest = self.manifest(root, mask)
            report = analyze_manifest(manifest, {})
            self.assertTrue(report["compatible"])
            self.assertEqual(report["summary"]["holes_repaired"], 1)
            self.assertEqual(report["summary"]["pixels_repaired"], 1)
            prepare_yolo_dataset(
                manifest_path, root / "run-copy", "instance_segmentation", {},
            )
            self.assertTrue((root / "run-copy/labels/train/A001.txt").read_text().strip())


if __name__ == "__main__":
    unittest.main()
