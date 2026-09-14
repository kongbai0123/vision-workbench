from __future__ import annotations

import hashlib
import unittest

from workbench.splitting import stratified_split


class StratifiedSplitTests(unittest.TestCase):
    @staticmethod
    def asset(index, label, sha=None):
        return {
            "id": f"asset-{index:03d}",
            "sha256": sha or hashlib.sha256(f"image-{index}".encode()).hexdigest(),
            "shapes": [{"id": f"shape-{index}", "label": label}],
        }

    def test_class_instance_counts_follow_requested_ratio_deterministically(self):
        assets = [self.asset(index, "A" if index < 25 else "B") for index in range(50)]
        first, report = stratified_split(assets, {"train": 70, "val": 20, "test": 10})
        second, _ = stratified_split(reversed(assets), {"train": 70, "val": 20, "test": 10})
        self.assertEqual(first, second)
        self.assertEqual(report["image_counts"], {"train": 35, "val": 10, "test": 5})
        for label in ("A", "B"):
            targets = {"train": 17.5, "val": 5, "test": 2.5}
            for split, target in targets.items():
                self.assertLessEqual(abs(report["class_counts"][split][label] - target), 0.5)

    def test_duplicate_images_never_cross_splits(self):
        duplicate = "a" * 64
        assets = [self.asset(0, "A", duplicate), self.asset(1, "A", duplicate)]
        assets += [self.asset(index, "A") for index in range(2, 12)]
        assigned, _ = stratified_split(assets)
        self.assertEqual(assigned["asset-000"], assigned["asset-001"])

    def test_too_few_unique_images_is_rejected(self):
        assets = [self.asset(0, "A", "a" * 64), self.asset(1, "A", "a" * 64), self.asset(2, "A", "b" * 64)]
        with self.assertRaisesRegex(ValueError, "唯一影像不足"):
            stratified_split(assets)


if __name__ == "__main__":
    unittest.main()
