import unittest
from unittest.mock import patch

from workbench.augmentation import augment_dense_target, normalize_augmentation, yolo_augmentation_args


class AugmentationProfileTests(unittest.TestCase):
    def test_presets_are_train_only_and_explicit(self):
        profile = normalize_augmentation({"preset": "standard"})
        self.assertEqual(profile["apply_to"], "train")
        self.assertEqual(profile["mode"], "online")
        self.assertEqual(profile["mosaic"], 1.0)
        self.assertEqual(yolo_augmentation_args(profile)["close_mosaic"], 10)

    def test_custom_profile_is_range_checked(self):
        profile = normalize_augmentation({"preset": "custom", "brightness": .3, "fliplr": .25})
        self.assertEqual(profile["brightness"], .3)
        self.assertEqual(profile["fliplr"], .25)
        with self.assertRaisesRegex(ValueError, "fliplr"):
            normalize_augmentation({"preset": "custom", "fliplr": 1.1})

    def test_unknown_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支援"):
            normalize_augmentation({"preset": "off", "surprise": True})

    def test_detection_boxes_and_masks_follow_horizontal_flip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is optional in the workbench runtime")
        image = torch.zeros((3, 4, 5))
        target = {"boxes": torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
                  "masks": torch.tensor([[[1, 0, 0, 0, 0]] * 4])}
        profile = {"preset": "custom", "fliplr": 1.0, "flipud": 0.0,
                   "brightness": 0.0, "contrast": 0.0}
        with patch("workbench.augmentation.random.random", side_effect=[0.0, 1.0]):
            _image, transformed = augment_dense_target(image, target, profile, torch)
        self.assertEqual(transformed["boxes"].tolist(), [[3.0, 1.0, 5.0, 3.0]])
        self.assertEqual(transformed["masks"][0, 0].tolist(), [0, 0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
