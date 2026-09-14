import sys
import tempfile
from pathlib import Path
import unittest

from workbench.model_registry import ModelRegistry


class ModelRegistryTests(unittest.TestCase):
    def test_catalog_keeps_ready_installable_and_planned_models_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory), python_executable=sys.executable)
            catalog = registry.snapshot()
        models = {model["key"]: model for model in catalog["models"]}
        for key in ("fasterrcnn_mobilenet_v3_large_fpn", "deeplabv3_mobilenet_v3_large",
                    "efficientad", "rt_detr_r50", "yolo26n_seg"):
            self.assertIn(key, models)
        self.assertEqual(models["efficientad"]["integration"], "planned")
        self.assertFalse(models["efficientad"]["train"])
        self.assertEqual(models["pixel_prototype_v1"]["runtime_state"], "ready")

    def test_unfinished_component_cannot_be_installed_as_if_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory), python_executable=sys.executable)
            with self.assertRaisesRegex(ValueError, "待 Workbench"):
                registry.install("anomalib", lambda *_args: None)


if __name__ == "__main__":
    unittest.main()
