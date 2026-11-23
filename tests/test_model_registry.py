import sys
import tempfile
from contextlib import chdir
from pathlib import Path
import unittest
from unittest.mock import patch

from workbench.model_registry import ModelRegistry


class ModelRegistryTests(unittest.TestCase):
    def test_catalog_keeps_ready_installable_and_planned_models_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory), python_executable=sys.executable)
            catalog = registry.snapshot()
        models = {model["key"]: model for model in catalog["models"]}
        for key in ("fasterrcnn_mobilenet_v3_large_fpn", "deeplabv3_mobilenet_v3_large",
                    "efficientad", "rt_detr_r50", "yolo26n_seg", "yolo26n_detect", "yolo26s_detect"):
            self.assertIn(key, models)
        self.assertEqual(models["efficientad"]["integration"], "planned")
        self.assertFalse(models["efficientad"]["train"])
        self.assertEqual(models["mobilenet_v3_large_classification"]["integration"], "ready")
        self.assertFalse(models["mobilenet_v3_large_classification"]["predict"])
        self.assertEqual(models["rt_detr_r50"]["integration"], "ready")
        self.assertEqual(models["rt_detr_r50"]["component"], "ultralytics")
        self.assertEqual(models['yolo26n_detect']['task'], 'object_detection')
        components = {component["id"]: component for component in catalog["components"]}
        self.assertTrue(components["ultralytics"]["installable"])
        self.assertEqual({key: component["install_path"] for key, component in components.items()
                          if component["installable"]},
                         {"torchvision": ".venv-training", "ultralytics": ".venv-models/ultralytics"})
        self.assertEqual(models["pixel_prototype_v1"]["runtime_state"], "ready")

    def test_training_interpreters_resolve_from_the_app_root(self):
        override_key = "VISION_WORKBENCH_TRAINING_PYTHON"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            app_root = root / "工作台 副本"
            registry = ModelRegistry(app_root)
            local_python = app_root / ".venv-training" / "Scripts" / "python.exe"
            local_python.parent.mkdir(parents=True)
            local_python.touch()
            with chdir(root), patch.dict("os.environ", {override_key: ""}):
                self.assertEqual(registry.component_python("torchvision"), local_python)
                self.assertEqual(registry.component_python("ultralytics"),
                                 app_root / ".venv-models" / "ultralytics" / "Scripts" / "python.exe")
            relative_python = "alternative runtime/Scripts/python.exe"
            with chdir(root), patch.dict("os.environ", {override_key: relative_python}):
                self.assertEqual(registry.training_python, app_root / relative_python)
            absolute_python = root / "external runtime" / "Scripts" / "python.exe"
            with chdir(root), patch.dict("os.environ", {override_key: str(absolute_python)}):
                self.assertEqual(registry.training_python, absolute_python)

    def test_ultralytics_install_targets_the_app_root_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            app_root = root / "工作台 副本"
            registry = ModelRegistry(app_root)
            with (chdir(root), patch("workbench.process_control.run_controlled") as run,
                  patch.object(registry, "component_status", return_value={"ultralytics": {"state": "ready"}}),
                  patch.object(registry, "snapshot")):
                registry.install("ultralytics", lambda *_args: None)
            runtime = app_root / ".venv-models" / "ultralytics"
            self.assertEqual([call.args[0] for call in run.call_args_list], [
                [sys.executable, "-m", "venv", str(runtime)],
                [str(runtime / "Scripts" / "python.exe"), "-m", "pip", "install", "-r",
                 str(app_root / "requirements-ultralytics.txt")],
            ])
            self.assertEqual([call.kwargs["cwd"] for call in run.call_args_list], [app_root, app_root])

    def test_unfinished_component_cannot_be_installed_as_if_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory), python_executable=sys.executable)
            with self.assertRaisesRegex(ValueError, "待 Workbench"):
                registry.install("anomalib", lambda *_args: None)


if __name__ == "__main__":
    unittest.main()
