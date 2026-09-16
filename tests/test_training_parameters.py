import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from workbench.model_registry import MODELS
from workbench.training import TrainingWorkspace
from workbench.training_parameters import create_optimizer, parameter_schema, validate_config
from workbench.classification_engine import train as train_classification


class TrainingParameterTests(unittest.TestCase):
    @staticmethod
    def definition(key):
        return {**next(model for model in MODELS if model["key"] == key), "train": True}

    def test_model_schemas_only_expose_consumed_parameters(self):
        baseline = parameter_schema(self.definition("pixel_prototype_v1"))
        self.assertEqual({item["key"] for item in baseline}, {"epochs", "threshold_min", "threshold_max"})
        self.assertEqual(parameter_schema(self.definition("efficientad")), [])
        for key, size, multiple in (("resnet18_classification", 224, None),
                                    ("maskrcnn_resnet50_fpn", 640, None), ("yolo26n_seg", 640, 32),
                                    ('yolo26n_detect', 640, 32)):
            with self.subTest(engine=key):
                schema = {item["key"]: item for item in parameter_schema(self.definition(key))}
                self.assertEqual(schema["image_size"]["default"], size)
                self.assertEqual(schema["image_size"].get("multiple_of"), multiple)
                self.assertEqual({option["value"] for option in schema["optimizer"]["options"]}, {"AdamW", "SGD"})

    def test_nonfinite_boolean_fractional_and_out_of_range_values_are_rejected(self):
        definition = self.definition("maskrcnn_resnet50_fpn")
        invalid = {"epochs": [0, 201, 2.5, True, float("nan")],
                   "seed": [-1, 2147483648, False, float("inf")],
                   "image_size": [127, 2049, 640.5], "batch_size": [0, 17, 1.5],
                   "learning_rate": [0, -1, 2, float("nan"), float("inf"), True],
                   "weight_decay": [-.1, 1.1, float("nan"), False],
                   "optimizer": ["auto", "RMSprop", None], "device": ["mps", True, ""]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(parameter=key, value=value), self.assertRaises(ValueError):
                    validate_config(definition, {key: value})
        with self.assertRaisesRegex(ValueError, "不支援"):
            validate_config(definition, {"lern_rate": .001})

    def test_model_specific_constraints_and_baseline_legacy_defaults(self):
        defaults = validate_config(self.definition("yolo26n_seg"), {})
        self.assertEqual(defaults["initialization"], "pretrained")
        self.assertEqual(defaults["gradient_accumulation"], 1)
        for invalid in ({"gradient_accumulation": 0}, {"gradient_accumulation": 1.5}, {"initialization": "unknown"}):
            with self.assertRaises(ValueError):
                validate_config(self.definition("yolo26n_seg"), invalid)
        with self.assertRaisesRegex(ValueError, "32"):
            validate_config(self.definition("yolo26n_seg"), {"image_size": 641})
        ultra = validate_config(self.definition("yolo26n_seg"), {"optimizer": "SGD"})
        self.assertEqual(ultra["momentum"], .9)
        detect = validate_config(self.definition('yolo26n_detect'), {})
        self.assertEqual(detect['initialization'], 'pretrained')
        self.assertNotIn('yolo_mask_policy', detect)
        baseline = self.definition("pixel_prototype_v1")
        for bounds in ({"threshold_min": 2, "threshold_max": 1}, {"threshold_min": 4}, {"threshold_min": 0}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                validate_config(baseline, bounds)
        effective = validate_config(baseline, {"seed": 42, "device": "auto", "image_size": 640,
                                               "batch_size": 1, "learning_rate": .0005})
        self.assertEqual(effective, {"epochs": 24, "threshold_min": .35, "threshold_max": 4., "device": "cpu"})
        with self.assertRaises(ValueError):
            validate_config(baseline, {"seed": -1})

    def test_invalid_run_never_creates_run_or_model_directories_or_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = TrainingWorkspace.__new__(TrainingWorkspace)
            workspace.datasets, workspace.runs, workspace.models = (root / name for name in ("datasets", "runs", "models"))
            workspace._maskrcnn_available = False
            workspace.registry = SimpleNamespace(model=lambda _key: self.definition("maskrcnn_resnet50_fpn"))
            manifest = workspace.datasets / "project" / "D001" / "manifest.json"
            manifest.parent.mkdir(parents=True); manifest.write_text(json.dumps({"assets":[{"split":"train","shapes":[{"label":"a"}]},{"split":"test","shapes":[{"label":"a"}]}]}))
            for invalid in ({"device": "invalid"}, {"learning_rate": float("nan")}, {"batch_size": 200},
                            {"seed": False}, {"image_size": 3.5}, {"weight_decay": -1}, {"scheduler":"plateau"}):
                with self.subTest(config=invalid), patch("workbench.training.subprocess.Popen") as process:
                    with self.assertRaises(ValueError):
                        workspace.start_run("project", "D001", {"engine": "maskrcnn_resnet50_fpn", **invalid})
                    process.assert_not_called()
                    self.assertFalse(workspace.runs.exists())
                    self.assertFalse(workspace.models.exists())

    def test_effective_parameters_are_frozen_in_run_before_worker_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = TrainingWorkspace.__new__(TrainingWorkspace)
            workspace.datasets, workspace.runs, workspace.models = (root / name for name in ("datasets", "runs", "models"))
            workspace._maskrcnn_available = False; workspace.lock = threading.RLock(); workspace.processes = {}
            workspace.registry = SimpleNamespace(model=lambda _key: self.definition("maskrcnn_resnet50_fpn"),
                                                  component_python=lambda _component: "python")
            manifest = workspace.datasets / "project" / "D001" / "manifest.json"
            manifest.parent.mkdir(parents=True); manifest.write_text(json.dumps({"assets":[{"split":"train","shapes":[{"label":"a"}]},{"split":"val","shapes":[{"label":"a"}]}]}))
            submitted = {"engine": "maskrcnn_resnet50_fpn", "epochs": 7, "seed": 9, "device": "cpu",
                         "image_size": 512, "batch_size": 3, "learning_rate": .012, "weight_decay": .023, "optimizer": "SGD",
                         "scheduler":"cosine", "min_learning_rate":.00012, "warmup_epochs":2}

            def launch(*_args, **_kwargs):
                persisted = json.loads((workspace.runs / "project/R001/run.json").read_text(encoding="utf-8"))
                self.assertEqual(persisted["config"], {key: value for key, value in submitted.items() if key != "engine"})
                return SimpleNamespace(pid=12345)

            with patch("workbench.training.subprocess.Popen", side_effect=launch), patch("workbench.training.threading.Thread"):
                created = workspace.start_run("project", "D001", submitted)
            self.assertEqual(created["config"]["learning_rate"], .012)
            self.assertEqual(created["config"]["optimizer"], "SGD")

    def test_optimizer_choice_and_numeric_values_reach_torch(self):
        torch = SimpleNamespace(optim=SimpleNamespace(AdamW=Mock(), SGD=Mock()))
        parameters = [object()]
        for name in ("AdamW", "SGD"):
            create_optimizer(torch, parameters, {"optimizer": name, "learning_rate": .015, "weight_decay": .025})
            getattr(torch.optim, name).assert_called_once_with(parameters, lr=.015, weight_decay=.025)

    def test_classification_passes_size_above_512_to_dataset_without_clamping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run_dir = root / "run"; run_dir.mkdir()
            manifest = root / "manifest.json"; manifest.write_text(json.dumps({"classes": ["ok", "ng"]}))
            (run_dir / "run.json").write_text(json.dumps({"engine": "resnet18_classification", "config": {"image_size": 1024}}))
            torch = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(is_available=lambda: False), device=lambda name: name)
            with patch("workbench.classification_engine._modules", return_value=(torch, None, None)), \
                 patch("workbench.classification_engine._model", return_value=(Mock(), torch)), \
                 patch("workbench.classification_engine.ClassificationDataset", side_effect=RuntimeError("dataset captured")) as dataset:
                with self.assertRaisesRegex(RuntimeError, "dataset captured"):
                    train_classification(manifest, run_dir, root / "model")
            self.assertEqual(dataset.call_args.args[-1], 1024)


if __name__ == "__main__":
    unittest.main()
