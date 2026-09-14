import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import numpy as np
from PIL import Image

from composer_core.geometry import encode_rle
from workbench.classification_engine import _asset_label
from workbench.training_engine import atomic_json
from workbench.ultralytics_engine import (
    _numeric_metrics, _result_metrics, prepare_yolo_dataset, train as train_ultralytics,
)


class NewModelAdapterTests(unittest.TestCase):
    def test_classification_requires_exactly_one_distinct_label(self):
        classes = ["ok", "ng"]
        self.assertEqual(_asset_label({"shapes": [{"label": "ng"}, {"label": "ng"}]}, classes), 1)
        with self.assertRaisesRegex(ValueError, "只有一個"):
            _asset_label({"name": "mixed.png", "shapes": [{"label": "ok"}, {"label": "ng"}]}, classes)
        with self.assertRaisesRegex(ValueError, "只有一個"):
            _asset_label({"name": "empty.png", "shapes": []}, classes)

    def test_rtdetr_dataset_uses_tight_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "images").mkdir()
            image = root / "images" / "A001.png"; Image.new("RGB", (100, 50)).save(image)
            import hashlib
            asset = {"asset_id": "A001", "name": "sample.png", "width": 100, "height": 50,
                     "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "image_file": "images/A001.png",
                     "split": "train", "shapes": [{"id": "s1", "type": "rectangle", "label": "part",
                     "x": 10, "y": 5, "width": 20, "height": 10}],
                     "objects": [{"shape_id": "s1", "class_id": 0, "label": "part", "type": "rectangle",
                     "bbox_xywh": [10, 5, 20, 10]}]}
            manifest = root / "manifest.json"; atomic_json(manifest, {"classes": ["part"], "assets": [asset]})
            data = prepare_yolo_dataset(manifest, root / "converted", "object_detection")
            self.assertTrue(data.is_file())
            self.assertEqual((root / "converted/labels/train/A001.txt").read_text().strip(),
                             "0 0.20000000 0.20000000 0.20000000 0.20000000")

    def test_yolo_seg_rejects_mask_holes_instead_of_losing_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "images").mkdir()
            image = root / "images" / "A001.png"; Image.new("RGB", (20, 20)).save(image)
            import hashlib
            mask = np.zeros((20, 20), np.uint8); mask[2:18, 2:18] = 1; mask[7:13, 7:13] = 0
            shape = {"id": "s1", "type": "mask", "label": "part", "x": 0, "y": 0,
                     "width": 20, "height": 20, "counts": encode_rle(mask)}
            asset = {"asset_id": "A001", "name": "hole.png", "width": 20, "height": 20,
                     "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "image_file": "images/A001.png",
                     "split": "train", "shapes": [shape], "objects": [{"bbox_xywh": [2, 2, 16, 16]}]}
            manifest = root / "manifest.json"; atomic_json(manifest, {"classes": ["part"], "assets": [asset]})
            with self.assertRaisesRegex(ValueError, "無法無損"):
                prepare_yolo_dataset(manifest, root / "converted", "instance_segmentation")

    def test_ultralytics_metrics_distinguish_map50_from_map50_95(self):
        trainer = SimpleNamespace(metrics={"metrics/mAP50-95(B)": .31, "metrics/mAP50(B)": .62,
                                          "metrics/mAP50-95(M)": .23, "metrics/mAP50(M)": .54},
                                  loss_items=None)
        self.assertEqual(_numeric_metrics(trainer, "detect"),
                         {"val/box_map50_95": .31, "val/box_map50": .62})
        self.assertEqual(_numeric_metrics(trainer, "segment"),
                         {"val/mask_map50_95": .23, "val/mask_map50": .54})

    def test_ultralytics_metrics_omit_unmeasured_values_and_preserve_real_zero(self):
        trainer = SimpleNamespace(metrics={"metrics/mAP50-95(M)": float("nan"),
            "metrics/mAP50(M)": 0., "train/loss": float("inf"), "val/loss": .9}, loss_items=None)
        self.assertEqual(_numeric_metrics(trainer, "segment"), {"val/mask_map50": 0.})
        self.assertEqual(_numeric_metrics(SimpleNamespace(metrics={}), "detect"), {})
        trainer.metrics["train/loss"] = 0.
        self.assertEqual(_numeric_metrics(trainer, "segment"), {"train/loss": 0., "val/mask_map50": 0.})

    def test_ultralytics_result_omits_missing_metrics_without_faking_scores(self):
        self.assertEqual(_result_metrics(SimpleNamespace(), "detect", "test", 3),
                         {"split": "test", "images": 3})
        result = SimpleNamespace(seg=SimpleNamespace(map=None, map50=0., map75=float("nan")))
        self.assertEqual(_result_metrics(result, "segment", "val", 2),
                         {"split": "val", "images": 2, "mask_map50": 0.})

    def _run_fake_ultralytics(self, splits):
        class FakeModel:
            def __init__(self, architecture):
                self.architecture = architecture; self.callbacks = {}
                self.trainer = SimpleNamespace(epoch=0, metrics={"metrics/mAP50-95(B)": .31,
                    "metrics/mAP50(B)": .62}, loss_items=None, best="", last="")
            def add_callback(self, name, callback):
                self.callbacks[name] = callback
            def train(self, **options):
                best = Path(options["project"]) / options["name"] / "weights" / "best.pt"
                best.parent.mkdir(parents=True); best.write_bytes(b"checkpoint")
                self.trainer.best = str(best)
                for epoch in range(options["epochs"]):
                    self.trainer.epoch = epoch; self.callbacks["on_fit_epoch_end"](self.trainer)
            def val(self, **_options):
                metric = SimpleNamespace(map=.4, map50=.7, map75=.3)
                return SimpleNamespace(box=metric, seg=metric)
        fake = SimpleNamespace(RTDETR=FakeModel, YOLO=FakeModel)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"ultralytics": fake}):
            root = Path(directory); dataset = root / "dataset"; (dataset / "images").mkdir(parents=True)
            assets = []
            import hashlib
            for index, split in enumerate(splits):
                image = dataset / "images" / f"A{index}.png"; Image.new("RGB", (20, 20), (index, 0, 0)).save(image)
                shape = {"id": f"s{index}", "type": "rectangle", "label": "part", "x": 2, "y": 3,
                         "width": 10, "height": 8}
                assets.append({"asset_id": f"A{index}", "name": image.name, "width": 20, "height": 20,
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "image_file": f"images/{image.name}",
                    "split": split, "shapes": [shape], "objects": [{"shape_id": shape["id"], "class_id": 0,
                    "label": "part", "type": "rectangle", "bbox_xywh": [2, 3, 10, 8]}]})
            manifest = dataset / "manifest.json"; atomic_json(manifest, {"dataset_version_id": "D001",
                "classes": ["part"], "assets": assets})
            run_dir, model_dir = root / "run", root / "model"; run_dir.mkdir(); model_dir.mkdir()
            atomic_json(run_dir / "run.json", {"run_id": "R001", "model_version_id": "M001",
                "engine": "rt_detr_r50", "config": {"epochs": 2, "device": "cpu", "image_size": 128,
                "batch_size": 1, "seed": 42}})
            result = train_ultralytics(manifest, run_dir, model_dir)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len((run_dir / "metrics.jsonl").read_text().splitlines()), 2)
            self.assertEqual((model_dir / "checkpoint.pt").read_bytes(), b"checkpoint")
            record = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(record["test"]["box_map50_95"], .4)
            metrics = [json.loads(row) for row in (run_dir / "metrics.jsonl").read_text().splitlines()]
            for row in metrics:
                self.assertEqual(row["val/box_map50"], .62)
                self.assertNotIn("train/loss", row)
            data_yaml = json.loads((run_dir / "dataset/data.yaml").read_text(encoding="utf-8"))
            return result, record, data_yaml

    def test_ultralytics_worker_contract_writes_metrics_checkpoint_and_record(self):
        result, record, _data = self._run_fake_ultralytics(("train", "val", "test"))
        self.assertEqual(result["validation_split"], "val")
        self.assertEqual(record["validation"]["split"], "val")
        self.assertEqual(record["test"]["split"], "test")

    def test_ultralytics_validation_fallback_records_actual_test_split(self):
        result, record, data = self._run_fake_ultralytics(("train", "test", "test"))
        self.assertEqual(result["validation_split"], "test")
        self.assertEqual(data["val"], "images/test")
        self.assertEqual(record["validation"]["split"], "test")
        self.assertEqual(record["validation"]["images"], 2)
        self.assertEqual(result["evaluation"]["validation"]["split"], "test")

    def test_ultralytics_missing_test_keeps_validation_identity(self):
        result, record, _data = self._run_fake_ultralytics(("train", "val", "val"))
        self.assertEqual(result["validation_split"], "val")
        self.assertEqual(record["test"]["split"], "val")
        self.assertEqual(record["test"]["images"], 2)


if __name__ == "__main__":
    unittest.main()
