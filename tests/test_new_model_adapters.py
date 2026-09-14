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
from workbench.ultralytics_engine import prepare_yolo_dataset, train as train_ultralytics


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

    def test_ultralytics_worker_contract_writes_metrics_checkpoint_and_record(self):
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
            for index, split in enumerate(("train", "val", "test")):
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


if __name__ == "__main__":
    unittest.main()
