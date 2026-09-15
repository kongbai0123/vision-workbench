from pathlib import Path
import json
import sys
import tempfile
import time
import unittest
import zipfile

import numpy as np
from PIL import Image

from composer_core.geometry import decode_rle, encode_rle
from workbench.store import ProjectStore
from workbench.training import TrainingWorkspace, _with_evaluation_reassessment


class TrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="vision-workbench-training-")
        self.root = Path(self.tmp.name)
        self.store = ProjectStore(self.root / "data" / "projects")
        self.project = self.store.create_project("像素遮罩訓練")
        self.pid = self.project["id"]
        self.records = []
        for index, split in enumerate(["train", "train", "val", "val", "test", "test"]):
            image = np.full((36, 48, 3), (24, 28, 32), np.uint8)
            left = 5 + index
            image[8:29, left:left + 22] = (185, 72, 48)
            path = self.root / f"sample-{index}.png"
            Image.fromarray(image).save(path)
            mask = np.zeros((36, 48), np.uint8)
            mask[8:29, left:left + 22] = 1
            self.records.append({"path": str(path), "name": path.name, "split": split,
                "batch_id": f"{split}-{index}", "review_state": "approved",
                "source": {"kind": "test", "index": index},
                "shapes": [{"id": f"shape-{index}", "type": "mask", "label": "handlebar",
                            "x": 0, "y": 0, "width": 48, "height": 36, "counts": encode_rle(mask)}]})
        self.store.add_assets(self.pid, self.records)
        self.workspace = TrainingWorkspace(self.root / "data", self.store)

    def tearDown(self):
        self.workspace.close()
        self.tmp.cleanup()

    def wait_run(self, run_id, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.workspace.run(self.pid, run_id)
            if run["status"] in {"completed", "failed", "stopped"}:
                # The worker persists its terminal state just before Python exits.
                # Wait for the reaper to close inherited Windows log handles so
                # TemporaryDirectory cleanup cannot race the child process.
                while (self.pid, run_id) in self.workspace.processes and time.monotonic() < deadline:
                    time.sleep(.02)
                return run
            time.sleep(.04)
        self.fail("training worker timed out")

    def test_dataset_version_is_immutable_and_preserves_native_masks(self):
        report = self.workspace.readiness(self.pid)
        self.assertTrue(report["ready"], report)
        dataset = self.workspace.create_dataset_version(self.pid)
        self.assertEqual(dataset["id"], "D001")
        self.assertEqual(dataset["splits"], {"train": 2, "val": 2, "test": 2})
        manifest_path = self.workspace.datasets / self.pid / "D001" / "manifest.json"
        before = manifest_path.read_bytes()
        manifest = json.loads(before)
        shape = manifest["assets"][0]["shapes"][0]
        decoded = decode_rle(shape["counts"], 48, 36)
        self.assertEqual(int(np.count_nonzero(decoded)), 21 * 22)
        current = self.store.get_asset(self.pid, manifest["assets"][0]["asset_id"])
        self.store.save_asset(self.pid, current["id"], [], current["revision"])
        self.assertEqual(manifest_path.read_bytes(), before)

    def test_review_compatibility_scans_latest_pending_and_approved_annotations(self):
        initial = self.workspace.review_yolo_compatibility(self.pid)
        self.assertTrue(initial["compatible"], initial)
        self.assertEqual(initial["review_scope"], {"pending": 0, "approved": 6, "excluded_rejected": 0})

        asset = self.store.get_asset(self.pid, self.store.get_project(self.pid)["assets"][0]["id"])
        mask = decode_rle(asset["shapes"][0]["counts"], asset["width"], asset["height"])
        mask[14:20, 14:20] = 0
        shapes = list(asset["shapes"]); shapes[0] = dict(shapes[0], counts=encode_rle(mask))
        self.store.save_asset(self.pid, asset["id"], shapes, asset["revision"])

        current = self.store.get_project(self.pid)
        report = self.workspace.review_yolo_compatibility(self.pid)
        self.assertEqual(report["project_revision"], current["revision"])
        self.assertEqual(report["review_scope"]["pending"], 1)
        self.assertFalse(report["compatible"])
        self.assertEqual(report["summary"]["blocked_assets"], 1)

    def test_real_train_evaluate_predict_accept_roundtrip(self):
        dataset = self.workspace.create_dataset_version(self.pid)
        run = self.workspace.start_run(self.pid, dataset["id"], {"engine": "pixel_prototype_v1", "epochs": 6,
                                                                "threshold_min": .5, "threshold_max": 2.5})
        finished = self.wait_run(run["run_id"])
        self.assertEqual(finished["status"], "completed", finished)
        self.assertGreaterEqual(finished["evaluation"]["test"]["mean_iou"], .95)
        metrics = [json.loads(row) for row in (self.workspace.runs / self.pid / run["run_id"] / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(metrics[0]["threshold"], .5)
        self.assertEqual(metrics[-1]["threshold"], 2.5)
        model_id = finished["model_version_id"]
        model = self.workspace.model(self.pid, model_id)
        self.assertEqual(model["dataset_version_id"], "D001")
        exported = self.workspace.export_model(self.pid, model_id)
        self.assertEqual(exported["format"], "vision-workbench-model-bundle")
        self.assertEqual(exported["model_version_id"], model_id)
        self.assertEqual(len(exported["sha256"]), 64)
        with zipfile.ZipFile(exported["path"]) as archive:
            self.assertIn("export-manifest.json", archive.namelist())
            self.assertIn("model/model.json", archive.namelist())
            manifest = json.loads(archive.read("export-manifest.json"))
        self.assertEqual(manifest["dataset_version_id"], "D001")
        self.assertEqual(self.workspace.list_model_exports(self.pid)[0]["export_id"], "E001")
        target_image = np.full((36, 48, 3), (24, 28, 32), np.uint8)
        target_image[9:30, 11:33] = (185, 72, 48)
        target = self.root / "predict.png"
        Image.fromarray(target_image).save(target)
        added = self.store.add_assets(self.pid, [{"path": str(target), "name": "predict.png",
            "batch_id": "prediction", "review_state": "pending", "source": {"kind": "test"}}])
        aid = added["asset_ids"][0]
        candidate = self.workspace.create_predictions(self.pid, model_id, [aid])
        shapes = candidate["assets"][0]["shapes"]
        self.assertTrue(shapes)
        self.assertTrue(all(shape["type"] == "mask" for shape in shapes))
        result = self.workspace.accept_predictions(candidate["candidate_id"], [aid])
        self.assertEqual(result["accepted"], [aid])
        saved = self.store.get_asset(self.pid, aid)
        self.assertEqual(saved["review_state"], "pending")
        self.assertEqual(saved["shapes"][0]["metadata"]["model_version_id"], model_id)

    def test_small_thresholds_keep_their_precision_in_metrics_and_model(self):
        dataset = self.workspace.create_dataset_version(self.pid)
        run = self.workspace.start_run(self.pid, dataset["id"], {"engine": "pixel_prototype_v1", "epochs": 2,
                                                                "threshold_min": 1e-8, "threshold_max": 1e-7})
        finished = self.wait_run(run["run_id"])
        self.assertEqual(finished["status"], "completed", finished)
        metrics = [json.loads(row) for row in (self.workspace.runs / self.pid / run["run_id"] / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["threshold"] for row in metrics], [1e-8, 1e-7])
        model = self.workspace.model(self.pid, finished["model_version_id"])
        self.assertTrue(all(1e-8 <= value <= 1e-7 for value in model["thresholds"].values()))

    def test_readiness_blocks_unreviewed_and_source_group_leakage(self):
        project = self.store.create_project("資料檢查")
        source = self.root / "blocked.png"
        Image.new("RGB", (10, 10), (0, 0, 0)).save(source)
        self.store.add_assets(project["id"], [{"path": str(source), "split": "train", "batch_id": "same",
            "review_state": "approved", "shapes": [{"id": "x", "type": "rectangle", "label": "part",
            "x": 1, "y": 1, "width": 5, "height": 5}]}])
        report = self.workspace.readiness(project["id"])
        codes = {item["code"] for item in report["blockers"]}
        self.assertIn("no_validation_split", codes)

    def test_versioned_reassessment_overlays_scores_and_preserves_legacy_values(self):
        directory = self.root / "historical"
        directory.mkdir()
        legacy = {"model_version_id": "M001", "validation": {"mean_iou": .81},
                  "test": {"mean_iou": .87}}
        reassessment = {"schema_version": 2, "valid": True, "reason": "metric correction",
                        "validation": {"mean_iou": .33}, "test": {"mean_iou": .59},
                        "protocol": {"selection_split": "val", "test_present": True}}
        (directory / "evaluation.v2.json").write_text(json.dumps(reassessment), encoding="utf-8")
        current = _with_evaluation_reassessment(legacy, directory, model=True)
        self.assertEqual(current["validation"]["mean_iou"], .33)
        self.assertEqual(current["test"]["mean_iou"], .59)
        self.assertEqual(current["legacy_evaluation"], {"validation": {"mean_iou": .81},
                                                         "test": {"mean_iou": .87}})
        self.assertEqual(legacy["validation"]["mean_iou"], .81)

    def test_reopen_marks_orphaned_active_run_failed(self):
        run_dir = self.root / "runs" / self.pid / "R099"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({"run_id": "R099", "project_id": self.pid,
            "status": "running", "worker_pid": 99999999, "updated_at": 0}), encoding="utf-8")
        reopened = TrainingWorkspace(self.root, self.store, python_executable=sys.executable)
        self.assertEqual(reopened.run(self.pid, "R099")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
