import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from workbench.ultralytics_engine import (
    ULTRALYTICS_ENGINES, _RunMetricsRecorder, _initialization_record,
    _initialization_source, _numeric_metrics,
)


class UltralyticsMetricsTests(unittest.TestCase):
    def test_epoch_means_are_collected_from_current_dict_and_legacy_vector(self):
        for average in ({"box_loss": 1.25, "seg_loss": 2.5}, np.array([1.25, 2.5])):
            with self.subTest(kind=type(average).__name__):
                trainer = SimpleNamespace(tloss=average, loss_items={"box_loss": 99, "seg_loss": 999},
                                          loss_names=("box_loss", "seg_loss"), metrics={"metrics/mAP50(M)": 0.0})
                metrics = _numeric_metrics(trainer, "segment")
                self.assertEqual(metrics["train/box_loss"], 1.25)
                self.assertEqual(metrics["train/seg_loss"], 2.5)
                self.assertEqual(metrics["train/loss"], 3.75)
                self.assertEqual(metrics["val/mask_map50"], 0.0)

    def test_partial_nonfinite_components_do_not_become_a_misleading_total(self):
        trainer = SimpleNamespace(tloss={"box_loss": 1.25, "seg_loss": float("nan")}, metrics={})
        self.assertEqual(_numeric_metrics(trainer, "segment"), {"train/box_loss": 1.25})
        # A last-batch observation must not be mislabeled as an epoch average.
        self.assertEqual(_numeric_metrics(SimpleNamespace(loss_items=np.array([3.0]), metrics={}), "segment"), {})

    def test_final_evaluation_after_early_stop_is_not_an_extra_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = _RunMetricsRecorder(root, {"config": {"epochs": 50, "batch_size": 1}},
                                          ULTRALYTICS_ENGINES["yolo26n_seg"])
            trainer = SimpleNamespace(epoch=0, tloss={"seg_loss": 2}, metrics={},
                                      optimizer=SimpleNamespace(param_groups=[{"lr": .001}]))
            for epoch in range(3):
                trainer.epoch = epoch
                recorder.on_epoch_start(trainer)
                recorder.on_epoch_end(trainer)
                recorder.on_epoch_end(trainer)
            trainer.epoch = 3  # final_eval increments the epoch after early stop
            trainer.metrics = {"metrics/mAP50(M)": .8}
            recorder.on_epoch_end(trainer)
            rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in rows], [1, 2, 3])
            self.assertEqual(recorder.run["epoch"], 3)
            self.assertNotIn("val/mask_map50", rows[-1])
            self.assertNotIn("optimizer_steps", recorder.run["execution"])

    def test_actual_optimizer_post_hook_excludes_amp_skipped_updates(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is optional in the workbench runtime")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = _RunMetricsRecorder(root, {"config": {"epochs": 2, "batch_size": 1}},
                                          ULTRALYTICS_ENGINES["yolo26n_seg"])
            parameter = torch.nn.Parameter(torch.tensor(1.0))
            optimizer = torch.optim.SGD([parameter], lr=.01)
            scaler = torch.amp.GradScaler("cpu")
            trainer = SimpleNamespace(epoch=0, optimizer=optimizer, batch_size=1, accumulate=1,
                                      tloss={"seg_loss": torch.tensor(2.0)}, metrics={})
            try:
                recorder.on_epoch_start(trainer)
                for multiplier in (1., float("inf"), 2.):
                    optimizer.zero_grad()
                    scaler.scale(parameter * multiplier).backward()
                    scaler.step(optimizer)
                    scaler.update()
                recorder.on_epoch_end(trainer)
                trainer.epoch = 1
                recorder.on_epoch_start(trainer)
                # No optimizer update in this epoch: no invented actual LR.
                optimizer.param_groups[0]["lr"] = .005
                recorder.on_epoch_end(trainer)
                rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
                self.assertEqual(rows[0]["train/optimizer_steps"], 2)
                self.assertEqual(rows[0]["train/optimizer_steps_epoch"], 2)
                self.assertEqual(rows[0]["train/learning_rate"], .01)
                self.assertEqual(rows[1]["train/optimizer_steps"], 2)
                self.assertEqual(rows[1]["train/optimizer_steps_epoch"], 0)
                self.assertNotIn("train/learning_rate", rows[1])
                self.assertEqual(recorder.run["execution"]["optimizer_step_measurement"], "post_step_hook")
            finally:
                recorder.close()
            self.assertFalse(optimizer._optimizer_step_post_hooks)

    def test_pretrained_is_yolo_default_and_weights_are_auditable(self):
        yolo = ULTRALYTICS_ENGINES["yolo26n_seg"]
        self.assertEqual(_initialization_source(yolo, {}), ("pretrained", "yolo26n-seg.pt"))
        self.assertEqual(_initialization_source(yolo, {"initialization": "scratch"}),
                         ("scratch", "yolo26n-seg.yaml"))
        self.assertEqual(_initialization_source(ULTRALYTICS_ENGINES["rt_detr_r50"], {}),
                         ("scratch", "rtdetr-resnet50.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "yolo26n-seg.pt"
            checkpoint.write_bytes(b"known-initial-weights")
            record = _initialization_record(SimpleNamespace(ckpt_path=str(checkpoint)), "pretrained", checkpoint.name)
            self.assertEqual(record["weights_sha256"], hashlib.sha256(b"known-initial-weights").hexdigest())
            self.assertEqual(record["weights_path"], str(checkpoint.resolve()))


if __name__ == "__main__":
    unittest.main()
