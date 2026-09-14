"""Explicit end-to-end smoke test for the optional Mask R-CNN runtime."""
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from composer_core.geometry import encode_rle
from workbench.store import ProjectStore
from workbench.training import TrainingWorkspace


def main():
    with tempfile.TemporaryDirectory(prefix="vision-maskrcnn-") as directory:
        root = Path(directory)
        store = ProjectStore(root / "projects")
        project_id = store.create_project("Mask R-CNN smoke")["id"]
        for index, split in enumerate(("train", "val", "test")):
            pixels = np.full((128, 128, 3), 235, dtype=np.uint8)
            pixels[30:98, 28 + index:96 + index] = (35, 95, 160)
            path = root / f"sample-{index}.png"; Image.fromarray(pixels).save(path)
            mask = np.zeros((128, 128), dtype=np.uint8); mask[30:98, 28 + index:96 + index] = 1
            store.add_assets(project_id, [{"path": str(path), "split": split, "batch_id": f"batch-{index}",
                "review_state": "approved", "shapes": [{"id": f"mask-{index}", "type": "mask", "label": "part",
                "counts": encode_rle(mask)}]}])
        workspace = TrainingWorkspace(root, store)
        capability = next(engine for engine in workspace.capabilities()["engines"] if engine["key"] == "maskrcnn_resnet50_fpn")
        if not capability["train"]:
            raise RuntimeError("Mask R-CNN runtime is unavailable; run bootstrap.ps1 -Training")
        dataset = workspace.create_dataset_version(project_id)
        run = workspace.start_run(project_id, dataset["id"], {"engine": "maskrcnn_resnet50_fpn", "epochs": 1,
                                  "device": "auto", "image_size": 128, "batch_size": 1})
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            current = workspace.run(project_id, run["run_id"])
            if current["status"] in {"completed", "failed", "stopped"}:
                break
            time.sleep(.25)
        else:
            workspace.stop_run(project_id, run["run_id"])
            raise TimeoutError("Mask R-CNN smoke test timed out")
        if current["status"] != "completed":
            log = workspace._run_path(project_id, run["run_id"]).parent / "worker.stderr.log"
            raise AssertionError(f"{current}\n{log.read_text(encoding='utf-8', errors='replace') if log.is_file() else ''}")
        model = workspace.model(project_id, current["model_version_id"])
        assert (workspace._model_path(project_id, model["model_version_id"]).parent / "checkpoint.pt").is_file()
        candidate = workspace.create_predictions(project_id, model["model_version_id"], [store.snapshot(project_id)["assets"][0]["id"]])
        assert len(candidate["assets"]) == 1
        deadline = time.monotonic() + 5
        while workspace.processes and time.monotonic() < deadline:
            time.sleep(.05)
        workspace.close()
        print("MASKRCNN_SMOKE_OK", current["run_id"], model["model_version_id"], current["evaluation"])


if __name__ == "__main__":
    main()
