"""Explicit TorchVision classification adapter smoke test."""
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workbench.classification_engine import CLASSIFICATION_ENGINES, _model
from workbench.store import ProjectStore
from workbench.training import TrainingWorkspace


def wait(workspace, project_id, run_id, timeout=360):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = workspace.run(project_id, run_id)
        if run["status"] in {"completed", "failed", "stopped"}:
            return run
        time.sleep(.25)
    workspace.stop_run(project_id, run_id)
    raise TimeoutError(f"{run_id} timed out")


def main():
    # Verify every registered constructor with the pinned TorchVision runtime.
    for engine in CLASSIFICATION_ENGINES:
        model, torch = _model(engine, 2)
        with torch.inference_mode():
            assert tuple(model(torch.zeros((1, 3, 128, 128))).shape) == (1, 2)
        del model
    with tempfile.TemporaryDirectory(prefix="vision-classification-") as directory:
        root = Path(directory); store = ProjectStore(root / "projects")
        project_id = store.create_project("classification smoke")["id"]
        records = []
        for index, (split, label) in enumerate((("train", "ok"), ("train", "ng"), ("val", "ok"),
                                                ("val", "ng"), ("test", "ok"), ("test", "ng"))):
            pixels = np.full((64, 64, 3), 32 if label == "ok" else 220, np.uint8)
            pixels[0, 0] = (index, index, index)
            path = root / f"{split}-{label}.png"; Image.fromarray(pixels).save(path)
            records.append({"path": str(path), "split": split, "batch_id": f"b{index}",
                "review_state": "approved", "shapes": [{"id": f"s{index}", "type": "rectangle", "label": label,
                "x": 8, "y": 8, "width": 48, "height": 48}]})
        store.add_assets(project_id, records)
        workspace = TrainingWorkspace(root, store)
        dataset = workspace.create_dataset_version(project_id)
        run = workspace.start_run(project_id, dataset["id"], {"engine": "resnet18_classification", "epochs": 1,
            "device": "auto", "image_size": 128, "batch_size": 2})
        finished = wait(workspace, project_id, run["run_id"])
        if finished["status"] != "completed":
            raise RuntimeError(finished)
        model = workspace.model(project_id, finished["model_version_id"])
        assert model["task"] == "image_classification"
        assert "macro_f1" in model["test"]
        try:
            workspace.create_predictions(project_id, model["model_version_id"], [store.snapshot(project_id)["assets"][0]["id"]])
        except ValueError as exc:
            assert "Bounding Box" in str(exc)
        else:
            raise AssertionError("classification must not create full-image boxes")
        workspace.close()
        print("CLASSIFICATION_MODELS_SMOKE_OK", model["model_version_id"], model["test"])


if __name__ == "__main__":
    main()
