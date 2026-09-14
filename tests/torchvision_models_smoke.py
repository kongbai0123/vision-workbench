"""Explicit CUDA/CPU smoke test for added TorchVision model adapters."""
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from composer_core.geometry import encode_rle
from workbench.store import ProjectStore
from workbench.training import TrainingWorkspace


ENGINES = ("fasterrcnn_mobilenet_v3_large_320_fpn", "deeplabv3_mobilenet_v3_large")


def wait(workspace, project_id, run_id, timeout=360):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = workspace.run(project_id, run_id)
        if run["status"] in {"completed", "failed", "stopped"}: return run
        time.sleep(.25)
    workspace.stop_run(project_id, run_id)
    raise TimeoutError(f"{run_id} timed out")


def main():
    with tempfile.TemporaryDirectory(prefix="vision-torchvision-models-") as directory:
        root = Path(directory); store = ProjectStore(root / "projects")
        project_id = store.create_project("TorchVision adapters smoke")["id"]
        for index, split in enumerate(("train", "train", "val", "test")):
            pixels = np.full((96, 96, 3), 225, dtype=np.uint8)
            pixels[22:76, 20 + index:70 + index] = (35, 100, 170)
            path = root / f"sample-{index}.png"; Image.fromarray(pixels).save(path)
            mask = np.zeros((96, 96), dtype=np.uint8); mask[22:76, 20 + index:70 + index] = 1
            store.add_assets(project_id, [{"path": str(path), "split": split, "batch_id": f"batch-{index}",
                "review_state": "approved", "shapes": [{"id": f"mask-{index}", "type": "mask",
                "label": "part", "counts": encode_rle(mask)}]}])
        workspace = TrainingWorkspace(root, store)
        available = {item["key"]: item["train"] for item in workspace.capabilities()["engines"]}
        dataset = workspace.create_dataset_version(project_id); results = []
        for engine in ENGINES:
            if not available.get(engine): raise RuntimeError(f"{engine} unavailable")
            run = workspace.start_run(project_id, dataset["id"], {"engine": engine, "epochs": 1,
                "device": "auto", "image_size": 128, "batch_size": 2})
            finished = wait(workspace, project_id, run["run_id"])
            if finished["status"] != "completed": raise RuntimeError(f"{engine}: {finished}")
            model = workspace.model(project_id, finished["model_version_id"])
            results.append((engine, model["task"], model["test"]))
            process_deadline = time.monotonic() + 10
            while workspace.processes and time.monotonic() < process_deadline:
                time.sleep(.05)
        workspace.close()
        print("TORCHVISION_MODELS_SMOKE_OK", results)


if __name__ == "__main__": main()
