"""CLI boundary for a Vision Workbench training run."""
from __future__ import annotations

import argparse
from pathlib import Path

from .training_engine import ENGINE_KEY as BASELINE_ENGINE, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vision Workbench isolated training worker")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    engine = read_json(args.run_dir / "run.json").get("engine")
    if engine == BASELINE_ENGINE:
        from .training_engine import train
    elif engine == "maskrcnn_resnet50_fpn":
        from .maskrcnn_engine import train
    elif engine.startswith("fasterrcnn_") or engine.startswith("deeplabv3_"):
        from .torchvision_engines import train
    else:
        raise ValueError(f"未知訓練引擎：{engine}")
    train(args.dataset.resolve(), args.run_dir.resolve(), args.model_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
