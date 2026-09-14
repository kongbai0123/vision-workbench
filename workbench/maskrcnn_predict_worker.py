"""Isolated Mask R-CNN inference process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .maskrcnn_engine import Predictor
from .training_engine import atomic_json, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vision Workbench Mask R-CNN prediction worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request = read_json(args.request)
    model_path = Path(request["model_path"])
    predictor = Predictor(read_json(model_path), model_path.parent, request.get("device", "auto"))
    results = []
    for asset in request["assets"]:
        with Image.open(asset["image_path"]) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
        results.append({"asset_id": asset["asset_id"], "shapes": predictor.predict(rgb, asset["width"], asset["height"])})
    atomic_json(args.output, {"assets": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
