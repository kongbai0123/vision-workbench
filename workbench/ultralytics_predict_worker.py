"""Isolated inference entry point for RT-DETR and YOLO26 Detect/Seg."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .training_engine import atomic_json, read_json
from .ultralytics_engine import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); request = read_json(Path(args.request)); model_path = Path(request["model_path"])
    record = read_json(model_path); predictor = Predictor(record, model_path.parent, request.get("device", "auto"))
    rows = []
    for asset in request["assets"]:
        with Image.open(asset["image_path"]) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        rows.append({"asset_id": asset["asset_id"], "shapes": predictor.predict(rgb, asset["width"], asset["height"])})
    atomic_json(Path(args.output), {"assets": rows})


if __name__ == "__main__":
    main()
