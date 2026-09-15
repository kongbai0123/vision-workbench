"""Recompute historical Workbench scores with evaluation schema v2.

This is an explicit, potentially GPU-heavy backtest and is intentionally kept
out of the default unit-test discovery.  It never modifies datasets,
annotations, checkpoints, or legacy records.  ``--install`` may add a separate
``evaluation.v2.json`` reassessment beside a historical run/model.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from composer_core.geometry import decode_rle
from workbench.evaluation_metrics import PixelMetrics, evaluation_protocol
from workbench.maskrcnn_engine import NativeMaskDataset, _evaluate as evaluate_maskrcnn, _model
from workbench.training import dataset_readiness
from workbench.training_engine import atomic_json, class_masks
from workbench.ultralytics_engine import Predictor


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def backtest_maskrcnn(root, project_id, device):
    import torch

    dataset_root = root / "data" / "datasets" / project_id / "D001"
    model_root = root / "data" / "models" / project_id / "M001"
    manifest = _read(dataset_root / "manifest.json")
    record = _read(model_root / "model.json")
    model, _ = _model(len(record["classes"]), record.get("image_size"))
    checkpoint = torch.load(model_root / record["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return {"run_id": "R001", "model_version_id": "M001", "engine": record["engine"],
            "legacy": {"validation": record.get("validation"), "test": record.get("test")},
            "schema_v2": {split: evaluate_maskrcnn(
                model, NativeMaskDataset(manifest, dataset_root, split, torch), device, torch,
                threshold=float(record.get("score_threshold", .5))) for split in ("val", "test")}}


def backtest_yolo(root, project_id, requested_device):
    dataset_root = root / "data" / "datasets" / project_id / "D005"
    model_root = root / "data" / "models" / project_id / "M005"
    manifest = _read(dataset_root / "manifest.json")
    record = _read(model_root / "model.json")
    predictor = Predictor(record, model_root, requested_device=requested_device)
    results = {}
    for split in ("val", "test"):
        assets = [asset for asset in manifest["assets"] if asset["split"] == split]
        metrics = PixelMetrics(manifest["classes"])
        per_image = []
        for asset in assets:
            with Image.open(dataset_root / asset["image_file"]) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            expected = class_masks(asset, manifest["classes"])
            predicted = {label: np.zeros(rgb.shape[:2], dtype=bool) for label in manifest["classes"]}
            shapes = predictor.predict(rgb, asset["width"], asset["height"])
            for shape in shapes:
                if shape["label"] in predicted:
                    predicted[shape["label"]] |= decode_rle(shape["counts"], asset["width"], asset["height"]) > 0
            for label in manifest["classes"]:
                metrics.update(label, predicted[label], expected[label])
            labels = sorted({shape["label"] for shape in asset.get("shapes", [])})
            label = labels[0] if len(labels) == 1 else None
            if label:
                intersection = int(np.count_nonzero(predicted[label] & expected[label]))
                union = int(np.count_nonzero(predicted[label] | expected[label]))
                per_image.append({"asset_id": asset["asset_id"], "name": asset["name"], "label": label,
                                  "iou": round(intersection / union, 6) if union else None,
                                  "candidate_count": len(shapes)})
        summary = metrics.summary(split, len(assets))
        summary["score_threshold"] = float(record.get("score_threshold", .5))
        summary["per_image"] = sorted(per_image, key=lambda row: (row["iou"] is None, row["iou"] or 0))
        results[split] = summary
    return {"run_id": "R005", "model_version_id": "M005", "engine": record["engine"],
            "legacy_ultralytics_map": {"validation": record.get("validation"), "test": record.get("test")},
            "deployment_predictor_schema_v2": results}


def _source_hashes(paths):
    return {str(path): sha256(Path(path).read_bytes()).hexdigest() for path in paths}


def install_reassessments(root, project_id, report):
    created_at = time.time()
    run_root = root / "data" / "runs" / project_id
    model_root = root / "data" / "models" / project_id
    written = []

    def persist(run_id, model_id, value):
        for directory in (run_root / run_id, model_root / model_id):
            if directory.is_dir():
                target = directory / "evaluation.v2.json"
                atomic_json(target, value)
                written.append(str(target))

    r001 = report["r001"]
    d001 = _read(root / "data" / "datasets" / project_id / "D001" / "manifest.json")
    r001_sources = [run_root / "R001" / "evaluation.json", model_root / "M001" / "model.json",
                    model_root / "M001" / "checkpoint.pt"]
    persist("R001", "M001", {
        "schema_version": 2, "created_at": created_at, "valid": True,
        "reason": "修正逐圖空白遮罩計為 1.0 所造成的 IoU 膨脹",
        "source_sha256": _source_hashes(r001_sources),
        "protocol": {**evaluation_protocol(checkpoint="historical_final_epoch", has_test=True, manifest=d001),
                     "historical_reassessment": True},
        "validation": r001["schema_v2"]["val"], "test": r001["schema_v2"]["test"],
    })

    d004_manifest = root / "data" / "datasets" / project_id / "D004" / "manifest.json"
    r004_sources = [d004_manifest, run_root / "R004" / "evaluation.json", model_root / "M004" / "model.json"]
    persist("R004", "M004", {
        "schema_version": 2, "created_at": created_at, "valid": False,
        "reason": "D004 的 Train、Validation、Test 類別互斥，歷史分數不具模型評估效力",
        "source_sha256": _source_hashes([path for path in r004_sources if path.is_file()]),
        "protocol": {**evaluation_protocol(checkpoint="historical_invalid_split", has_test=True,
                                             manifest=_read(d004_manifest)),
                     "historical_reassessment": True},
        "validation": None, "test": None, "readiness": report["r004_readiness"],
    })

    r005 = report["r005"]
    d005 = _read(root / "data" / "datasets" / project_id / "D005" / "manifest.json")
    merged = {}
    for name, key in (("validation", "val"), ("test", "test")):
        native = r005["deployment_predictor_schema_v2"][key]
        merged[name] = {**(r005["legacy_ultralytics_map"][name] or {}),
                        "deployment_score_threshold": native["score_threshold"],
                        "mean_iou": native["mean_iou"], "micro_iou": native["micro_iou"],
                        "per_class_iou": native["per_class_iou"], "per_class": native["per_class"],
                        "deployment_per_image": native["per_image"],
                        "metric_scope": "v2.7.6_deployment_predictor_original_pixels"}
    r005_sources = [run_root / "R005" / "evaluation.json", model_root / "M005" / "model.json",
                    model_root / "M005" / "checkpoint.pt"]
    persist("R005", "M005", {
        "schema_version": 2, "created_at": created_at, "valid": True,
        "reason": "補入 v2.7.6 實際部署 Predictor 在原圖像素的 IoU、FP 與 FN",
        "source_sha256": _source_hashes(r005_sources),
        "protocol": {**evaluation_protocol(checkpoint="best_validation", has_test=True, manifest=d005),
                     "historical_reassessment": True},
        **merged,
    })
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--project-id", required=True,
                        help="local project identifier containing D001/M001, D004/M004 and D005/M005")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--install", action="store_true",
                        help="write versioned evaluation.v2.json overlays beside historical records")
    args = parser.parse_args()
    root = args.root.resolve()
    d004 = _read(root / "data" / "datasets" / args.project_id / "D004" / "manifest.json")
    report = {"schema_version": 2,
              "purpose": "historical_evaluation_backtest",
              "source_data_unchanged": True,
              "r004_readiness": dataset_readiness(d004),
              "r001": backtest_maskrcnn(root, args.project_id, args.device),
              "r005": backtest_yolo(root, args.project_id, args.device)}
    if args.install:
        report["installed_reassessments"] = install_reassessments(root, args.project_id, report)
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
