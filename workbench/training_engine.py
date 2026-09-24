"""Small native-mask segmentation engine used by the first deliverable.

The engine intentionally has no dependency on the Workbench database.  It reads
an immutable dataset manifest, writes one run directory, and can therefore run
in a separate Python process or in contract tests.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from composer_core.geometry import decode_rle, encode_rle
from .evaluation_metrics import PixelMetrics, evaluation_protocol


ENGINE_KEY = "pixel_prototype_v1"
ENGINE_NAME = "像素原型分割（內建基準）"


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == 'run.json' and 'status' in value:
        from .run_events import append_state
        from .file_lock import exclusive_file_lock
        with exclusive_file_lock(path.with_name('.events.lock')):
            _atomic_json_file(path, append_state(path, value))
        return
    _atomic_json_file(path, value)


def _atomic_json_file(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        # Windows virus scanners and concurrent readers can hold the previous
        # JSON file for a few milliseconds.  Keep the atomic replace contract
        # while tolerating that transient lock.
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.015 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path):
    for attempt in range(8):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            # Atomic replacement can briefly deny new readers on Windows.
            if attempt == 7:
                raise
            time.sleep(.015 * (attempt + 1))


def annotation_mask(shape: dict, width: int, height: int) -> np.ndarray:
    kind = shape.get("type")
    if kind == "mask":
        return decode_rle(shape.get("counts"), width, height) > 0
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if kind == "rectangle":
        x, y = float(shape["x"]), float(shape["y"])
        draw.rectangle((x, y, x + float(shape["width"]), y + float(shape["height"])), fill=1)
    elif kind in {"polygon", "obb"}:
        draw.polygon([(float(x), float(y)) for x, y in shape.get("points", [])], fill=1)
    else:
        raise ValueError(f"{kind} 無法用於像素分割訓練")
    return np.asarray(mask, dtype=np.uint8) > 0


def class_masks(asset: dict, classes: list[str]) -> dict[str, np.ndarray]:
    width, height = int(asset["width"]), int(asset["height"])
    masks = {label: np.zeros((height, width), dtype=bool) for label in classes}
    for shape in asset.get("shapes", []):
        label = shape.get("label")
        if label in masks:
            masks[label] |= annotation_mask(shape, width, height)
    return masks


def load_rgb(dataset_dir: Path, asset: dict) -> np.ndarray:
    path = (dataset_dir / asset["image_file"]).resolve()
    if not path.is_relative_to(dataset_dir.resolve()) or not path.is_file():
        raise FileNotFoundError(f"找不到固定資料版本圖片：{asset.get('name')}")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != asset["sha256"]:
        raise ValueError(f"固定資料版本圖片雜湊不符：{asset.get('name')}")
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.float32)


def _stats(values: np.ndarray) -> dict:
    if not len(values):
        return {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "pixels": 0}
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), 8.0)
    return {"mean": mean.round(6).tolist(), "std": std.round(6).tolist(), "pixels": int(len(values))}


def _distance(rgb: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return np.sqrt(np.mean(((rgb - mean) / std) ** 2, axis=2))


def predict_class_masks(rgb: np.ndarray, model: dict) -> dict[str, np.ndarray]:
    background = _distance(rgb, model["background"])
    output = {}
    for label in model["classes"]:
        if model['class_stats'].get(label) is None:
            output[label] = np.zeros(rgb.shape[:2], dtype=bool)
            continue
        foreground = _distance(rgb, model["class_stats"][label])
        threshold = float(model["thresholds"][label])
        output[label] = (foreground <= threshold) & (foreground < background)
    return output


def _iou(predicted: np.ndarray, expected: np.ndarray) -> float:
    union = int(np.count_nonzero(predicted | expected))
    return int(np.count_nonzero(predicted & expected)) / union if union else float("nan")


def _evaluate(manifest: dict, dataset_dir: Path, model: dict, split: str) -> dict:
    metrics = PixelMetrics(manifest["classes"])
    images = 0
    for asset in manifest["assets"]:
        if asset["split"] != split:
            continue
        images += 1
        rgb = load_rgb(dataset_dir, asset)
        expected = class_masks(asset, manifest["classes"])
        predicted = predict_class_masks(rgb, model)
        for label in manifest["classes"]:
            metrics.update(label, predicted[label], expected[label])
    return metrics.summary(split, images)


def _status(run_dir: Path, run: dict, **changes) -> dict:
    from .time_estimation import record_timing
    record_timing(run_dir, run, changes)
    run.update(changes, updated_at=time.time())
    atomic_json(run_dir / "run.json", run)
    return run


def _stopping(run_dir: Path) -> bool:
    return (run_dir / 'control' / 'stop.requested').exists() or (run_dir / 'stop.requested').exists()


stop_requested = _stopping


def train(dataset_manifest: Path, run_dir: Path, model_dir: Path) -> dict:
    """Train and evaluate an RGB foreground/background prototype model."""
    dataset_manifest, run_dir, model_dir = map(Path, (dataset_manifest, run_dir, model_dir))
    manifest = read_json(dataset_manifest)
    from .split_quality import loose_split_applies
    loose = loose_split_applies(manifest.get('split_plan'), manifest.get('assets', []))
    dataset_dir = dataset_manifest.parent
    run = read_json(run_dir / "run.json")
    from .time_estimation import timed_phase
    epochs = max(1, min(200, int(run["config"].get("epochs", 24))))
    try:
        _status(run_dir, run, status="preparing", message="驗證固定資料版本", progress=2)
        classes = list(manifest.get("classes") or [])
        if not classes:
            raise ValueError("訓練資料沒有類別")
        samples = {label: [] for label in classes}
        backgrounds = []
        train_assets = [asset for asset in manifest["assets"] if asset["split"] == "train"]
        if not train_assets:
            raise ValueError("訓練資料版本沒有 Train 圖片")
        _status(run_dir, run, phase='read_images', timing_work={'completed': 0, 'total': len(train_assets)})
        for index, asset in enumerate(train_assets):
            if _stopping(run_dir):
                return _status(run_dir, run, status="stopped", message="已安全停止", progress=None)
            rgb = load_rgb(dataset_dir, asset)
            masks = class_masks(asset, classes)
            occupied = np.zeros(rgb.shape[:2], dtype=bool)
            for label, mask in masks.items():
                if np.any(mask):
                    samples[label].append(rgb[mask])
                    occupied |= mask
            if np.any(~occupied):
                backgrounds.append(rgb[~occupied])
            _status(run_dir, run, status="preparing", message=f"讀取訓練圖片 {index + 1} / {len(train_assets)}",
                    progress=2 + round((index + 1) / len(train_assets) * 20), timing_work={'completed': index + 1, 'total': len(train_assets)})
        empty = [label for label, rows in samples.items() if not rows]
        if empty and not loose:
            raise ValueError(f"Train 缺少類別標註：{'、'.join(empty)}")
        if not backgrounds:
            raise ValueError("Train 圖片沒有可學習的背景像素")
        model = {"schema_version": 1, "engine": ENGINE_KEY, "engine_name": ENGINE_NAME,
                 "classes": classes, "background": _stats(np.concatenate(backgrounds)),
                 "class_stats": {label: _stats(np.concatenate(rows)) if rows else None for label, rows in samples.items()},
                 "unlearned_classes": empty,
                 "thresholds": {label: 1.0 for label in classes}, "dataset_version_id": manifest["dataset_version_id"]}
        validation_split = "val"
        if not any(a["split"] == "val" for a in manifest["assets"]):
            raise ValueError("Validation 沒有圖片；Test 不可用於調整模型門檻")
        candidates = np.linspace(float(run["config"].get("threshold_min", .35)),
                                 float(run["config"].get("threshold_max", 4.0)), epochs)
        metrics_path = run_dir / "metrics.jsonl"
        best = {label: (1.0, -1.0) for label in classes}
        metrics_path.write_text("", encoding="utf-8")
        _status(run_dir, run, status='running', phase='threshold', timing_work={'completed': 0, 'total': epochs})
        for epoch, threshold in enumerate(candidates, 1):
            if _stopping(run_dir):
                return _status(run_dir, run, status="stopped", message="已安全停止", progress=None)
            for label in classes:
                model["thresholds"][label] = float(threshold)
            evaluation = _evaluate(manifest, dataset_dir, model, validation_split)
            for label, value in evaluation["per_class_iou"].items():
                if loose and not evaluation['per_class'][label]['ground_truth_pixels']:
                    continue
                if value is not None and value > best[label][1]:
                    best[label] = (float(threshold), float(value))
            row = {"epoch": epoch, "threshold": float(threshold), "val/mean_iou": evaluation["mean_iou"]}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            _status(run_dir, run, status="running", message=f"調整像素分類門檻 {epoch} / {epochs}", epoch=epoch,
                    progress=22 + round(epoch / epochs * 65), metrics=row, timing_work={'completed': epoch, 'total': epochs})
        missing_validation = [label for label, (_threshold, score) in best.items() if score < 0]
        if missing_validation and not loose:
            raise ValueError(f"Validation 缺少可評估像素的類別：{'、'.join(missing_validation)}")
        model["thresholds"] = {label: float(best[label][0]) for label in classes}
        model['uncalibrated_classes'] = missing_validation
        with timed_phase(run_dir, run, 'final_validation'):
            validation = _evaluate(manifest, dataset_dir, model, validation_split)
        has_test = any(a["split"] == "test" for a in manifest["assets"])
        test = None
        if has_test:
            with timed_phase(run_dir, run, 'test'):
                test = _evaluate(manifest, dataset_dir, model, 'test')
        protocol = evaluation_protocol(checkpoint="validation_selected_thresholds", has_test=has_test,
                                       manifest=manifest)
        with timed_phase(run_dir, run, 'saving'):
            model.update(validation=validation, test=test, evaluation_protocol=protocol, created_at=time.time(), model_version_id=run["model_version_id"],
                          run_id=run["run_id"])
            atomic_json(model_dir / "model.json", model)
            atomic_json(run_dir / "evaluation.json", {"schema_version": 2, "protocol": protocol,
                                                       "validation": validation, "test": test})
            artifact_files = [run_dir / "run.json", run_dir / "metrics.jsonl", run_dir / "evaluation.json", model_dir / "model.json"]
            artifacts = []
            for path in artifact_files[1:]:
                raw = path.read_bytes()
                artifacts.append({"path": path.relative_to(run_dir.parent.parent.parent).as_posix()
                                  if path.is_relative_to(run_dir.parent.parent.parent) else path.name,
                                  "sha256": sha256(raw).hexdigest(), "bytes": len(raw)})
            atomic_json(run_dir / "artifact-manifest.json", {"schema_version": 1, "run_id": run["run_id"],
                        "dataset_version_id": manifest["dataset_version_id"], "model_version_id": run["model_version_id"],
                        "artifacts": artifacts})
        return _status(run_dir, run, status="completed", message="訓練與評估完成", progress=100,
                       completed_at=time.time(), evaluation={"schema_version": 2, "protocol": protocol,
                                                              "validation": validation, "test": test})
    except Exception as exc:
        _status(run_dir, run, status="failed", message=str(exc), error=str(exc), progress=None, completed_at=time.time())
        raise


def predict(rgb: np.ndarray, model: dict, width: int, height: int, minimum_area=8) -> list[dict]:
    """Return one native RLE candidate per connected foreground component."""
    if rgb.shape[:2] != (height, width):
        raise ValueError("推論圖片尺寸不符")
    outputs = predict_class_masks(rgb.astype(np.float32), model)
    shapes = []
    import uuid
    for label, mask in outputs.items():
        for instance in _connected_masks(mask):
            area = int(np.count_nonzero(instance))
            if area < minimum_area:
                continue
            ys, xs = np.nonzero(instance)
            shapes.append({"id": uuid.uuid4().hex, "type": "mask", "label": label,
                           "x": 0, "y": 0, "width": width, "height": height,
                           "counts": encode_rle(instance), "metadata": {"source": "model_candidate",
                           "engine": model["engine"], "model_version_id": model["model_version_id"],
                           "foreground_pixels": area,
                           "bounds": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}})
    return shapes


def _connected_masks(mask):
    """Yield 8-connected masks; OpenCV is optional in the training runtime."""
    try:
        import cv2
        count, labels = cv2.connectedComponents(mask.astype(np.uint8))
        for component in range(1, count):
            yield labels == component
        return
    except (ImportError, OSError):
        pass
    remaining = mask.astype(bool).copy()
    height, width = remaining.shape
    while np.any(remaining):
        first = np.argwhere(remaining)[0]; stack = [(int(first[0]), int(first[1]))]
        component = np.zeros_like(remaining)
        remaining[stack[0]] = False
        while stack:
            y, x = stack.pop(); component[y, x] = True
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if remaining[ny, nx]: remaining[ny, nx] = False; stack.append((ny, nx))
        yield component
