"""RT-DETR and YOLO26 Seg adapters for an isolated Ultralytics runtime."""
from __future__ import annotations

from .learning_rates import measured_rates

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from composer_core.geometry import encode_rle, shape_polygons
from .training_engine import atomic_json, read_json, _status, _stopping
from .yolo_compatibility import analyze_manifest, blocker_message, compatible_shape


ULTRALYTICS_ENGINES = {
    "rt_detr_r50": {"name": "RT-DETR · ResNet50", "task": "object_detection",
                    "architecture": "rtdetr-resnet50.yaml", "kind": "detect"},
    "yolo26n_seg": {"name": "YOLO26n Seg", "task": "instance_segmentation",
                    "architecture": "yolo26n-seg.yaml", "kind": "segment"},
    "yolo26s_seg": {"name": "YOLO26s Seg", "task": "instance_segmentation",
                    "architecture": "yolo26s-seg.yaml", "kind": "segment"},
}


def _label_path(root, split, asset):
    return root / "labels" / split / f"{asset['asset_id']}.txt"


def prepare_yolo_dataset(dataset_manifest: Path, output_dir: Path, task: str, config=None) -> Path:
    """Create a deterministic YOLO dataset from an immutable Workbench manifest.

    Segmentation may fill only policy-approved enclosed holes in this output
    copy. Source annotations and the immutable DatasetVersion are never edited.
    """
    dataset_manifest, output_dir = Path(dataset_manifest), Path(output_dir)
    manifest = read_json(dataset_manifest)
    if task not in {"object_detection", "instance_segmentation"}:
        raise ValueError("Ultralytics 資料轉接器不支援此任務")
    compatibility = analyze_manifest(manifest, config) if task == "instance_segmentation" else None
    if compatibility and not compatibility["compatible"]:
        raise ValueError(blocker_message(compatibility))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    class_ids = {name: index for index, name in enumerate(manifest["classes"])}
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for asset in manifest["assets"]:
        split, width, height = asset["split"], float(asset["width"]), float(asset["height"])
        source = (dataset_manifest.parent / asset["image_file"]).resolve()
        if sha256(source.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"固定資料版本圖片雜湊不符：{asset['name']}")
        destination = output_dir / "images" / split / f"{asset['asset_id']}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        lines = []
        for shape_index, (shape, obj) in enumerate(zip(asset.get("shapes", []), asset.get("objects", []))):
            class_id = class_ids.get(shape.get("label"))
            if class_id is None:
                raise ValueError(f"{asset['name']} 含有未知類別：{shape.get('label')}")
            if task == "object_detection":
                x, y, box_width, box_height = map(float, obj["bbox_xywh"])
                values = ((x + box_width / 2) / width, (y + box_height / 2) / height,
                          box_width / width, box_height / height)
                lines.append(str(class_id) + " " + " ".join(f"{max(0., min(1., value)):.8f}" for value in values))
            else:
                converted_shape, _repair, blocker = compatible_shape(asset, shape, shape_index, config)
                if blocker:
                    raise ValueError(blocker_message({"compatible": False, "blockers": [blocker],
                        "summary": {"blocked_assets": 1}}))
                polygons, diagnostics = shape_polygons(converted_shape, int(width), int(height), tolerance=0)
                if diagnostics.get("holes_omitted") or len(polygons) != 1 or diagnostics.get("pixel_iou", 1) < .999:
                    raise ValueError(f"{asset['name']} 的 {shape.get('label')} 是含孔洞或多區塊遮罩，無法無損轉成 YOLO Seg")
                points = polygons[0]
                if len(points) < 3:
                    raise ValueError(f"{asset['name']} 含有無效的分割多邊形")
                values = [coordinate for x, y in points for coordinate in
                          (max(0., min(1., float(x) / width)), max(0., min(1., float(y) / height)))]
                lines.append(str(class_id) + " " + " ".join(f"{value:.8f}" for value in values))
        _label_path(output_dir, split, asset).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    has_val = any(asset["split"] == "val" for asset in manifest["assets"])
    data = {"path": str(output_dir.resolve()), "train": "images/train",
            "val": "images/val" if has_val else "images/test",
            "test": "images/test", "names": {index: name for index, name in enumerate(manifest["classes"])}}
    path = output_dir / "data.yaml"
    # JSON is a valid YAML subset and avoids another dependency in the main app.
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if compatibility is not None:
        atomic_json(output_dir / "yolo-compatibility.json", compatibility)
    return path


def _finite_metric(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(number, 6) if math.isfinite(number) else None


def _numeric_metrics(trainer, kind):
    raw = getattr(trainer, "metrics", {}) or {}
    values = {str(key).lower(): number for key, value in raw.items()
              if (number := _finite_metric(value)) is not None}

    def get(name):
        for key, value in values.items():
            if key.rsplit("/", 1)[-1] == name:
                return value
        return None

    loss_items = getattr(trainer, "loss_items", None)
    try:
        loss = _finite_metric(loss_items.detach().sum().cpu())
    except Exception:
        loss = None
    if loss is None:
        loss = values.get("train/loss", values.get("loss"))
    prefix, suffix = ("box", "b") if kind == "detect" else ("mask", "m")
    measured = {"train/loss": loss, f"val/{prefix}_map50_95": get(f"map50-95({suffix})"),
                f"val/{prefix}_map50": get(f"map50({suffix})")}
    return {key: value for key, value in measured.items() if value is not None}


def _result_metrics(result, kind, split, image_count):
    metrics = getattr(result, "box" if kind == "detect" else "seg", None)
    prefix = "box" if kind == "detect" else "mask"
    measured = {f"{prefix}_{name}": number for name, attribute in
                (("map50_95", "map"), ("map50", "map50"), ("map75", "map75"))
                if (number := _finite_metric(getattr(metrics, attribute, None))) is not None}
    return {"split": split, "images": image_count, **measured}


def _artifact_manifest(run_dir, model_dir, run_id, checkpoint):
    rows = []
    for path in (run_dir / "metrics.jsonl", run_dir / "evaluation.json", model_dir / "model.json", checkpoint):
        raw = path.read_bytes(); rows.append({"path": path.name, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)})
    atomic_json(run_dir / "artifact-manifest.json", {"schema_version": 1, "run_id": run_id, "artifacts": rows})


def train(dataset_manifest: Path, run_dir: Path, model_dir: Path):
    dataset_manifest, run_dir, model_dir = map(Path, (dataset_manifest, run_dir, model_dir))
    manifest, run = read_json(dataset_manifest), read_json(run_dir / "run.json")
    definition = ULTRALYTICS_ENGINES[run["engine"]]
    try:
        os.environ.setdefault("YOLO_OFFLINE", "true")
        from ultralytics import RTDETR, YOLO
        data_yaml = prepare_yolo_dataset(dataset_manifest, run_dir / "dataset", definition["task"], run.get("config"))
        compatibility = (read_json(run_dir / "dataset" / "yolo-compatibility.json")
                         if definition["task"] == "instance_segmentation" else None)
        validation_split = "val" if any(asset["split"] == "val" for asset in manifest["assets"]) else "test"
        metrics_path = run_dir / "metrics.jsonl"; metrics_path.write_text("", encoding="utf-8")
        _status(run_dir, run, status="preparing", message=f"建立 {definition['name']} 資料轉接", progress=3,
                validation_split=validation_split)
        model = (RTDETR if definition["kind"] == "detect" else YOLO)(definition["architecture"])
        epochs = int(run["config"]["epochs"])
        def on_epoch(trainer):
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            row = {"epoch": epoch, **_numeric_metrics(trainer, definition["kind"]), **measured_rates(getattr(trainer, "optimizer", None))}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            _status(run_dir, run, status="running", message=f"{definition['name']} {epoch} / {epochs}", epoch=epoch,
                    progress=5 + round(epoch / epochs * 82), metrics=row)
            if _stopping(run_dir):
                trainer.stop = True
        model.add_callback("on_fit_epoch_end", on_epoch)
        requested = run["config"].get("device", "auto")
        device = "0" if requested == "cuda" else "cpu" if requested == "cpu" else None
        schedule = run["config"].get("scheduler", "linear")
        initial_lr = float(run["config"].get("learning_rate", .0005))
        model.train(data=str(data_yaml), epochs=epochs, imgsz=int(run["config"].get("image_size", 640)),
                             batch=int(run["config"].get("batch_size", 1)), device=device, workers=0,
                             lr0=initial_lr, cos_lr=schedule == "cosine",
                             lrf=1.0 if schedule == "fixed" else float(run["config"].get("min_learning_rate", initial_lr*.01))/initial_lr,
                             warmup_epochs=0 if schedule == "fixed" else int(run["config"].get("warmup_epochs", 0)),
                             warmup_bias_lr=0.0,
                             weight_decay=float(run["config"].get("weight_decay", .0001)),
                             optimizer=run["config"].get("optimizer", "AdamW"),
                             **({"momentum": float(run["config"].get("momentum", .9))}
                                if run["config"].get("optimizer") == "SGD" else {}),
                             project=str(run_dir / "ultralytics"), name="fit", exist_ok=True, pretrained=False,
                             plots=False, verbose=False, deterministic=False, seed=int(run["config"].get("seed", 42)))
        if _stopping(run_dir):
            return _status(run_dir, run, status="stopped", message="已安全停止", progress=None, completed_at=time.time())
        trainer = getattr(model, "trainer", None)
        best = Path(str(getattr(trainer, "best", "")))
        if not best.is_file():
            candidate = run_dir / "ultralytics" / "fit" / "weights" / "best.pt"
            best = candidate if candidate.is_file() else Path(str(getattr(trainer, "last", "")))
        if not best.is_file():
            raise RuntimeError("訓練完成但找不到 checkpoint")
        checkpoint = model_dir / "checkpoint.pt"; shutil.copy2(best, checkpoint)
        validation_raw = model.val(data=str(data_yaml), split="val", device=device, plots=False, verbose=False)
        val_count = sum(asset["split"] == validation_split for asset in manifest["assets"])
        validation = _result_metrics(validation_raw, definition["kind"], validation_split, val_count)
        test_count = sum(asset["split"] == "test" for asset in manifest["assets"])
        test = (_result_metrics(model.val(data=str(data_yaml), split="test", device=device, plots=False, verbose=False),
                                definition["kind"], "test", test_count) if test_count else validation)
        record = {"schema_version": 1, "engine": run["engine"], "engine_name": definition["name"],
                  "task": definition["task"], "model_version_id": run["model_version_id"], "run_id": run["run_id"],
                  "dataset_version_id": manifest["dataset_version_id"], "classes": manifest["classes"],
                  "image_size": int(run["config"].get("image_size", 640)), "score_threshold": .5,
                  "validation": validation, "test": test, "created_at": time.time(), "checkpoint": "checkpoint.pt"}
        if compatibility is not None:
            record["yolo_compatibility"] = compatibility
        atomic_json(model_dir / "model.json", record)
        evaluation = {"validation": validation, "test": test}
        if compatibility is not None:
            evaluation["yolo_compatibility"] = compatibility
        atomic_json(run_dir / "evaluation.json", evaluation)
        _artifact_manifest(run_dir, model_dir, run["run_id"], checkpoint)
        return _status(run_dir, run, status="completed", message=f"{definition['name']} 訓練與評估完成", progress=100,
                       completed_at=time.time(), evaluation=evaluation)
    except Exception as exc:
        _status(run_dir, run, status="failed", message=str(exc), error=str(exc), progress=None, completed_at=time.time())
        raise


class Predictor:
    def __init__(self, record, model_dir, requested_device="auto"):
        os.environ.setdefault("YOLO_OFFLINE", "true")
        from ultralytics import RTDETR, YOLO
        self.record = record; self.definition = ULTRALYTICS_ENGINES[record["engine"]]
        constructor = RTDETR if self.definition["kind"] == "detect" else YOLO
        self.model = constructor(str(Path(model_dir) / record["checkpoint"]))
        self.device = "0" if requested_device == "cuda" else "cpu" if requested_device == "cpu" else None

    def predict(self, rgb, width, height):
        import cv2
        result = self.model.predict(source=np.asarray(rgb, dtype=np.uint8), imgsz=int(self.record.get("image_size", 640)),
                                    device=self.device, verbose=False)[0]
        shapes, boxes = [], getattr(result, "boxes", None)
        if boxes is None:
            return shapes
        labels = boxes.cls.cpu().tolist(); scores = boxes.conf.cpu().tolist()
        if self.definition["kind"] == "detect":
            for label_id, score, box in zip(labels, scores, boxes.xyxy.cpu().tolist()):
                if score < float(self.record.get("score_threshold", .5)) or not 0 <= int(label_id) < len(self.record["classes"]): continue
                x1, y1, x2, y2 = map(float, box)
                shapes.append({"id": uuid.uuid4().hex, "type": "rectangle", "label": self.record["classes"][int(label_id)],
                               "x": max(0., x1), "y": max(0., y1), "width": max(1., min(width, x2)-max(0., x1)),
                               "height": max(1., min(height, y2)-max(0., y1)),
                               "metadata": {"source": "model_candidate", "engine": self.record["engine"],
                                            "model_version_id": self.record["model_version_id"], "confidence": round(float(score), 6)}})
        else:
            masks = getattr(result, "masks", None)
            if masks is None: return shapes
            for label_id, score, raw in zip(labels, scores, masks.data.cpu().numpy()):
                if score < float(self.record.get("score_threshold", .5)) or not 0 <= int(label_id) < len(self.record["classes"]): continue
                bitmap = cv2.resize(raw.astype(np.float32), (int(width), int(height)), interpolation=cv2.INTER_NEAREST) >= .5
                if not bitmap.any(): continue
                ys, xs = np.nonzero(bitmap)
                shapes.append({"id": uuid.uuid4().hex, "type": "mask", "label": self.record["classes"][int(label_id)],
                               "x": 0, "y": 0, "width": width, "height": height, "counts": encode_rle(bitmap),
                               "metadata": {"source": "model_candidate", "engine": self.record["engine"],
                                            "model_version_id": self.record["model_version_id"], "confidence": round(float(score), 6),
                                            "bounds": [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]}})
        return shapes
