"""RT-DETR and YOLO26 detection/segmentation adapters."""
from __future__ import annotations

from .learning_rates import measured_rates
from .evaluation_metrics import EVALUATION_SCHEMA_VERSION, evaluation_protocol, training_only_protocol

from collections.abc import Mapping
import csv
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
from .augmentation import normalize_augmentation, yolo_augmentation_args


ULTRALYTICS_ENGINES = {
    "rt_detr_r50": {"name": "RT-DETR · ResNet50", "task": "object_detection",
                    "architecture": "rtdetr-resnet50.yaml", "kind": "detect"},
    "yolo26n_seg": {"name": "YOLO26n Seg", "task": "instance_segmentation",
                    "architecture": "yolo26n-seg.yaml", "kind": "segment"},
    "yolo26s_seg": {"name": "YOLO26s Seg", "task": "instance_segmentation",
                    "architecture": "yolo26s-seg.yaml", "kind": "segment"},
    "yolo26n_detect": {"name": "YOLO26n Detect", "task": "object_detection",
                       "architecture": "yolo26n.yaml", "kind": "detect"},
    "yolo26s_detect": {"name": "YOLO26s Detect", "task": "object_detection",
                       "architecture": "yolo26s.yaml", "kind": "detect"},
}


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
    augmentation = normalize_augmentation((config or {}).get("augmentation"))
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for asset in manifest["assets"]:
        split, width, height = asset["split"], float(asset["width"]), float(asset["height"])
        source = (dataset_manifest.parent / asset["image_file"]).resolve()
        if sha256(source.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"固定資料版本圖片雜湊不符：{asset['name']}")
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
        event_count = 1 + augmentation["expansion_count"] if split == "train" else 1
        label_text = "\n".join(lines) + ("\n" if lines else "")
        for event in range(event_count):
            event_id = asset["asset_id"] if event == 0 else f"{asset['asset_id']}__aug{event:03d}"
            destination = output_dir / "images" / split / f"{event_id}{source.suffix.lower()}"
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            (output_dir / "labels" / split / f"{event_id}.txt").write_text(label_text, encoding="utf-8")
    data = {"path": str(output_dir.resolve()), "train": "images/train",
            "val": "images/val",
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


def _canonical_ultralytics_values(raw):
    """Map Ultralytics CSV/callback names to one stable chart vocabulary."""
    measured = {}
    for raw_key, raw_value in (raw or {}).items():
        value = _finite_metric(raw_value)
        if value is None:
            continue
        key = str(raw_key).strip().lower()
        if key.startswith(('train/', 'val/')) and key.endswith('_loss'):
            measured[key] = value
            continue
        for source, target in (('precision', 'precision'), ('recall', 'recall'),
                               ('map50-95', 'map50_95'), ('map50', 'map50')):
            if key.endswith(f'{source}(b)'):
                measured[f'val/box_{target}'] = value
                break
            if key.endswith(f'{source}(m)'):
                measured[f'val/mask_{target}'] = value
                break
        if key == 'lr/pg0':
            measured['train/learning_rate'] = value
        elif key.startswith('lr/pg') and key[5:].isdigit():
            measured[f'lr/group_{int(key[5:])}'] = value
        elif key in {'train/loss', 'val/loss'}:
            measured[key] = value
    for split in ('train', 'val'):
        components = [value for key, value in measured.items()
                      if key.startswith(split + '/') and key.endswith('_loss')]
        if components:
            measured[split + '/loss'] = _finite_metric(sum(components))
    return measured


def read_ultralytics_results(path):
    """Read native results.csv without exposing duplicate package aliases."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for raw in csv.DictReader(handle):
            epoch = _finite_metric(raw.get('epoch'))
            if epoch is None or epoch < 1 or epoch != int(epoch):
                continue
            rows.append({'epoch': int(epoch), **_canonical_ultralytics_values(raw)})
    return rows


def _numeric_metrics(trainer, kind):
    raw = getattr(trainer, "metrics", {}) or {}
    values = {str(key).lower(): value for key, value in raw.items()}

    # tloss is the running mean for the whole epoch. loss_items contains only
    # the final batch and is not a substitute for an epoch-average loss.
    average = getattr(trainer, "tloss", None)
    components = {}
    if average is not None:
        label = getattr(trainer, "label_loss_items", None)
        if callable(label):
            try:
                labeled = label(average, prefix="train")
                if isinstance(labeled, Mapping):
                    components = dict(labeled)
            except (AttributeError, TypeError, ValueError):
                pass
        if not components and isinstance(average, Mapping):
            components = {f"train/{key}": value for key, value in average.items()}
        elif not components:
            if hasattr(average, "detach"):
                average = average.detach().cpu()
            try:
                entries = average.reshape(-1).tolist() if hasattr(average, "reshape") else list(average)
            except TypeError:
                entries = [average]
            names = getattr(trainer, "loss_names", ())
            components = {f"train/{names[i] if i < len(names) else f'loss_component_{i + 1}'}": value
                          for i, value in enumerate(entries)}
    finite_components = {key: number for key, value in components.items()
                         if (number := _finite_metric(value)) is not None}
    measured = _canonical_ultralytics_values(values)
    if kind == 'detect':
        measured = {key: value for key, value in measured.items() if not key.startswith('val/mask_')}
    measured.update(finite_components)
    if components and len(finite_components) == len(components):
        measured['train/loss'] = _finite_metric(sum(finite_components.values()))
    elif 'train/loss' not in measured:
        fallback = _finite_metric(values.get('loss'))
        if fallback is not None:
            measured['train/loss'] = fallback
    return measured


class _RunMetricsRecorder:
    """Record real training epochs and successful optimizer updates only."""

    def __init__(self, run_dir, run, definition):
        self.run_dir, self.run, self.definition = Path(run_dir), run, definition
        self.path = self.run_dir / "metrics.jsonl"
        self.epochs = int(run["config"]["epochs"])
        self.pending_epoch = None
        self.completed_epochs = set()
        self.optimizer = None
        self.hook = None
        self.optimizer_steps = 0
        self.epoch_start_steps = 0
        self.optimizer_attempts = 0
        self.epoch_start_attempts = 0
        self.skipped_updates = 0
        self.epoch_start_skipped = 0
        self._wrapped_trainer = None
        self.last_step_rates = {}
        self.execution = {}

    def close(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def _after_optimizer_step(self, optimizer, _args, _kwargs):
        # GradScaler does not call optimizer.step when it skips an update for
        # non-finite gradients, so this hook does not count those attempts.
        self.optimizer_steps += 1
        self.last_step_rates = measured_rates(optimizer)

    def on_epoch_start(self, trainer):
        epoch = int(trainer.epoch) + 1
        if self.pending_epoch != epoch:
            self.epoch_start_steps = self.optimizer_steps
            self.epoch_start_attempts = self.optimizer_attempts
            self.epoch_start_skipped = self.skipped_updates
            self.last_step_rates = {}
        self.pending_epoch = epoch
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not self.optimizer:
            self.close()
            self.optimizer = optimizer
            register = getattr(optimizer, "register_step_post_hook", None)
            if callable(register):
                self.hook = register(self._after_optimizer_step)
        original = getattr(trainer, 'optimizer_step', None)
        if self.hook is not None and callable(original) and trainer is not self._wrapped_trainer:
            def measured_step(*args, **kwargs):
                before = self.optimizer_steps
                self.optimizer_attempts += 1
                result = original(*args, **kwargs)
                if self.optimizer_steps == before:
                    self.skipped_updates += 1
                return result
            trainer.optimizer_step = measured_step
            self._wrapped_trainer = trainer
        config = self.run["config"]
        batch = int(getattr(trainer, "batch_size", config.get("batch_size", 1)))
        accumulation = int(getattr(trainer, "accumulate", config.get("gradient_accumulation", 1)))
        self.execution = {"batch_size": batch, "gradient_accumulation": accumulation,
                          "effective_batch_size": batch * accumulation,
                          "optimizer_step_measurement": "post_step_hook" if self.hook is not None else "unavailable"}

    def on_epoch_end(self, trainer):
        epoch = int(trainer.epoch) + 1
        # final_eval also emits on_fit_epoch_end, including after early stopping.
        # Only an epoch armed by on_train_epoch_start can enter the time series.
        if self.pending_epoch != epoch or epoch in self.completed_epochs:
            return
        self.pending_epoch = None
        self.completed_epochs.add(epoch)
        row = {"epoch": epoch, **_numeric_metrics(trainer, self.definition["kind"])}
        if self.hook is not None:
            row.update(self.last_step_rates)
            row.update({"train/optimizer_steps": self.optimizer_steps,
                        "train/optimizer_steps_epoch": self.optimizer_steps - self.epoch_start_steps,
                        "train/optimizer_attempts_epoch": self.optimizer_attempts - self.epoch_start_attempts,
                        "train/optimizer_skipped_epoch": self.skipped_updates - self.epoch_start_skipped})
            self.execution.update(optimizer_steps=self.optimizer_steps,
                                  optimizer_attempts=self.optimizer_attempts,
                                  optimizer_skipped_updates=self.skipped_updates)
            self.execution["learning_rate_source"] = "last_successful_optimizer_step"
        else:
            row.update(measured_rates(getattr(trainer, "optimizer", None)))
            self.execution["learning_rate_source"] = "optimizer_groups_at_epoch_end"
        accumulation = int(getattr(trainer, "accumulate", self.execution["gradient_accumulation"]))
        self.execution.update(gradient_accumulation=accumulation,
                              effective_batch_size=self.execution["batch_size"] * accumulation)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        _status(self.run_dir, self.run, status="running", message=f"{self.definition['name']} {epoch} / {self.epochs}",
                epoch=epoch, progress=5 + round(epoch / self.epochs * 82), metrics=row,
                execution=dict(self.execution))
        if _stopping(self.run_dir):
            trainer.stop = True

    def on_batch_end(self, trainer):
        """Expose live mini-batch progress without calling it an optimizer step."""
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        batch_index = int(getattr(trainer, "batch_i", -1)) + 1
        loader = getattr(trainer, "train_loader", None)
        try:
            batches = len(loader)
        except (TypeError, AttributeError):
            batches = 0
        if batch_index < 1 or batches < 1:
            return
        progress = 5 + round((((epoch - 1) + batch_index / batches) / self.epochs) * 82)
        execution = {**self.execution, "optimizer_steps": self.optimizer_steps,
                     "optimizer_attempts": self.optimizer_attempts,
                     "optimizer_skipped_updates": self.skipped_updates}
        _status(self.run_dir, self.run, status="running", phase="training",
                message=f"{self.definition['name']} · Epoch {epoch}/{self.epochs} · Batch {batch_index}/{batches}",
                epoch=epoch, batch=batch_index, batches_per_epoch=batches,
                progress=progress, execution=execution)
        if _stopping(self.run_dir):
            trainer.stop = True


def _initialization_source(definition, config):
    mode = config.get("initialization", "pretrained" if definition["architecture"].startswith("yolo26") else "scratch")
    if mode not in {"pretrained", "scratch"}:
        raise ValueError("模型初始化必須選擇 pretrained 或 scratch")
    source = definition["architecture"].replace(".yaml", ".pt") if mode == "pretrained" else definition["architecture"]
    return mode, source


def _initialization_record(model, mode, source):
    record = {"mode": mode, "source": source}
    candidate = getattr(model, "ckpt_path", None)
    if mode == "pretrained" and candidate and Path(candidate).is_file():
        path = Path(candidate).resolve()
        record.update(weights_path=str(path), weights_sha256=sha256(path.read_bytes()).hexdigest())
    return record


def _result_metrics(result, kind, split, image_count):
    metrics = getattr(result, "box" if kind == "detect" else "seg", None)
    prefix = "box" if kind == "detect" else "mask"
    measured = {f"{prefix}_{name}": number for name, attribute in
                (("map50_95", "map"), ("map50", "map50"), ("map75", "map75"))
                if (number := _finite_metric(getattr(metrics, attribute, None))) is not None}
    for key, attribute in (("precision_best_f1", "mp"), ("recall_best_f1", "mr")):
        number = _finite_metric(getattr(metrics, attribute, None))
        if number is not None:
            measured[f"{prefix}_{key}"] = number
    return {"schema_version": EVALUATION_SCHEMA_VERSION, "split": split,
            "images": image_count, **measured}


def _artifact_manifest(run_dir, model_dir, run_id, checkpoint):
    rows = []
    for path in (run_dir / "metrics.jsonl", run_dir / "evaluation.json", model_dir / "model.json", checkpoint):
        raw = path.read_bytes(); rows.append({"path": path.name, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)})
    atomic_json(run_dir / "artifact-manifest.json", {"schema_version": 1, "run_id": run_id, "artifacts": rows})


def train(dataset_manifest: Path, run_dir: Path, model_dir: Path):
    dataset_manifest, run_dir, model_dir = map(Path, (dataset_manifest, run_dir, model_dir))
    manifest, run = read_json(dataset_manifest), read_json(run_dir / "run.json")
    definition = ULTRALYTICS_ENGINES[run["engine"]]
    recorder = None
    try:
        os.environ.setdefault("YOLO_OFFLINE", "true")
        from ultralytics import RTDETR, YOLO
        data_yaml = prepare_yolo_dataset(dataset_manifest, run_dir / "dataset", definition["task"], run.get("config"))
        compatibility = (read_json(run_dir / "dataset" / "yolo-compatibility.json")
                         if definition["task"] == "instance_segmentation" else None)
        validation_split = "val"
        train_only = run.get('data_purpose') == 'all_train'
        if not any(asset["split"] == "val" for asset in manifest["assets"]) and not train_only:
            raise ValueError("Validation 沒有圖片；Test 不可用於選擇 YOLO checkpoint")
        metrics_path = run_dir / "metrics.jsonl"; metrics_path.write_text("", encoding="utf-8")
        _status(run_dir, run, status="preparing", message=f"建立 {definition['name']} 資料轉接", progress=3,
                validation_split=validation_split)
        mode, source = _initialization_source(definition, run["config"])
        _status(run_dir, run, message="載入預訓練權重" if mode == "pretrained" else "建立隨機初始化模型")
        model_class = RTDETR if run['engine'].startswith('rt_detr_') else YOLO
        model = model_class(source)
        initialization = _initialization_record(model, mode, source)
        _status(run_dir, run, initialization=initialization)
        epochs = int(run["config"]["epochs"])
        recorder = _RunMetricsRecorder(run_dir, run, definition)
        model.add_callback("on_train_epoch_start", recorder.on_epoch_start)
        model.add_callback("on_train_batch_end", recorder.on_batch_end)
        model.add_callback("on_fit_epoch_end", recorder.on_epoch_end)
        requested = run["config"].get("device", "auto")
        device = "0" if requested == "cuda" else "cpu" if requested == "cpu" else None
        schedule = run["config"].get("scheduler", "linear")
        initial_lr = float(run["config"].get("learning_rate", .0005))
        batch_size = int(run["config"].get("batch_size", 1))
        accumulation = int(run["config"].get("gradient_accumulation", 1))
        augmentation_args = yolo_augmentation_args(run["config"].get("augmentation"))
        model.train(data=str(data_yaml), epochs=epochs, imgsz=int(run["config"].get("image_size", 640)),
                             batch=batch_size, nbs=batch_size * accumulation, device=device, workers=0,
                             lr0=initial_lr, cos_lr=schedule == "cosine",
                             lrf=1.0 if schedule == "fixed" else float(run["config"].get("min_learning_rate", initial_lr*.01))/initial_lr,
                             warmup_epochs=0 if schedule == "fixed" else int(run["config"].get("warmup_epochs", 0)),
                             warmup_bias_lr=0.0,
                             weight_decay=float(run["config"].get("weight_decay", .0001)),
                             optimizer=run["config"].get("optimizer", "AdamW"),
                             **({"momentum": float(run["config"].get("momentum", .9))}
                                if run["config"].get("optimizer") == "SGD" else {}),
                             project=str(run_dir / "ultralytics"), name="fit", exist_ok=True, pretrained=mode == "pretrained",
                             plots=False, verbose=False, deterministic=False, seed=int(run["config"].get("seed", 42)), val=not train_only,
                             **augmentation_args)
        if _stopping(run_dir):
            return _status(run_dir, run, status="stopped", message="已安全停止", progress=None, completed_at=time.time())
        trainer = getattr(model, "trainer", None)
        best = Path(str(getattr(trainer, "last" if train_only else "best", "")))
        if not best.is_file():
            candidate = run_dir / "ultralytics" / "fit" / "weights" / "best.pt"
            best = candidate if candidate.is_file() else Path(str(getattr(trainer, "last", "")))
        if not best.is_file():
            raise RuntimeError("訓練完成但找不到 checkpoint")
        checkpoint = model_dir / "checkpoint.pt"; shutil.copy2(best, checkpoint)
        test_count = sum(asset["split"] == "test" for asset in manifest["assets"])
        if train_only:
            validation = test = None
            protocol = training_only_protocol()
        else:
            evaluator = model_class(str(checkpoint))
            validation_raw = evaluator.val(data=str(data_yaml), split="val", device=device, plots=False, verbose=False)
            val_count = sum(asset["split"] == validation_split for asset in manifest["assets"])
            validation = _result_metrics(validation_raw, definition["kind"], validation_split, val_count)
            test = (_result_metrics(evaluator.val(data=str(data_yaml), split="test", device=device, plots=False, verbose=False),
                                    definition["kind"], "test", test_count) if test_count else None)
            protocol = evaluation_protocol(checkpoint="best_validation", has_test=bool(test_count), manifest=manifest)
        record = {"schema_version": 1, "engine": run["engine"], "engine_name": definition["name"],
                  "task": definition["task"], "model_version_id": run["model_version_id"], "run_id": run["run_id"],
                  "dataset_version_id": manifest["dataset_version_id"], "classes": manifest["classes"],
                  "image_size": int(run["config"].get("image_size", 640)), "score_threshold": .5,
                  "validation": validation, "test": test, "evaluation_protocol": protocol,
                  "created_at": time.time(), "checkpoint": "checkpoint.pt",
                  "initialization": initialization, "execution": dict(recorder.execution)}
        if compatibility is not None:
            record["yolo_compatibility"] = compatibility
        if "data_quality" in run:
            record["data_quality"] = run["data_quality"]
        atomic_json(model_dir / "model.json", record)
        evaluation = {"schema_version": 2, "protocol": protocol, "validation": validation, "test": test}
        if compatibility is not None:
            evaluation["yolo_compatibility"] = compatibility
        if "data_quality" in run:
            evaluation["data_quality"] = run["data_quality"]
        atomic_json(run_dir / "evaluation.json", evaluation)
        _artifact_manifest(run_dir, model_dir, run["run_id"], checkpoint)
        return _status(run_dir, run, status="completed", message=f"{definition['name']} {'最終訓練完成（無獨立評估）' if train_only else '訓練與評估完成'}", progress=100,
                       completed_at=time.time(), evaluation=evaluation)
    except Exception as exc:
        _status(run_dir, run, status="failed", message=str(exc), error=str(exc), progress=None, completed_at=time.time())
        raise
    finally:
        if recorder is not None:
            recorder.close()


class Predictor:
    def __init__(self, record, model_dir, requested_device="auto"):
        os.environ.setdefault("YOLO_OFFLINE", "true")
        from ultralytics import RTDETR, YOLO
        self.record = record; self.definition = ULTRALYTICS_ENGINES[record["engine"]]
        constructor = RTDETR if record['engine'].startswith('rt_detr_') else YOLO
        self.model = constructor(str(Path(model_dir) / record["checkpoint"]))
        self.device = "0" if requested_device == "cuda" else "cpu" if requested_device == "cpu" else None

    def predict(self, rgb, width, height):
        width, height = int(width), int(height)
        pixels = np.asarray(rgb, dtype=np.uint8)
        if pixels.shape != (height, width, 3):
            raise ValueError(f"推論圖片必須是 {width} × {height} 的 RGB 影像；實際尺寸為 {pixels.shape}")
        # The worker supplies RGB; Ultralytics interprets numpy images as BGR.
        # Native masks remove letterbox padding and return original-image pixels.
        source = np.ascontiguousarray(pixels[..., ::-1])
        threshold = float(self.record.get("score_threshold", .5))
        result = self.model.predict(source=source, imgsz=int(self.record.get("image_size", 640)),
                                    conf=threshold, retina_masks=self.definition["kind"] == "segment",
                                    device=self.device, verbose=False)[0]
        shapes, boxes = [], getattr(result, "boxes", None)
        if boxes is None:
            return shapes
        labels = boxes.cls.cpu().tolist(); scores = boxes.conf.cpu().tolist()
        if self.definition["kind"] == "detect":
            for label_id, score, box in zip(labels, scores, boxes.xyxy.cpu().tolist()):
                if score < threshold or not 0 <= int(label_id) < len(self.record["classes"]): continue
                x1, y1, x2, y2 = map(float, box)
                shapes.append({"id": uuid.uuid4().hex, "type": "rectangle", "label": self.record["classes"][int(label_id)],
                               "x": max(0., x1), "y": max(0., y1), "width": max(1., min(width, x2)-max(0., x1)),
                               "height": max(1., min(height, y2)-max(0., y1)),
                               "metadata": {"source": "model_candidate", "engine": self.record["engine"],
                                            "model_version_id": self.record["model_version_id"], "confidence": round(float(score), 6)}})
        else:
            masks = getattr(result, "masks", None)
            if masks is None: return shapes
            raw_masks = masks.data.cpu().numpy()
            if raw_masks.ndim != 3 or raw_masks.shape[1:] != (height, width):
                raise ValueError(f"模型未回傳原圖尺寸的遮罩（預期 {width} × {height}，實際 {raw_masks.shape}）；"
                                 "已停止匯入，避免遮罩位置錯誤")
            for label_id, score, raw in zip(labels, scores, raw_masks):
                if score < threshold or not 0 <= int(label_id) < len(self.record["classes"]): continue
                bitmap = raw >= .5
                if not bitmap.any(): continue
                ys, xs = np.nonzero(bitmap)
                shapes.append({"id": uuid.uuid4().hex, "type": "mask", "label": self.record["classes"][int(label_id)],
                               "x": 0, "y": 0, "width": width, "height": height, "counts": encode_rle(bitmap),
                               "metadata": {"source": "model_candidate", "engine": self.record["engine"],
                                            "model_version_id": self.record["model_version_id"], "confidence": round(float(score), 6),
                                            "bounds": [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]}})
        return shapes
