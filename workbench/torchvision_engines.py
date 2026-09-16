"""TorchVision detection and semantic-segmentation adapters."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np

from composer_core.geometry import encode_rle
from .maskrcnn_engine import NativeMaskDataset, _collate, _device
from .training_engine import annotation_mask, atomic_json, load_rgb, read_json, _status, _stopping
from .evaluation_metrics import PixelMetrics, detection_summary, evaluation_protocol, training_only_protocol
from .training_parameters import create_optimizer
from .learning_rates import LearningRateSchedule
from .augmentation import augment_dense_target


DETECTION_ENGINES = {
    "fasterrcnn_mobilenet_v3_large_fpn": "Faster R-CNN · MobileNet V3 FPN",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "Faster R-CNN · MobileNet V3 320 FPN",
    "fasterrcnn_resnet50_fpn_v2": "Faster R-CNN · ResNet50 FPN V2",
}
SEMANTIC_ENGINES = {
    "deeplabv3_mobilenet_v3_large": "DeepLabV3 · MobileNet V3",
    "deeplabv3_resnet50": "DeepLabV3 · ResNet50",
}
ENGINE_NAMES = {**DETECTION_ENGINES, **SEMANTIC_ENGINES}


def _modules():
    try:
        import torch
        from torch.utils.data import DataLoader
        import torchvision
    except Exception as exc:
        raise RuntimeError("TorchVision 訓練環境尚未安裝或無法載入") from exc
    return torch, DataLoader, torchvision


def _detection_model(engine, class_count, image_size):
    torch, _loader, torchvision = _modules()
    constructor = getattr(torchvision.models.detection, engine, None)
    if constructor is None:
        raise RuntimeError(f"目前 TorchVision 版本不支援 {engine}")
    options = {"weights": None, "weights_backbone": None, "num_classes": class_count + 1,
               "min_size": int(image_size), "max_size": int(image_size)}
    return constructor(**options), torch


def _semantic_model(engine, class_count):
    torch, _loader, torchvision = _modules()
    constructor = getattr(torchvision.models.segmentation, engine, None)
    if constructor is None:
        raise RuntimeError(f"目前 TorchVision 版本不支援 {engine}")
    return constructor(weights=None, weights_backbone=None, num_classes=class_count + 1), torch


def _evaluate_detection(model, dataset, device, torch, threshold=.5):
    model.eval(); samples = []
    with torch.inference_mode():
        for image, target, _asset in dataset:
            output = model([image.to(device)])[0]
            expected_by_label = {label: [] for label in dataset.manifest["classes"]}
            for label, expected in zip(target["labels"].tolist(), target["boxes"].tolist()):
                class_id = int(label)
                if 1 <= class_id <= len(dataset.manifest["classes"]):
                    expected_by_label[dataset.manifest["classes"][class_id - 1]].append([float(v) for v in expected])
            predicted_by_label = {label: [] for label in dataset.manifest["classes"]}
            for label, score, box in zip(output["labels"].detach().cpu().tolist(),
                                         output["scores"].detach().cpu().tolist(),
                                         output["boxes"].detach().cpu().tolist()):
                class_id = int(label)
                if 1 <= class_id <= len(dataset.manifest["classes"]):
                    predicted_by_label[dataset.manifest["classes"][class_id - 1]].append(
                        (float(score), [float(v) for v in box]))
            samples.append({"expected": expected_by_label, "predicted": predicted_by_label})
    split = dataset.assets[0]["split"] if dataset.assets else ""
    return detection_summary(dataset.manifest["classes"], samples, split, len(dataset),
                             score_threshold=threshold, iou_threshold=.5)


class SemanticDataset:
    def __init__(self, manifest, dataset_dir, split, torch, image_size, augmentation=None):
        self.manifest, self.dataset_dir, self.torch = manifest, Path(dataset_dir), torch
        self.assets = [asset for asset in manifest["assets"] if asset["split"] == split]
        self.class_ids = {label: index + 1 for index, label in enumerate(manifest["classes"])}
        self.image_size = int(image_size)
        self.augmentation = augmentation

    def __len__(self): return len(self.assets)

    def __getitem__(self, index):
        torch = self.torch; asset = self.assets[index]
        rgb = load_rgb(self.dataset_dir, asset)
        image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
        target = np.zeros((int(asset["height"]), int(asset["width"])), dtype=np.int64)
        for shape in asset.get("shapes", []):
            class_id = self.class_ids.get(shape.get("label"))
            if class_id:
                target[annotation_mask(shape, int(asset["width"]), int(asset["height"]))] = class_id
        functional = torch.nn.functional
        image = functional.interpolate(image[None], (self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
        target_tensor = torch.from_numpy(target)[None, None].float()
        target_tensor = functional.interpolate(target_tensor, (self.image_size, self.image_size), mode="nearest")[0, 0].long()
        if self.augmentation:
            image, target_tensor = augment_dense_target(image, target_tensor, self.augmentation, torch)
        return image, target_tensor, asset


def _evaluate_semantic(model, dataset, device, torch):
    model.eval(); metrics = PixelMetrics(dataset.manifest["classes"])
    with torch.inference_mode():
        for image, target, _asset in dataset:
            output = model(image[None].to(device))["out"].argmax(1)[0].cpu().numpy()
            expected = target.numpy()
            for class_id in range(1, len(dataset.manifest["classes"]) + 1):
                predicted_mask, expected_mask = output == class_id, expected == class_id
                metrics.update(dataset.manifest["classes"][class_id - 1], predicted_mask, expected_mask)
    split = dataset.assets[0]["split"] if dataset.assets else ""
    return metrics.summary(split, len(dataset), include_dice=True)


def _semantic_training_mode(model, torch, batch_size):
    model.train()
    if batch_size == 1:
        # DeepLab's pooled branch has a 1x1 BatchNorm input. Keep every image,
        # using running statistics for singleton batches while training affine weights.
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()


def _artifacts(run_dir, model_dir, run_id, checkpoint):
    paths = (run_dir / "metrics.jsonl", run_dir / "evaluation.json", model_dir / "model.json", checkpoint)
    rows = []
    for path in paths:
        raw = path.read_bytes(); rows.append({"path": path.name, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)})
    atomic_json(run_dir / "artifact-manifest.json", {"schema_version": 1, "run_id": run_id, "artifacts": rows})


def _base_record(run, manifest, engine, image_size, validation, test, protocol):
    return {"schema_version": 1, "engine": engine, "engine_name": ENGINE_NAMES[engine],
            "task": "object_detection" if engine in DETECTION_ENGINES else "semantic_segmentation",
            "model_version_id": run["model_version_id"], "run_id": run["run_id"],
            "dataset_version_id": manifest["dataset_version_id"], "classes": manifest["classes"],
            "image_size": image_size, "score_threshold": .5, "validation": validation, "test": test,
            "evaluation_protocol": protocol,
            "created_at": time.time(), "checkpoint": "checkpoint.pt"}


def train_detection(dataset_manifest: Path, run_dir: Path, model_dir: Path):
    dataset_manifest, run_dir, model_dir = map(Path, (dataset_manifest, run_dir, model_dir))
    manifest, run = read_json(dataset_manifest), read_json(run_dir / "run.json")
    try:
        torch, DataLoader, _tv = _modules(); seed = int(run["config"].get("seed", 42))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        device = _device(torch, run["config"].get("device")); image_size = int(run["config"].get("image_size", 640))
        model, _ = _detection_model(run["engine"], len(manifest["classes"]), image_size); model.to(device)
        training = NativeMaskDataset(manifest, dataset_manifest.parent, "train", torch,
                                     run["config"].get("augmentation"))
        validation = NativeMaskDataset(manifest, dataset_manifest.parent, "val", torch)
        testing = NativeMaskDataset(manifest, dataset_manifest.parent, "test", torch)
        train_only = run.get('data_purpose') == 'all_train'
        if not training.assets or (not validation.assets and not train_only): raise RuntimeError("Faster R-CNN 需要非空的 Train 與 Validation；Test 不會替代 Validation")
        loader = DataLoader(training, batch_size=max(1, int(run["config"].get("batch_size", 1))), shuffle=True,
                            num_workers=0, collate_fn=_collate)
        optimizer = create_optimizer(torch, [p for p in model.parameters() if p.requires_grad], run["config"])
        scheduler = LearningRateSchedule(optimizer, run["config"])
        epochs = int(run["config"].get("epochs", 10)); metrics_path = run_dir / "metrics.jsonl"
        metrics_path.write_text("", encoding="utf-8"); _status(run_dir, run, status="preparing", message=f"載入 {ENGINE_NAMES[run['engine']]} · {device}", progress=3)
        optimizer_steps = 0
        for epoch in range(1, epochs + 1):
            rates = scheduler.start_epoch(epoch)
            model.train(); losses = []
            for batch, (images, targets, _assets) in enumerate(loader, 1):
                if _stopping(run_dir): return _status(run_dir, run, status="stopped", message="已安全停止", progress=None)
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items() if key != "masks"} for target in targets]
                loss = sum(model(images, targets).values()); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); optimizer_steps += 1
                losses.append(float(loss.detach().cpu()))
                _status(run_dir, run, status="running", phase="training",
                        message=f"{ENGINE_NAMES[run['engine']]} · Epoch {epoch}/{epochs} · Batch {batch}/{len(loader)}",
                        epoch=epoch, batch=batch, batches_per_epoch=len(loader),
                        progress=5 + round((((epoch - 1) + batch / len(loader)) / epochs) * 82),
                        execution={"batch_size": loader.batch_size, "gradient_accumulation": 1,
                                   "effective_batch_size": loader.batch_size, "optimizer_steps": optimizer_steps})
            score = _evaluate_detection(model, validation, device, torch) if validation.assets else None
            row = {"epoch": epoch, "train/loss": round(sum(losses) / max(1, len(losses)), 6)}
            if score: row.update({"val/box_map50": score["box_map50"], "val/recall_50": score["recall_50"]})
            row.update(rates)
            scheduler.finish_epoch(score.get("box_map50") if score else None)
            atomic_json(run_dir / "lr-state.json", scheduler.state_dict())
            with metrics_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            _status(run_dir, run, status="running", message=f"{ENGINE_NAMES[run['engine']]} {epoch} / {epochs}", epoch=epoch,
                    phase="training" if train_only else "validation", batch=len(loader), batches_per_epoch=len(loader),
                    progress=5 + round(epoch / epochs * 85), metrics=row, device=str(device),
                    execution={"batch_size": loader.batch_size, "gradient_accumulation": 1,
                               "effective_batch_size": loader.batch_size, "optimizer_steps": optimizer_steps})
        checkpoint = model_dir / "checkpoint.pt"; torch.save({"model_state": model.state_dict(), "classes": manifest["classes"], "image_size": image_size, "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict()}, checkpoint)
        validation_result = _evaluate_detection(model, validation, device, torch) if validation.assets else None
        test_result = _evaluate_detection(model, testing, device, torch) if testing.assets else None
        protocol = training_only_protocol() if train_only else evaluation_protocol(checkpoint="final_epoch", has_test=bool(testing.assets), manifest=manifest)
        record = _base_record(run, manifest, run["engine"], image_size, validation_result, test_result, protocol)
        atomic_json(model_dir / "model.json", record); atomic_json(run_dir / "evaluation.json", {"schema_version": 2, "protocol": protocol, "validation": validation_result, "test": test_result})
        _artifacts(run_dir, model_dir, run["run_id"], checkpoint)
        return _status(run_dir, run, status="completed", message=f"{ENGINE_NAMES[run['engine']]} {'最終訓練完成（無獨立評估）' if train_only else '訓練與評估完成'}", progress=100,
                       completed_at=time.time(), evaluation={"schema_version": 2, "protocol": protocol, "validation": validation_result, "test": test_result})
    except Exception as exc:
        _status(run_dir, run, status="failed", message=str(exc), error=str(exc), progress=None, completed_at=time.time()); raise


def train_semantic(dataset_manifest: Path, run_dir: Path, model_dir: Path):
    dataset_manifest, run_dir, model_dir = map(Path, (dataset_manifest, run_dir, model_dir))
    manifest, run = read_json(dataset_manifest), read_json(run_dir / "run.json")
    try:
        torch, DataLoader, _tv = _modules(); seed = int(run["config"].get("seed", 42))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        device = _device(torch, run["config"].get("device")); image_size = int(run["config"].get("image_size", 512))
        model, _ = _semantic_model(run["engine"], len(manifest["classes"])); model.to(device)
        training = SemanticDataset(manifest, dataset_manifest.parent, "train", torch, image_size,
                                   run["config"].get("augmentation"))
        validation = SemanticDataset(manifest, dataset_manifest.parent, "val", torch, image_size)
        testing = SemanticDataset(manifest, dataset_manifest.parent, "test", torch, image_size)
        train_only = run.get('data_purpose') == 'all_train'
        if not training.assets or (not validation.assets and not train_only): raise RuntimeError("DeepLabV3 需要非空的 Train 與 Validation；Test 不會替代 Validation")
        loader = DataLoader(training, batch_size=max(1, int(run["config"].get("batch_size", 1))), shuffle=True, num_workers=0)
        optimizer = create_optimizer(torch, model.parameters(), run["config"])
        scheduler = LearningRateSchedule(optimizer, run["config"])
        epochs = int(run["config"].get("epochs", 10)); metrics_path = run_dir / "metrics.jsonl"
        metrics_path.write_text("", encoding="utf-8"); _status(run_dir, run, status="preparing", message=f"載入 {ENGINE_NAMES[run['engine']]} · {device}", progress=3)
        optimizer_steps = 0
        for epoch in range(1, epochs + 1):
            rates = scheduler.start_epoch(epoch)
            model.train(); losses = []
            for batch, (images, targets, _assets) in enumerate(loader, 1):
                if _stopping(run_dir): return _status(run_dir, run, status="stopped", message="已安全停止", progress=None)
                _semantic_training_mode(model, torch, len(images))
                output = model(images.to(device))["out"]
                loss = torch.nn.functional.cross_entropy(output, targets.to(device)); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); optimizer_steps += 1
                losses.append(float(loss.detach().cpu()))
                _status(run_dir, run, status="running", phase="training",
                        message=f"{ENGINE_NAMES[run['engine']]} · Epoch {epoch}/{epochs} · Batch {batch}/{len(loader)}",
                        epoch=epoch, batch=batch, batches_per_epoch=len(loader),
                        progress=5 + round((((epoch - 1) + batch / len(loader)) / epochs) * 82),
                        execution={"batch_size": loader.batch_size, "gradient_accumulation": 1,
                                   "effective_batch_size": loader.batch_size, "optimizer_steps": optimizer_steps})
            score = _evaluate_semantic(model, validation, device, torch) if validation.assets else None
            row = {"epoch": epoch, "train/loss": round(sum(losses) / max(1, len(losses)), 6)}
            if score: row.update({"val/mean_iou": score["mean_iou"], "val/mean_dice": score["mean_dice"]})
            row.update(rates)
            scheduler.finish_epoch(score.get("box_mean_iou", score.get("mean_iou")) if score else None)
            atomic_json(run_dir / "lr-state.json", scheduler.state_dict())
            with metrics_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            _status(run_dir, run, status="running", message=f"{ENGINE_NAMES[run['engine']]} {epoch} / {epochs}", epoch=epoch,
                    phase="training" if train_only else "validation", batch=len(loader), batches_per_epoch=len(loader),
                    progress=5 + round(epoch / epochs * 85), metrics=row, device=str(device),
                    execution={"batch_size": loader.batch_size, "gradient_accumulation": 1,
                               "effective_batch_size": loader.batch_size, "optimizer_steps": optimizer_steps})
        checkpoint = model_dir / "checkpoint.pt"; torch.save({"model_state": model.state_dict(), "classes": manifest["classes"], "image_size": image_size, "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict()}, checkpoint)
        validation_result = _evaluate_semantic(model, validation, device, torch) if validation.assets else None
        test_result = _evaluate_semantic(model, testing, device, torch) if testing.assets else None
        protocol = training_only_protocol() if train_only else evaluation_protocol(checkpoint="final_epoch", has_test=bool(testing.assets), manifest=manifest)
        record = _base_record(run, manifest, run["engine"], image_size, validation_result, test_result, protocol)
        atomic_json(model_dir / "model.json", record); atomic_json(run_dir / "evaluation.json", {"schema_version": 2, "protocol": protocol, "validation": validation_result, "test": test_result})
        _artifacts(run_dir, model_dir, run["run_id"], checkpoint)
        return _status(run_dir, run, status="completed", message=f"{ENGINE_NAMES[run['engine']]} {'最終訓練完成（無獨立評估）' if train_only else '訓練與評估完成'}", progress=100,
                       completed_at=time.time(), evaluation={"schema_version": 2, "protocol": protocol, "validation": validation_result, "test": test_result})
    except Exception as exc:
        _status(run_dir, run, status="failed", message=str(exc), error=str(exc), progress=None, completed_at=time.time()); raise


def train(dataset_manifest: Path, run_dir: Path, model_dir: Path):
    engine = read_json(Path(run_dir) / "run.json")["engine"]
    if engine in DETECTION_ENGINES: return train_detection(dataset_manifest, run_dir, model_dir)
    if engine in SEMANTIC_ENGINES: return train_semantic(dataset_manifest, run_dir, model_dir)
    raise ValueError(f"不支援的 TorchVision 引擎：{engine}")


class Predictor:
    def __init__(self, record, model_dir, requested_device="auto"):
        self.record = record; self.engine = record["engine"]
        if self.engine in DETECTION_ENGINES: self.model, self.torch = _detection_model(self.engine, len(record["classes"]), record["image_size"])
        elif self.engine in SEMANTIC_ENGINES: self.model, self.torch = _semantic_model(self.engine, len(record["classes"]))
        else: raise ValueError("模型引擎不支援推論")
        self.device = _device(self.torch, requested_device)
        checkpoint = self.torch.load(Path(model_dir) / record["checkpoint"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"]); self.model.to(self.device); self.model.eval()

    def predict(self, rgb, width, height):
        torch = self.torch; image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().to(self.device) / 255.0
        shapes = []
        with torch.inference_mode():
            if self.engine in DETECTION_ENGINES:
                output = self.model([image])[0]
                for score, label, box in zip(output["scores"].cpu().tolist(), output["labels"].cpu().tolist(), output["boxes"].cpu().tolist()):
                    if score < float(self.record.get("score_threshold", .5)) or not 1 <= label <= len(self.record["classes"]): continue
                    x1, y1, x2, y2 = box
                    shapes.append({"id": uuid.uuid4().hex, "type": "rectangle", "label": self.record["classes"][label - 1],
                                   "x": max(0., x1), "y": max(0., y1), "width": max(1., min(width, x2) - max(0., x1)),
                                   "height": max(1., min(height, y2) - max(0., y1)),
                                   "metadata": {"source": "model_candidate", "engine": self.engine,
                                                "model_version_id": self.record["model_version_id"], "confidence": round(float(score), 6)}})
            else:
                size = int(self.record.get("image_size", 512)); resized = torch.nn.functional.interpolate(image[None], (size, size), mode="bilinear", align_corners=False)
                logits = self.model(resized)["out"]
                output = torch.nn.functional.interpolate(logits, (height, width), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
                for label_id, label in enumerate(self.record["classes"], 1):
                    bitmap = output == label_id
                    if not bitmap.any(): continue
                    ys, xs = np.nonzero(bitmap)
                    shapes.append({"id": uuid.uuid4().hex, "type": "mask", "label": label, "x": 0, "y": 0,
                                   "width": width, "height": height, "counts": encode_rle(bitmap),
                                   "metadata": {"source": "model_candidate", "engine": self.engine,
                                                "model_version_id": self.record["model_version_id"],
                                                "bounds": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}})
        return shapes
