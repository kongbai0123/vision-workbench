"""TorchVision image-classification training adapter.

An approved image is a classification sample when all of its annotations use
exactly one class.  Shapes are only used to derive that image-level class; the
adapter never turns a classification result into a full-image bounding box.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
import time

import numpy as np

from .training_engine import atomic_json, load_rgb, read_json, _status, _stopping
from .training_parameters import create_optimizer
from .learning_rates import LearningRateSchedule
from .evaluation_metrics import EVALUATION_SCHEMA_VERSION, evaluation_protocol, training_only_protocol
from .augmentation import augment_image_tensor


CLASSIFICATION_ENGINES = {
    "mobilenet_v3_large_classification": "MobileNet V3 · 分類",
    "efficientnet_b0_classification": "EfficientNet-B0 · 分類",
    "resnet18_classification": "ResNet18 · 分類",
}


def _modules():
    try:
        import torch
        from torch.utils.data import DataLoader
        import torchvision
    except Exception as exc:
        raise RuntimeError("TorchVision 訓練環境尚未安裝或無法載入") from exc
    return torch, DataLoader, torchvision


def _model(engine, class_count):
    torch, _loader, torchvision = _modules()
    constructors = {
        "mobilenet_v3_large_classification": torchvision.models.mobilenet_v3_large,
        "efficientnet_b0_classification": torchvision.models.efficientnet_b0,
        "resnet18_classification": torchvision.models.resnet18,
    }
    constructor = constructors.get(engine)
    if constructor is None:
        raise ValueError(f"不支援的分類引擎：{engine}")
    # weights=None is deliberate: selecting or training a model must not
    # silently download a checkpoint.
    return constructor(weights=None, num_classes=class_count), torch


def _asset_label(asset, classes):
    labels = {shape.get("label") for shape in asset.get("shapes", []) if shape.get("label") in classes}
    if len(labels) != 1:
        raise ValueError(f"分類圖片 {asset.get('name') or asset.get('asset_id')} 必須只有一個標註類別")
    return classes.index(next(iter(labels)))


class ClassificationDataset:
    def __init__(self, manifest, dataset_dir, split, torch, image_size, augmentation=None):
        self.manifest, self.dataset_dir, self.torch = manifest, Path(dataset_dir), torch
        self.assets = [asset for asset in manifest["assets"] if asset["split"] == split]
        self.classes, self.image_size = manifest["classes"], int(image_size)
        # Fail before the first epoch so an ambiguous image never receives an
        # arbitrary class.
        self.targets = [_asset_label(asset, self.classes) for asset in self.assets]
        self.augmentation = augmentation

    def __len__(self):
        return len(self.assets)

    def __getitem__(self, index):
        torch, asset = self.torch, self.assets[index]
        rgb = load_rgb(self.dataset_dir, asset)
        image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
        image = torch.nn.functional.interpolate(
            image[None], (self.image_size, self.image_size), mode="bilinear", align_corners=False
        )[0]
        if self.augmentation:
            image, _horizontal, _vertical = augment_image_tensor(image, self.augmentation, torch)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=image.dtype)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=image.dtype)[:, None, None]
        return (image - mean) / std, self.targets[index], asset


def _evaluate(model, dataset, device, torch):
    model.eval()
    confusion = np.zeros((len(dataset.classes), len(dataset.classes)), dtype=np.int64)
    with torch.inference_mode():
        for image, expected, _asset in dataset:
            predicted = int(model(image[None].to(device)).argmax(1).cpu().item())
            confusion[int(expected), predicted] += 1
    recalls, f1s = [], []
    per_class = {}
    for class_id in range(len(dataset.classes)):
        tp = int(confusion[class_id, class_id])
        fn = int(confusion[class_id, :].sum()) - tp
        fp = int(confusion[:, class_id].sum()) - tp
        support = tp + fn
        predicted = tp + fp
        recall = tp / support if support else None
        precision = tp / predicted if predicted else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        if support:
            recalls.append(recall)
            f1s.append(f1 or 0.0)
        per_class[dataset.classes[class_id]] = {
            "support": support, "predictions": predicted, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 6) if precision is not None else None,
            "recall": round(recall, 6) if recall is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
        }
    total = int(confusion.sum())
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "split": dataset.assets[0]["split"] if dataset.assets else "",
        "images": len(dataset),
        "accuracy": round(float(np.trace(confusion)) / max(1, total), 6),
        "macro_f1": round(sum(f1s) / max(1, len(f1s)), 6),
        "macro_recall": round(sum(recalls) / max(1, len(recalls)), 6),
        "per_class_recall": {name: row["recall"] for name, row in per_class.items()},
        "per_class": per_class,
        "macro_policy": "ground_truth_supported_classes",
        "classes": list(dataset.classes),
        "confusion_matrix": confusion.tolist(),
    }


def _artifacts(run_dir, model_dir, run_id, checkpoint):
    paths = (run_dir / "metrics.jsonl", run_dir / "evaluation.json", model_dir / "model.json", checkpoint)
    rows = []
    for path in paths:
        raw = path.read_bytes()
        rows.append({"path": path.name, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)})
    atomic_json(run_dir / "artifact-manifest.json", {"schema_version": 1, "run_id": run_id, "artifacts": rows})


def train(dataset_manifest: Path, run_dir: Path, model_dir: Path):
    dataset_manifest, run_dir, model_dir = map(Path, (dataset_manifest, run_dir, model_dir))
    manifest, run = read_json(dataset_manifest), read_json(run_dir / "run.json")
    try:
        torch, DataLoader, _torchvision = _modules()
        seed = int(run["config"].get("seed", 42))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        requested = run["config"].get("device", "auto")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("已指定 CUDA，但目前環境無法使用 CUDA")
        device = torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")
        image_size = int(run["config"].get("image_size", 224))
        model, _ = _model(run["engine"], len(manifest["classes"])); model.to(device)
        training = ClassificationDataset(manifest, dataset_manifest.parent, "train", torch, image_size,
                                         augmentation=run["config"].get("augmentation"))
        validation = ClassificationDataset(manifest, dataset_manifest.parent, "val", torch, image_size)
        testing = ClassificationDataset(manifest, dataset_manifest.parent, "test", torch, image_size)
        train_only = run.get('data_purpose') == 'all_train'
        if not training.assets or (not validation.assets and not train_only):
            raise RuntimeError("影像分類需要非空的 Train 與 Validation；Test 不會替代 Validation")
        train_classes = {training.targets[index] for index in range(len(training.targets))}
        missing = [name for index, name in enumerate(manifest["classes"]) if index not in train_classes]
        from .split_quality import loose_split_applies
        if missing and not loose_split_applies(manifest.get('split_plan'), manifest['assets']):
            raise RuntimeError(f"Train 缺少分類樣本：{'、'.join(missing)}")
        loader = DataLoader(training, batch_size=max(1, int(run["config"].get("batch_size", 1))),
                            shuffle=True, num_workers=0)
        optimizer = create_optimizer(torch, model.parameters(), run["config"])
        scheduler = LearningRateSchedule(optimizer, run["config"])
        epochs = int(run["config"].get("epochs", 10)); metrics_path = run_dir / "metrics.jsonl"
        metrics_path.write_text("", encoding="utf-8")
        _status(run_dir, run, status="preparing", message=f"載入 {CLASSIFICATION_ENGINES[run['engine']]} · {device}", progress=3)
        optimizer_steps = 0
        for epoch in range(1, epochs + 1):
            rates = scheduler.start_epoch(epoch)
            model.train(); losses = []
            for batch, (images, targets, _assets) in enumerate(loader, 1):
                if _stopping(run_dir):
                    return _status(run_dir, run, status="stopped", message="已安全停止", progress=None)
                logits = model(images.to(device)); loss = torch.nn.functional.cross_entropy(logits, targets.to(device))
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach().cpu()))
                _status(run_dir, run, status="running", phase="training",
                        message=f"{CLASSIFICATION_ENGINES[run['engine']]} · Epoch {epoch}/{epochs} · Batch {batch}/{len(loader)}",
                        epoch=epoch, batch=batch, batches_per_epoch=len(loader),
                        progress=5 + round((((epoch - 1) + batch / len(loader)) / epochs) * 82),
                        execution={"batch_size": loader.batch_size, "gradient_accumulation": 1,
                                   "effective_batch_size": loader.batch_size, "optimizer_steps": optimizer_steps})
            score = _evaluate(model, validation, device, torch) if validation.assets else None
            row = {"epoch": epoch, "train/loss": round(sum(losses) / max(1, len(losses)), 6)}
            if score: row.update({"val/accuracy": score["accuracy"], "val/macro_f1": score["macro_f1"],
                                  "val/macro_recall": score["macro_recall"]})
            row.update(rates)
            scheduler.finish_epoch(score["accuracy"] if score else None)
            atomic_json(run_dir / "lr-state.json", scheduler.state_dict())
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            _status(run_dir, run, status="running", message=f"{CLASSIFICATION_ENGINES[run['engine']]} {epoch} / {epochs}",
                    phase="training" if train_only else "validation", epoch=epoch, batch=len(loader), batches_per_epoch=len(loader),
                    progress=5 + round(epoch / epochs * 85), metrics=row, device=str(device),
                    execution={"batch_size": loader.batch_size, "gradient_accumulation": 1,
                               "effective_batch_size": loader.batch_size, "optimizer_steps": optimizer_steps})
        checkpoint = model_dir / "checkpoint.pt"
        torch.save({"model_state": model.state_dict(), "classes": manifest["classes"], "image_size": image_size, "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict()}, checkpoint)
        validation_result = _evaluate(model, validation, device, torch) if validation.assets else None
        test_result = _evaluate(model, testing, device, torch) if testing.assets else None
        protocol = (training_only_protocol() if train_only else
                    evaluation_protocol(checkpoint="final_epoch", has_test=bool(testing.assets), manifest=manifest))
        record = {"schema_version": 1, "engine": run["engine"], "engine_name": CLASSIFICATION_ENGINES[run["engine"]],
                  "task": "image_classification", "model_version_id": run["model_version_id"], "run_id": run["run_id"],
                  "dataset_version_id": manifest["dataset_version_id"], "classes": manifest["classes"],
                  "image_size": image_size, "validation": validation_result, "test": test_result,
                  "evaluation_protocol": protocol,
                  "created_at": time.time(), "checkpoint": "checkpoint.pt", "prediction_mode": "image_class"}
        atomic_json(model_dir / "model.json", record)
        atomic_json(run_dir / "evaluation.json", {"schema_version": 2, "protocol": protocol,
                                                   "validation": validation_result, "test": test_result})
        _artifacts(run_dir, model_dir, run["run_id"], checkpoint)
        return _status(run_dir, run, status="completed", message=f"{CLASSIFICATION_ENGINES[run['engine']]} {'最終訓練完成（無獨立評估）' if train_only else '訓練與評估完成'}",
                       progress=100, completed_at=time.time(), evaluation={"schema_version": 2, "protocol": protocol,
                                                                          "validation": validation_result, "test": test_result})
    except Exception as exc:
        _status(run_dir, run, status="failed", message=str(exc), error=str(exc), progress=None, completed_at=time.time())
        raise
