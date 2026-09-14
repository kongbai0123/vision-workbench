"""TorchVision Mask R-CNN adapter for immutable Workbench datasets."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np

from composer_core.geometry import encode_rle
from .training_engine import atomic_json, annotation_mask, load_rgb, read_json, _iou, _status, _stopping


ENGINE_KEY = "maskrcnn_resnet50_fpn"
ENGINE_NAME = "Mask R-CNN · ResNet50 FPN"


def _torch_modules():
    try:
        import torch
        from torch.utils.data import DataLoader
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except Exception as exc:
        raise RuntimeError("Mask R-CNN 訓練 runtime 尚未安裝或無法載入") from exc
    return torch, DataLoader, torchvision, FastRCNNPredictor, MaskRCNNPredictor


def _device(torch, requested):
    requested = str(requested or "auto").lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已指定 CUDA，但訓練 runtime 無法使用 GPU")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def _model(class_count, image_size=None):
    torch, _loader, torchvision, FastRCNNPredictor, MaskRCNNPredictor = _torch_modules()
    constructor = getattr(torchvision.models.detection, "maskrcnn_resnet50_fpn_v2",
                          torchvision.models.detection.maskrcnn_resnet50_fpn)
    options = {"weights": None, "weights_backbone": None}
    if image_size:
        options.update(min_size=int(image_size), max_size=int(image_size))
    model = constructor(**options)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, class_count + 1)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, class_count + 1)
    return model, torch


class NativeMaskDataset:
    def __init__(self, manifest, dataset_dir, split, torch):
        self.manifest, self.dataset_dir, self.torch = manifest, Path(dataset_dir), torch
        self.assets = [asset for asset in manifest["assets"] if asset["split"] == split]
        self.class_ids = {label: index + 1 for index, label in enumerate(manifest["classes"])}

    def __len__(self):
        return len(self.assets)

    def __getitem__(self, index):
        asset = self.assets[index]
        rgb = load_rgb(self.dataset_dir, asset)
        image = self.torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
        masks, boxes, labels = [], [], []
        for shape in asset.get("shapes", []):
            if shape.get("label") not in self.class_ids:
                continue
            mask = annotation_mask(shape, int(asset["width"]), int(asset["height"]))
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            masks.append(self.torch.from_numpy(mask.astype(np.uint8)))
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1])
            labels.append(self.class_ids[shape["label"]])
        target = {
            "boxes": self.torch.tensor(boxes, dtype=self.torch.float32).reshape(-1, 4),
            "labels": self.torch.tensor(labels, dtype=self.torch.int64),
            "masks": self.torch.stack(masks) if masks else self.torch.zeros((0, int(asset["height"]), int(asset["width"])), dtype=self.torch.uint8),
            "image_id": self.torch.tensor([index], dtype=self.torch.int64),
        }
        return image, target, asset


def _collate(rows):
    return tuple(zip(*rows))


def _evaluate(model, dataset, device, torch, threshold=0.5):
    model.eval(); values = defaultdict(list)
    with torch.inference_mode():
        for image, target, _asset in dataset:
            output = model([image.to(device)])[0]
            expected = defaultdict(lambda: np.zeros(image.shape[1:], dtype=bool))
            for label, mask in zip(target["labels"].tolist(), target["masks"].numpy()):
                expected[int(label)] |= mask > 0
            predicted = defaultdict(lambda: np.zeros(image.shape[1:], dtype=bool))
            for score, label, mask in zip(output["scores"].detach().cpu().tolist(), output["labels"].detach().cpu().tolist(), output["masks"].detach().cpu().numpy()):
                if score >= threshold:
                    predicted[int(label)] |= mask[0] >= 0.5
            for class_id in range(1, len(dataset.manifest["classes"]) + 1):
                values[class_id].append(_iou(predicted[class_id], expected[class_id]))
    per_class = {dataset.manifest["classes"][class_id - 1]: round(sum(rows) / max(1, len(rows)), 6)
                 for class_id, rows in values.items()}
    return {"split": dataset.assets[0]["split"] if dataset.assets else "", "images": len(dataset),
            "mean_iou": round(sum(per_class.values()) / max(1, len(per_class)), 6), "per_class_iou": per_class}


def train(dataset_manifest: Path, run_dir: Path, model_dir: Path):
    dataset_manifest, run_dir, model_dir = map(Path, (dataset_manifest, run_dir, model_dir))
    manifest, run = read_json(dataset_manifest), read_json(run_dir / "run.json")
    try:
        torch, DataLoader, _tv, _box, _mask = _torch_modules()
        seed = int(run["config"].get("seed", 42)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        device = _device(torch, run["config"].get("device"))
        image_size = int(run["config"].get("image_size", 640))
        model, _ = _model(len(manifest["classes"]), image_size)
        model.to(device)
        training = NativeMaskDataset(manifest, dataset_manifest.parent, "train", torch)
        validation_split = "val" if any(a["split"] == "val" for a in manifest["assets"]) else "test"
        validation = NativeMaskDataset(manifest, dataset_manifest.parent, validation_split, torch)
        testing = NativeMaskDataset(manifest, dataset_manifest.parent, "test", torch)
        if not training.assets or not validation.assets:
            raise RuntimeError("Mask R-CNN 需要非空的 Train 與 Validation/Test 資料")
        loader = DataLoader(training, batch_size=max(1, int(run["config"].get("batch_size", 1))), shuffle=True,
                            num_workers=0, collate_fn=_collate)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=float(run["config"].get("learning_rate", 0.0005)), weight_decay=0.0001)
        epochs = int(run["config"].get("epochs", 10)); metrics_path = run_dir / "metrics.jsonl"
        metrics_path.write_text("", encoding="utf-8")
        _status(run_dir, run, status="preparing", message=f"載入 Mask R-CNN · {device}", progress=3)
        for epoch in range(1, epochs + 1):
            model.train(); losses = []
            for images, targets, _assets in loader:
                if _stopping(run_dir):
                    return _status(run_dir, run, status="stopped", message="已安全停止", progress=None)
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                loss_values = model(images, targets); loss = sum(loss_values.values())
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
            score = _evaluate(model, validation, device, torch)
            row = {"epoch": epoch, "train/loss": round(sum(losses) / max(1, len(losses)), 6),
                   "val/mean_iou": score["mean_iou"]}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            _status(run_dir, run, status="running", message=f"Mask R-CNN {epoch} / {epochs}", epoch=epoch,
                    progress=5 + round(epoch / epochs * 85), metrics=row, device=str(device))
        checkpoint = model_dir / "checkpoint.pt"
        torch.save({"model_state": model.state_dict(), "classes": manifest["classes"], "image_size": image_size}, checkpoint)
        validation_result = _evaluate(model, validation, device, torch)
        test_result = _evaluate(model, testing, device, torch) if testing.assets else validation_result
        model_record = {"schema_version": 1, "engine": ENGINE_KEY, "engine_name": ENGINE_NAME,
                        "model_version_id": run["model_version_id"], "run_id": run["run_id"],
                        "dataset_version_id": manifest["dataset_version_id"], "classes": manifest["classes"],
                        "image_size": image_size, "score_threshold": 0.5, "validation": validation_result,
                        "test": test_result, "created_at": time.time(), "checkpoint": "checkpoint.pt"}
        atomic_json(model_dir / "model.json", model_record)
        atomic_json(run_dir / "evaluation.json", {"validation": validation_result, "test": test_result})
        artifacts = []
        for path in (metrics_path, run_dir / "evaluation.json", model_dir / "model.json", checkpoint):
            raw = path.read_bytes(); artifacts.append({"path": path.name, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)})
        atomic_json(run_dir / "artifact-manifest.json", {"schema_version": 1, "run_id": run["run_id"], "artifacts": artifacts})
        return _status(run_dir, run, status="completed", message="Mask R-CNN 訓練與評估完成", progress=100,
                       completed_at=time.time(), evaluation={"validation": validation_result, "test": test_result})
    except Exception as exc:
        _status(run_dir, run, status="failed", message=str(exc), error=str(exc), progress=None, completed_at=time.time())
        raise


class Predictor:
    def __init__(self, model_record, model_dir, requested_device="auto"):
        self.record = model_record
        self.model, self.torch = _model(len(model_record["classes"]), model_record.get("image_size"))
        self.device = _device(self.torch, requested_device)
        checkpoint = self.torch.load(Path(model_dir) / model_record["checkpoint"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"]); self.model.to(self.device); self.model.eval()

    def predict(self, rgb, width, height):
        image = self.torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().to(self.device) / 255.0
        with self.torch.inference_mode(): output = self.model([image])[0]
        shapes = []
        for score, label, mask in zip(output["scores"].detach().cpu().tolist(), output["labels"].detach().cpu().tolist(), output["masks"].detach().cpu().numpy()):
            if score < float(self.record.get("score_threshold", 0.5)) or not 1 <= label <= len(self.record["classes"]): continue
            bitmap = mask[0] >= 0.5; ys, xs = np.nonzero(bitmap)
            if not len(xs): continue
            shapes.append({"id": uuid.uuid4().hex, "type": "mask", "label": self.record["classes"][label - 1],
                           "x": 0, "y": 0, "width": width, "height": height, "counts": encode_rle(bitmap),
                           "metadata": {"source": "model_candidate", "engine": ENGINE_KEY,
                                        "model_version_id": self.record["model_version_id"], "confidence": round(float(score), 6),
                                        "bounds": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}})
        return shapes
