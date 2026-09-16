"""Validated, versioned training-only data augmentation profiles."""
from __future__ import annotations

from copy import deepcopy
import random


_PRESETS = {
    "off": {
        "brightness": 0.0, "contrast": 0.0, "fliplr": 0.0, "flipud": 0.0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.0, "mosaic": 0.0,
        "mixup": 0.0, "copy_paste": 0.0, "close_mosaic": 0,
    },
    "light": {
        "brightness": 0.10, "contrast": 0.10, "fliplr": 0.50, "flipud": 0.0,
        "degrees": 2.0, "translate": 0.03, "scale": 0.10, "mosaic": 0.0,
        "mixup": 0.0, "copy_paste": 0.0, "close_mosaic": 0,
    },
    "standard": {
        "brightness": 0.20, "contrast": 0.20, "fliplr": 0.50, "flipud": 0.0,
        "degrees": 0.0, "translate": 0.10, "scale": 0.50, "mosaic": 1.0,
        "mixup": 0.0, "copy_paste": 0.0, "close_mosaic": 10,
    },
}

_RANGES = {
    "brightness": (0.0, 0.5), "contrast": (0.0, 0.5),
    "fliplr": (0.0, 1.0), "flipud": (0.0, 1.0),
    "degrees": (0.0, 45.0), "translate": (0.0, 0.5), "scale": (0.0, 0.9),
    "mosaic": (0.0, 1.0), "mixup": (0.0, 1.0), "copy_paste": (0.0, 1.0),
}


def augmentation_presets():
    return deepcopy(_PRESETS)


def normalize_augmentation(value=None):
    """Return a strict, serializable profile. Unknown keys are rejected."""
    if value is None:
        value = {"preset": "off"}
    if not isinstance(value, dict):
        raise ValueError("資料增強設定必須是物件")
    preset = str(value.get("preset") or "off")
    if preset not in {*_PRESETS, "custom"}:
        raise ValueError("資料增強預設值無效")
    allowed = {"preset", "schema_version", "apply_to", "mode", *_RANGES, "close_mosaic"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError("不支援的資料增強欄位：" + "、".join(sorted(unknown)))
    base = deepcopy(_PRESETS["light"] if preset == "custom" else _PRESETS[preset])
    for key in _RANGES:
        if key not in value:
            continue
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"資料增強 {key} 必須是數值")
        low, high = _RANGES[key]
        if not low <= float(raw) <= high:
            raise ValueError(f"資料增強 {key} 必須介於 {low} 與 {high} 之間")
        base[key] = float(raw)
    close_mosaic = value.get("close_mosaic", base["close_mosaic"])
    if isinstance(close_mosaic, bool) or not isinstance(close_mosaic, int) or not 0 <= close_mosaic <= 200:
        raise ValueError("close_mosaic 必須是 0 到 200 的整數")
    base["close_mosaic"] = close_mosaic
    return {"schema_version": 1, "preset": preset, "apply_to": "train", "mode": "online", **base}


def yolo_augmentation_args(profile):
    profile = normalize_augmentation(profile)
    # Ultralytics separates HSV value jitter from brightness/contrast.  Use the
    # versioned brightness strength for HSV value and keep contrast available
    # to the native TorchVision adapters.
    return {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": profile["brightness"],
        "degrees": profile["degrees"], "translate": profile["translate"],
        "scale": profile["scale"], "fliplr": profile["fliplr"],
        "flipud": profile["flipud"], "mosaic": profile["mosaic"],
        "mixup": profile["mixup"], "copy_paste": profile["copy_paste"],
        "close_mosaic": profile["close_mosaic"],
    }


def augment_image_tensor(image, profile, torch):
    """Apply common photometric transforms and return flip decisions."""
    profile = normalize_augmentation(profile)
    horizontal = random.random() < profile["fliplr"]
    vertical = random.random() < profile["flipud"]
    if horizontal:
        image = torch.flip(image, (-1,))
    if vertical:
        image = torch.flip(image, (-2,))
    brightness = profile["brightness"]
    if brightness:
        image = image * random.uniform(1.0 - brightness, 1.0 + brightness)
    contrast = profile["contrast"]
    if contrast:
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = mean + (image - mean) * random.uniform(1.0 - contrast, 1.0 + contrast)
    return image.clamp(0.0, 1.0), horizontal, vertical


def augment_dense_target(image, target, profile, torch):
    """Synchronize flips for semantic maps or detection masks and boxes."""
    image, horizontal, vertical = augment_image_tensor(image, profile, torch)
    if not (horizontal or vertical):
        return image, target
    if isinstance(target, dict):
        height, width = image.shape[-2:]
        masks = target.get("masks")
        if masks is not None:
            if horizontal:
                masks = torch.flip(masks, (-1,))
            if vertical:
                masks = torch.flip(masks, (-2,))
            target["masks"] = masks
        boxes = target.get("boxes")
        if boxes is not None and boxes.numel():
            boxes = boxes.clone()
            if horizontal:
                left, right = boxes[:, 0].clone(), boxes[:, 2].clone()
                boxes[:, 0], boxes[:, 2] = width - right, width - left
            if vertical:
                top, bottom = boxes[:, 1].clone(), boxes[:, 3].clone()
                boxes[:, 1], boxes[:, 3] = height - bottom, height - top
            target["boxes"] = boxes
    else:
        if horizontal:
            target = torch.flip(target, (-1,))
        if vertical:
            target = torch.flip(target, (-2,))
    return image, target
