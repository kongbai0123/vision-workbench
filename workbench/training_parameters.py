"""The parameters each training adapter actually consumes and their API validation."""
from __future__ import annotations

import math


def parameter_schema(definition):
    if definition.get("integration", "ready") != "ready":
        return []
    baseline = definition["key"] == "pixel_prototype_v1"
    parameters = [{"key": "epochs", "label": "門檻搜尋次數" if baseline else "訓練輪數",
                   "type": "integer", "default": 24, "min": 1, "max": 200, "step": 1,
                   "advanced": False, "description": "均勻搜尋門檻候選值。" if baseline else "完整讀取 Train 資料的次數。"}]
    if baseline:
        return parameters + [
            {"key": "threshold_min", "label": "門檻搜尋下限", "type": "number", "default": .35,
             "min": 1e-8, "max": 100, "step": .05, "advanced": True,
             "description": "像素原型的距離門檻下限，必須小於上限；此引擎固定使用 CPU。"},
            {"key": "threshold_max", "label": "門檻搜尋上限", "type": "number", "default": 4.,
             "min": 1e-8, "max": 100, "step": .05, "advanced": True,
             "description": "在上下限之間搜尋最佳距離門檻；不使用梯度下降。"},
        ]
    ultralytics = definition.get("component") == "ultralytics"
    parameters += [
        {"key": "seed", "label": "隨機種子", "type": "integer", "default": 42,
         "min": 0, "max": 2147483647, "step": 1, "advanced": False,
         "description": "固定初始化與抽樣種子；GPU 運算仍可能有差異。"},
        {"key": "device", "label": "執行裝置", "type": "select", "default": "auto",
         "options": [{"value": "auto", "label": "自動選擇 GPU / CPU"},
                     {"value": "cpu", "label": "CPU"}, {"value": "cuda", "label": "CUDA GPU"}],
         "advanced": False, "description": "CUDA 模式需要可用的 NVIDIA GPU 環境。"},
        {"key": "image_size", "label": "訓練影像尺寸", "type": "integer",
         "default": 224 if definition.get("task") == "image_classification" else 640,
         "min": 128, "max": 2048, "step": 32 if ultralytics else 1,
         **({"multiple_of": 32} if ultralytics else {}), "advanced": True,
         "description": "模型輸入尺寸；Ultralytics 需要 32 的倍數。" if ultralytics else "模型內部縮放尺寸；不會修改專案原始圖片。"},
        {"key": "batch_size", "label": "批次大小", "type": "integer", "default": 1,
         "min": 1, "max": 16, "step": 1, "advanced": True,
         "description": "每次參數更新使用的圖片數；較大數值會增加記憶體用量。"
                        + ("單張批次使用 BatchNorm 既有統計值。" if definition.get("task") == "semantic_segmentation" else "")},
        {"key": "learning_rate", "label": "學習率", "type": "number", "default": .0005,
         "min": 1e-8, "max": 1, "step": .0001, "advanced": True,
         "description": "初始學習率；必須大於 0。"},
        {"key": "weight_decay", "label": "權重衰減", "type": "number", "default": .0001,
         "min": 0, "max": 1, "step": .0001, "advanced": True,
         "description": "最佳化器的權重衰減係數；0 表示不使用。"},
        {"key": "optimizer", "label": "最佳化器", "type": "select", "default": "AdamW",
         "options": [{"value": "AdamW", "label": "AdamW"}, {"value": "SGD", "label": "SGD"}],
         "advanced": True, "description": "SGD 使用固定動量 0.9（Nesterov）。" if ultralytics else "選擇參數更新演算法；SGD 不使用動量。"},
    ]
    strategies = [("fixed", "固定"), ("cosine", "暖身＋餘弦下降"), ("linear", "暖身＋線性下降")]
    if not ultralytics: strategies.append(("plateau", "Validation 停滯時下降"))
    parameters += [
        {"key":"scheduler", "label":"學習率策略", "type":"select", "default":"cosine", "advanced":True,
         "section":"schedule", "options":[{"value":v,"label":n} for v,n in strategies],
         "description":"固定不進行暖身或衰減；其他策略依訓練進度或驗證結果調整。"},
        {"key":"min_learning_rate", "label":"最低學習率", "type":"number", "default":.000005,
         "min":1e-12, "max":1, "advanced":True, "section":"schedule", "when":["cosine","linear","plateau"]},
        {"key":"warmup_epochs", "label":"暖身輪數", "type":"integer", "default":0,
         "min":0, "max":199, "advanced":True, "section":"schedule", "when":["cosine","linear"],
         "description":"0 表示不暖身；必須小於總訓練輪數。"},
    ]
    if not ultralytics:
        parameters += [
            {"key":"lr_patience", "label":"允許未改善輪數", "type":"integer", "default":3,
             "min":0, "max":199, "advanced":True, "section":"schedule", "when":["plateau"]},
            {"key":"lr_factor", "label":"下降倍率", "type":"number", "default":.5,
             "min":.001, "max":.999, "advanced":True, "section":"schedule", "when":["plateau"],
             "description":"監控 Validation 主要分數（越高越好），不使用 Test 調整。"},
        ]
    return parameters


def validate_config(definition, supplied):
    if not isinstance(supplied, dict):
        raise ValueError("訓練參數必須是物件")
    supplied = dict(supplied)
    if definition["key"] == "pixel_prototype_v1":
        # Versions before model-specific forms submitted these ignored defaults.
        legacy_keys = {"seed", "device", "image_size", "batch_size", "learning_rate"}
        legacy = {key: supplied[key] for key in legacy_keys if key in supplied}
        if legacy:
            validate_config({**definition, "key": "legacy_baseline"}, legacy)
            if legacy.get("device") == "cuda":
                raise ValueError("像素原型引擎固定使用 CPU")
            for key in legacy:
                supplied.pop(key)
    schema = parameter_schema(definition)
    if definition["key"] != "pixel_prototype_v1" and "min_learning_rate" not in supplied:
        rate = supplied.get("learning_rate", .0005)
        if type(rate) in (int, float): supplied["min_learning_rate"] = rate * .01
    allowed = {field["key"] for field in schema} | {"engine"}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError("此模型不支援參數：" + "、".join(sorted(unknown)))
    result = {}
    for field in schema:
        key, label = field["key"], field["label"]
        value = supplied.get(key, field["default"])
        if field["type"] == "select":
            if not isinstance(value, str) or value not in {option["value"] for option in field["options"]}:
                raise ValueError(f"{label}不是支援的選項")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label}必須是數值")
            try:
                finite = math.isfinite(value)
            except (ValueError, OverflowError):
                finite = False
            if not finite:
                raise ValueError(f"{label}必須是有限數值")
            if field["type"] == "integer" and value != int(value):
                raise ValueError(f"{label}必須是整數")
            if value < field["min"] or value > field["max"] or field.get("exclusive_min") and value == field["min"]:
                relation = "大於" if field.get("exclusive_min") else "至少"
                raise ValueError(f"{label}必須{relation} {field['min']}，且不超過 {field['max']}")
            if field.get("multiple_of") and value % field["multiple_of"]:
                raise ValueError(f"{label}必須是 {field['multiple_of']} 的倍數")
            value = int(value) if field["type"] == "integer" else float(value)
        result[key] = value
    if definition["key"] == "pixel_prototype_v1":
        if result["threshold_min"] >= result["threshold_max"]:
            raise ValueError("門檻搜尋下限必須小於上限")
        result["device"] = "cpu"
    elif definition.get("component") == "ultralytics" and result.get("optimizer") == "SGD":
        # Ultralytics builds Nesterov SGD, which requires positive momentum.
        result["momentum"] = .9
    if not definition["key"] == "pixel_prototype_v1":
        schedule = result["scheduler"]
        if schedule != "fixed" and result["min_learning_rate"] > result["learning_rate"]:
            raise ValueError("最低學習率不得大於初始學習率")
        if schedule in {"cosine","linear"} and result["warmup_epochs"] >= result["epochs"]:
            raise ValueError("暖身輪數必須小於總訓練輪數")
        if schedule == "fixed":
            result["min_learning_rate"] = result["learning_rate"]
        if schedule not in {"cosine","linear"}: result["warmup_epochs"] = 0
        if schedule != "plateau":
            result.pop("lr_patience", None); result.pop("lr_factor", None)
    return result


def create_optimizer(torch, parameters, config):
    name = config.get("optimizer", "AdamW")
    if name not in {"AdamW", "SGD"}:
        raise ValueError("不支援的最佳化器")
    return getattr(torch.optim, name)(parameters, lr=float(config.get("learning_rate", .0005)),
                                      weight_decay=float(config.get("weight_decay", .0001)))
