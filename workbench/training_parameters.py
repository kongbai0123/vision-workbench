"""The parameters each training adapter actually consumes and their API validation."""
from __future__ import annotations

import math


def parameter_schema(definition):
    if definition.get("integration", "ready") != "ready" or definition.get("inference_only"):
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
         "min": 0, "max": 2147483647, "step": 1, "advanced": True,
         "description": "固定初始化與抽樣種子；GPU 運算仍可能有差異。"},
        {"key": "device", "label": "執行裝置", "type": "select", "default": "auto",
         "options": [{"value": "auto", "label": "自動選擇 GPU / CPU"},
                     {"value": "cpu", "label": "CPU"}, {"value": "cuda", "label": "CUDA GPU"}],
         "advanced": False, "description": "CUDA 模式需要可用的 NVIDIA GPU 環境。"},
        {"key": "image_size", "label": "訓練影像尺寸", "type": "integer",
         "default": 224 if definition.get("task") == "image_classification" else 640,
         "min": 128, "max": 2048, "step": 32 if ultralytics else 1,
         **({"multiple_of": 32} if ultralytics else {}), "advanced": False,
         "description": "模型輸入尺寸；Ultralytics 需要 32 的倍數。" if ultralytics else "模型內部縮放尺寸；不會修改專案原始圖片。"},
        {"key": "batch_size", "label": "批次大小", "type": "integer", "default": 1,
         "min": 1, "max": 16, "step": 1, "advanced": False,
         "description": "每個小批次讀取的圖片數；梯度累積會影響實際更新頻率。較大數值會增加記憶體用量。"
                        + ("單張批次使用 BatchNorm 既有統計值。" if definition.get("task") == "semantic_segmentation" else "")},
        {"key": "learning_rate", "label": "學習率", "type": "number", "default": .0005,
         "min": 1e-8, "max": 1, "step": .0001, "advanced": True,
         "description": "初始學習率；必須大於 0。"},
        {"key": "weight_decay", "label": "權重衰減", "type": "number", "default": .0001,
         "min": 0, "max": 1, "step": .0001, "advanced": True,
         "description": "最佳化器的權重衰減係數；0 表示不使用。"},
        {"key": "optimizer", "label": "最佳化器", "type": "select", "default": "AdamW",
         "options": [{"value": "AdamW", "label": "AdamW"}, {"value": "SGD", "label": "SGD"}],
         "advanced": True, "description": "選擇參數更新演算法；SGD 使用動量設定，AdamW 使用自適應更新與解耦權重衰減。"},
    ]
    if ultralytics:
        parameters.append({"key": "gradient_accumulation", "label": "梯度累積批次數", "type": "integer",
                           "default": 1, "min": 1, "max": 64, "step": 1, "advanced": True,
                           "description": "累積幾個小批次後更新權重；有效批次＝批次大小 × 累積批次數。小資料集建議從 1 開始。"})
    if definition["key"].startswith("yolo26"):
        parameters.append({"key": "initialization", "label": "模型初始權重", "type": "select",
                           "default": "pretrained", "advanced": False,
                           "options": [{"value": "pretrained", "label": "預訓練權重微調（建議）"},
                                       {"value": "scratch", "label": "隨機權重 · 從零訓練"}],
                           "description": "首次開始預訓練微調時會下載官方權重，之後使用本機快取；選取選項本身不下載。"})
    if definition.get('task') == 'instance_segmentation' and definition['key'].startswith('yolo26'):
        repair_only = {"key": "yolo_mask_policy", "values": ["repair_tiny_holes"]}
        parameters += [
            {"key": "yolo_mask_policy", "label": "YOLO Seg 遮罩相容方式", "type": "select",
             "default": "repair_tiny_holes", "advanced": True, "section": "compatibility",
             "options": [{"value": "repair_tiny_holes", "label": "修補微小封閉孔洞（Run 副本）"},
                         {"value": "strict", "label": "嚴格無損（偵測孔洞即阻擋）"}],
             "description": "只修改本次 Run 的 YOLO 訓練副本；專案 Mask 與固定資料版本不變。"},
            {"key": "tiny_hole_max_pixels", "label": "單一孔洞上限（px）", "type": "integer",
             "default": 16, "min": 1, "max": 64, "step": 1, "advanced": True,
             "section": "compatibility", "depends_on": repair_only,
             "description": "單一封閉孔洞超過此面積仍會阻擋訓練。"},
            {"key": "tiny_hole_total_pixels", "label": "單一實例修補總上限（px）", "type": "integer",
             "default": 16, "min": 1, "max": 128, "step": 1, "advanced": True,
             "section": "compatibility", "depends_on": repair_only},
            {"key": "tiny_hole_max_ratio", "label": "修補占 Mask 比例上限", "type": "number",
             "default": .0001, "min": .000001, "max": .01, "step": .00001, "advanced": True,
             "section": "compatibility", "depends_on": repair_only,
             "description": "0.0001 等於 0.01%；像素與比例門檻必須同時通過。"},
            {"key": "tiny_hole_max_dimension", "label": "孔洞寬／高上限（px）", "type": "integer",
             "default": 16, "min": 1, "max": 64, "step": 1, "advanced": True,
             "section": "compatibility", "depends_on": repair_only,
             "description": "避免細長裂縫被當成雜點填補。"},
        ]
    strategies = [("fixed", "固定"), ("cosine", "暖身＋餘弦下降"), ("linear", "暖身＋線性下降")]
    if not ultralytics: strategies.append(("plateau", "Validation 停滯時下降"))
    parameters += [
        {"key":"scheduler", "label":"學習率排程", "type":"select", "default":"cosine", "advanced":True,
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
    if definition.get('task') == 'instance_segmentation' and definition['key'].startswith('yolo26'):
        if result["tiny_hole_total_pixels"] < result["tiny_hole_max_pixels"]:
            raise ValueError("單一實例修補總上限不得小於單一孔洞上限")
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
