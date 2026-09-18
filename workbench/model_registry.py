"""Discoverable model catalog and bounded component installation plans.

The catalog is product data: entries stay visible even when their optional
runtime is absent.  Runtime state and Workbench integration state are kept
separate so the UI never promises that installing a package is sufficient for
an unfinished adapter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


TASK_NAMES = {
    "instance_segmentation": "實例分割",
    "object_detection": "物件偵測",
    "semantic_segmentation": "語意分割",
    "image_classification": "影像分類",
    "anomaly_detection": "異常檢測",
}

COMPONENTS = {
    "builtin": {
        "name": "工作台內建引擎", "installable": False,
        "description": "隨 Vision Workbench 提供，不需另外下載。",
    },
    "torchvision": {
        "name": "TorchVision 訓練環境", "installable": True,
        "description": "共用於 Mask R-CNN、Faster R-CNN、DeepLabV3 與 TorchVision 分類模型。",
        "requirements": "requirements-training.txt",
    },
    "anomalib": {
        "name": "Anomalib 異常檢測環境", "installable": False,
        "description": "EfficientAD 與 PatchCore 的獨立環境；待 Workbench 資料 adapter 完成後開放安裝。",
    },
    "ultralytics": {
        "name": "Ultralytics 偵測與分割環境", "installable": True,
        "description": "RT-DETR 與 YOLO26 Detect／Seg 的獨立環境；安裝前請確認 AGPL-3.0 或 Enterprise 授權。",
        "requirements": "requirements-ultralytics.txt",
    },
}


MODELS = (
    {"key": "pixel_prototype_v1", "name": "像素原型分割（內建基準）", "family": "內建基準",
     "task": "instance_segmentation", "component": "builtin", "integration": "ready",
     "description": "直接讀取原生像素遮罩，適合先驗證資料到模型的完整流程。",
     "annotation": "實例遮罩", "metrics": ["Mean IoU"], "license": "Vision Workbench · MIT"},
    {"key": "maskrcnn_resnet50_fpn", "name": "Mask R-CNN · ResNet50 FPN", "family": "Mask R-CNN",
     "task": "instance_segmentation", "component": "torchvision", "integration": "ready",
     "description": "精確的實例遮罩基準，直接讀取原生 RLE。", "annotation": "實例遮罩",
     "metrics": ["Mask IoU"], "license": "TorchVision；權重條件另依來源"},
    {"key": "fasterrcnn_mobilenet_v3_large_fpn", "name": "Faster R-CNN · MobileNet V3 FPN", "family": "Faster R-CNN",
     "task": "object_detection", "component": "torchvision", "integration": "ready",
     "description": "較省資源的物件偵測，從現有遮罩自動取得緊密框。", "annotation": "矩形框或實例遮罩",
     "metrics": ["Box mAP50", "Precision／Recall@0.5"], "license": "TorchVision；權重條件另依來源"},
    {"key": "fasterrcnn_mobilenet_v3_large_320_fpn", "name": "Faster R-CNN · MobileNet V3 320 FPN", "family": "Faster R-CNN",
     "task": "object_detection", "component": "torchvision", "integration": "ready",
     "description": "低解析度快速偵測變體，適合先測試 CPU 或小型 GPU。", "annotation": "矩形框或實例遮罩",
     "metrics": ["Box mAP50", "Precision／Recall@0.5"], "license": "TorchVision；權重條件另依來源"},
    {"key": "fasterrcnn_resnet50_fpn_v2", "name": "Faster R-CNN · ResNet50 FPN V2", "family": "Faster R-CNN",
     "task": "object_detection", "component": "torchvision", "integration": "ready",
     "description": "較高容量的物件偵測模型，適合精度比較。", "annotation": "矩形框或實例遮罩",
     "metrics": ["Box mAP50", "Precision／Recall@0.5"], "license": "TorchVision；權重條件另依來源"},
    {"key": "deeplabv3_mobilenet_v3_large", "name": "DeepLabV3 · MobileNet V3", "family": "DeepLabV3",
     "task": "semantic_segmentation", "component": "torchvision", "integration": "ready",
     "description": "將同類實例合併為像素類別圖，兼顧訓練速度與分割能力。", "annotation": "面積標註",
     "metrics": ["mIoU", "Dice"], "license": "TorchVision；權重條件另依來源"},
    {"key": "deeplabv3_resnet50", "name": "DeepLabV3 · ResNet50", "family": "DeepLabV3",
     "task": "semantic_segmentation", "component": "torchvision", "integration": "ready",
     "description": "較高容量的語意分割變體。", "annotation": "面積標註",
     "metrics": ["mIoU", "Dice"], "license": "TorchVision；權重條件另依來源"},
    {"key": "mobilenet_v3_large_classification", "name": "MobileNet V3 · 分類", "family": "TorchVision Classification",
     "task": "image_classification", "component": "torchvision", "integration": "ready", "predict": False,
     "description": "適合 OK／NG 或整張圖片狀態；每張核准圖片需只有一個標註類別。", "annotation": "單一圖片類別",
     "metrics": ["Macro F1", "Recall"], "license": "TorchVision；權重條件另依來源"},
    {"key": "efficientnet_b0_classification", "name": "EfficientNet-B0 · 分類", "family": "TorchVision Classification",
     "task": "image_classification", "component": "torchvision", "integration": "ready", "predict": False,
     "description": "影像分類的平衡變體；每張核准圖片需只有一個標註類別。", "annotation": "單一圖片類別",
     "metrics": ["Macro F1", "Recall"], "license": "TorchVision；權重條件另依來源"},
    {"key": "resnet18_classification", "name": "ResNet18 · 分類", "family": "TorchVision Classification",
     "task": "image_classification", "component": "torchvision", "integration": "ready", "predict": False,
     "description": "穩定的分類基準；每張核准圖片需只有一個標註類別。", "annotation": "單一圖片類別",
     "metrics": ["Macro F1", "Recall"], "license": "TorchVision；權重條件另依來源"},
    {"key": "efficientad", "name": "EfficientAD", "family": "Anomalib", "task": "anomaly_detection",
     "component": "anomalib", "integration": "planned", "description": "正常樣本為主的快速異常檢測與熱圖。",
     "annotation": "正常／異常／未知", "metrics": ["Image AUROC", "Pixel AUROC"], "license": "Anomalib · Apache-2.0"},
    {"key": "patchcore", "name": "PatchCore", "family": "Anomalib", "task": "anomaly_detection",
     "component": "anomalib", "integration": "planned", "description": "少量缺陷資料的異常檢測基準。",
     "annotation": "正常／異常／未知", "metrics": ["Image AUROC", "AUPRO"], "license": "Anomalib · Apache-2.0"},
    {"key": "rt_detr_r50", "name": "RT-DETR · ResNet50", "family": "RT-DETR", "task": "object_detection",
     "component": "ultralytics", "integration": "ready", "description": "以套件內建架構從零訓練，不會自動下載預訓練權重。",
     "annotation": "矩形框或實例遮罩", "metrics": ["Box mAP50–95"], "license": "Ultralytics · AGPL-3.0 或 Enterprise"},
    {"key": "yolo26n_seg", "name": "YOLO26n Seg", "family": "YOLO26 Seg", "task": "instance_segmentation",
     "component": "ultralytics", "integration": "ready", "description": "YOLO26 nano 實例分割；可嚴格檢查，或只在 Run 副本修補通過門檻的微小封閉孔洞。",
     "annotation": "實例遮罩／多邊形", "metrics": ["Mask mAP50–95"], "license": "Ultralytics · AGPL-3.0 或 Enterprise"},
    {"key": "yolo26s_seg", "name": "YOLO26s Seg", "family": "YOLO26 Seg", "task": "instance_segmentation",
     "component": "ultralytics", "integration": "ready", "description": "YOLO26 small 實例分割；可嚴格檢查，或只在 Run 副本修補通過門檻的微小封閉孔洞。",
     "annotation": "實例遮罩／多邊形", "metrics": ["Mask mAP50–95"], "license": "Ultralytics · AGPL-3.0 或 Enterprise"},
    {"key": "yolo26n_detect", "name": "YOLO26n Detect", "family": "YOLO26 Detect", "task": "object_detection",
     "component": "ultralytics", "integration": "ready", "description": "YOLO26 nano 物件偵測；由矩形框或面積標註自動取得緊密框。",
     "annotation": "矩形框或實例遮罩", "metrics": ["Box mAP50–95"], "license": "Ultralytics · AGPL-3.0 或 Enterprise"},
    {"key": "yolo26s_detect", "name": "YOLO26s Detect", "family": "YOLO26 Detect", "task": "object_detection",
     "component": "ultralytics", "integration": "ready", "description": "YOLO26 small 物件偵測；容量較高，適合與 nano 比較精度。",
     "annotation": "矩形框或實例遮罩", "metrics": ["Box mAP50–95"], "license": "Ultralytics · AGPL-3.0 或 Enterprise"},
)


MODELS += tuple({"key": key, "name": name, "family": family, "task": task,
                 "component": "ultralytics", "integration": "ready", "inference_only": True,
                 "description": "匯入外部 .pt 權重後用於試跑與預標註，不代表已在本專案訓練或評估。",
                 "annotation": "外部模型類別", "metrics": [], "license": "依來源權重及 Ultralytics 授權"}
                for key, name, family, task in (
                    ("external_yolo_detect", "外部 YOLO · 偵測", "YOLO External", "object_detection"),
                    ("external_yolo_segment", "外部 YOLO · 分割", "YOLO External", "instance_segmentation"),
                    ("rt_detr_external", "外部 RT-DETR", "RT-DETR", "object_detection")))


class ModelRegistry:
    def __init__(self, app_root: Path, python_executable=None):
        self.app_root = Path(app_root).resolve()
        self._configured_python = python_executable
        self._lock = threading.RLock()
        self._probe_cache = None
        self._probe_time = 0.0
        self._probe_refreshing = False

    @property
    def training_python(self) -> Path:
        configured = os.environ.get("VISION_WORKBENCH_TRAINING_PYTHON") or self._configured_python
        isolated = self.app_root / ".venv-training" / "Scripts" / "python.exe"
        return Path(configured or (isolated if isolated.is_file() else sys.executable)).resolve()

    def component_python(self, component_id: str) -> Path:
        if component_id == "torchvision":
            return self.training_python
        if component_id == "builtin":
            return Path(sys.executable).resolve()
        return (self.app_root / ".venv-models" / component_id / "Scripts" / "python.exe").resolve()

    def _probe_torchvision(self):
        python = self.training_python
        isolated = self.app_root / ".venv-training" / "Scripts" / "python.exe"
        dedicated = bool(os.environ.get("VISION_WORKBENCH_TRAINING_PYTHON") or self._configured_python or isolated.is_file())
        code = ("import json,torch,torchvision; print(json.dumps({"
                "'torch':torch.__version__,'torchvision':torchvision.__version__,"
                "'cuda':bool(torch.cuda.is_available()),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}))")
        try:
            result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=30,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode:
                state = "broken" if dedicated else "not_installed"
                return {"state": state,
                        "message": "TorchVision 無法載入，請執行修復" if dedicated else "尚未安裝 TorchVision 獨立訓練環境",
                        "details": (result.stderr.strip().splitlines() or ["尚未安裝"])[-1], "python": str(python)}
            details = json.loads(result.stdout.strip().splitlines()[-1])
            return {"state": "ready", "message": "TorchVision 訓練環境可用", "python": str(python), **details}
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            state = "broken" if dedicated else "not_installed"
            return {"state": state, "message": "TorchVision 無法載入，請執行修復" if dedicated else "尚未安裝 TorchVision 獨立訓練環境",
                    "details": str(exc), "python": str(python)}

    def _probe_python_component(self, component_id, imports):
        python = self.component_python(component_id)
        if not python.is_file():
            return {"state": "not_installed", "message": f"尚未安裝 {COMPONENTS[component_id]['name']}",
                    "python": str(python)}
        versions = ",".join(repr(name) + ":getattr(" + name + ",'__version__','unknown')" for name in imports)
        code = "import json," + ",".join(imports) + "; print(json.dumps({'versions': {" + versions + "}}))"
        try:
            result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=30,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode:
                return {"state": "broken", "message": f"{COMPONENTS[component_id]['name']} 無法載入，請執行修復",
                        "details": (result.stderr.strip().splitlines() or ["載入失敗"])[-1], "python": str(python)}
            details = json.loads(result.stdout.strip().splitlines()[-1])
            return {"state": "ready", "message": f"{COMPONENTS[component_id]['name']}可用",
                    "python": str(python), **details}
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            return {"state": "broken", "message": f"{COMPONENTS[component_id]['name']}無法載入，請執行修復",
                    "details": str(exc), "python": str(python)}

    def component_status(self, refresh=False):
        with self._lock:
            if self._probe_cache is not None and not refresh:
                if time.monotonic() - self._probe_time > 300 and not self._probe_refreshing:
                    self._probe_refreshing = True
                    threading.Thread(target=self._background_probe, name='runtime-probe', daemon=True).start()
                return self._probe_cache
        torchvision = self._probe_torchvision()
        states = {
                "builtin": {"state": "ready", "message": "隨工作台提供", "python": sys.executable},
                "torchvision": torchvision,
                "anomalib": {"state": "planned", "message": "資料 adapter 與獨立 runtime 尚在開發"},
                "ultralytics": self._probe_python_component("ultralytics", ("torch", "ultralytics")),
        }
        with self._lock:
            self._probe_cache, self._probe_time = states, time.monotonic()
            return states

    def _background_probe(self):
        try:
            self.component_status(refresh=True)
        finally:
            with self._lock:
                self._probe_refreshing = False

    def snapshot(self, refresh=False):
        states = self.component_status(refresh)
        components = []
        for key, value in COMPONENTS.items():
            components.append({"id": key, **value, **states[key]})
        models = []
        for definition in MODELS:
            component = states[definition["component"]]
            train = definition["integration"] == "ready" and component["state"] == "ready"
            reason = "" if train else ("Workbench adapter 尚在開發" if definition["integration"] != "ready" else component["message"])
            models.append({**definition, "task_name": TASK_NAMES[definition["task"]],
                           "runtime_state": component["state"], "train": train and not definition.get("inference_only"), "evaluate": train and not definition.get("inference_only"),
                           "predict": train and definition.get("predict", True), "export": train,
                           "unavailable_reason": reason})
        return {"models": models, "components": components, "tasks": TASK_NAMES,
                "worker_python": str(self.training_python), "refreshed_at": time.time()}

    def model(self, key, refresh=False):
        return next((item for item in self.snapshot(refresh)["models"] if item["key"] == key), None)

    def install(self, component_id, progress):
        from .process_control import run_controlled, installation_options
        if component_id not in COMPONENTS:
            raise FileNotFoundError("找不到模型元件")
        component = COMPONENTS[component_id]
        if not component.get("installable"):
            raise ValueError(component["description"])
        progress(f"準備 {component['name']}", 5)
        if component_id == "torchvision":
            command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                       str(self.app_root / "bootstrap.ps1"), "-Training", "-TrainingOnly"]
        else:
            target = self.component_python(component_id).parent.parent
            target.parent.mkdir(parents=True, exist_ok=True)
            if not self.component_python(component_id).is_file():
                run_controlled([sys.executable, '-m', 'venv', str(target)], progress, cwd=self.app_root,
                               **installation_options())
            command = [str(self.component_python(component_id)), "-m", "pip", "install", "-r",
                       str(self.app_root / component["requirements"])]
        run_controlled(command, progress, cwd=self.app_root, **installation_options())
        self._probe_cache = None
        status = self.component_status(refresh=True)[component_id]
        if status["state"] != "ready":
            raise RuntimeError(status["message"])
        progress(f"{component['name']}已驗證", 100)
        return {"component_id": component_id, "status": status, "catalog": self.snapshot(refresh=True)}
