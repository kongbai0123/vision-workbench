"""Dataset versions, training runs, model versions, and prediction candidates."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import numpy as np
from PIL import Image

from composer_core.geometry import bounds, decode_rle
from .store import dump, timestamp
from .training_engine import ENGINE_KEY, ENGINE_NAME, atomic_json, predict, read_json
from .maskrcnn_engine import ENGINE_KEY as MASKRCNN_KEY, ENGINE_NAME as MASKRCNN_NAME
from .model_registry import ModelRegistry
from .torchvision_engines import DETECTION_ENGINES, SEMANTIC_ENGINES
from .classification_engine import CLASSIFICATION_ENGINES
from .ultralytics_engine import ULTRALYTICS_ENGINES
from .training_parameters import parameter_schema, validate_config
from .yolo_compatibility import analyze_manifest, blocker_message
from .split_quality import split_class_coverage, loose_split_applies, split_purpose
from .project_storage import ProjectStorage
from .augmentation import augmentation_event_layout, normalize_augmentation


AREA_SHAPES = {"mask", "polygon", "obb", "rectangle"}


def _next_id(parent: Path, prefix: str) -> str:
    used = []
    if parent.is_dir():
        for item in parent.iterdir():
            if item.is_dir() and item.name.startswith(prefix) and item.name[len(prefix):].isdigit():
                used.append(int(item.name[len(prefix):]))
    return f"{prefix}{max(used, default=0) + 1:03d}"


def _canonical_hash(value) -> str:
    return sha256(dump(value).encode("utf-8")).hexdigest()


def _safe_id(value, prefix):
    if not isinstance(value, str) or not value.startswith(prefix) or not value[len(prefix):].isdigit():
        raise ValueError(f"{prefix} 版本 ID 無效")
    return value


def _link_or_copy(source: Path, destination: Path) -> None:
    """Publish immutable dataset bytes efficiently on the same volume."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def dataset_readiness(manifest):
    assets = manifest.get("assets", [])
    active = {a.get("split") for a in assets}
    purpose = split_purpose(manifest.get('split_plan'), assets)
    report = split_class_coverage(assets, active_splits=active, policy=purpose)
    if not assets:
        report["blockers"].append({"code": "empty_dataset", "message": "固定資料版本沒有圖片", "action": "建立有效資料版本"})
    if not report["image_counts"]["train"]:
        report["blockers"].append({"code": "no_train_split", "message": "Train 沒有圖片", "action": "設定資料分割"})
    if not report["image_counts"]["val"] and purpose != 'all_train':
        report["blockers"].append({"code": "no_validation_split", "message": "Validation 沒有圖片，不能安全選模或調整門檻",
                                   "action": "設定獨立的 Validation；Test 僅供最後評估，不會替代 Validation"})
    if any(a.get("split") not in {"train", "val", "test"} for a in assets):
        report["blockers"].append({"code": "missing_split", "message": "固定資料版本含有未分割圖片", "action": "重新建立資料版本"})
    duplicate_splits = {}
    for asset in assets:
        if asset.get('sha256'):
            duplicate_splits.setdefault(asset['sha256'], set()).add(asset.get('split'))
    leaked_duplicates = [sha for sha, splits in duplicate_splits.items() if len(splits) > 1]
    if leaked_duplicates:
        report['blockers'].append({'code': 'duplicate_split_leak',
                                   'message': f'{len(leaked_duplicates)} 組完全相同圖片跨越不同集合',
                                   'action': '將相同 SHA-256 圖片移到同一集合後重新建立資料版本'})
    report["ready"] = not report["blockers"]
    report['purpose'] = purpose
    report["stats"] = {"approved": len(assets), "splits": report["image_counts"], "classes": report["class_totals"]}
    return report


def _with_evaluation_reassessment(record, directory, *, model=False):
    """Overlay a versioned reassessment while preserving the historical record."""
    path = Path(directory) / "evaluation.v2.json"
    if not path.is_file():
        return record
    reassessment = read_json(path)
    updated = dict(record)
    updated["evaluation_reassessment"] = {
        "schema_version": reassessment.get("schema_version"),
        "created_at": reassessment.get("created_at"),
        "reason": reassessment.get("reason"),
        "valid": reassessment.get("valid", True),
    }
    if model:
        updated["legacy_evaluation"] = {"validation": record.get("validation"), "test": record.get("test")}
        updated["validation"] = reassessment.get("validation")
        updated["test"] = reassessment.get("test")
        updated["evaluation_protocol"] = reassessment.get("protocol")
    else:
        updated["legacy_evaluation"] = record.get("evaluation")
        updated["evaluation"] = reassessment
    return updated


class TrainingWorkspace:
    """Own training data below one Workbench data root.

    ProjectStore remains the only writer of project annotations.  This class
    creates immutable copies and persists all long-running run state as files.
    """
    def __init__(self, data_root: Path, store, python_executable=None):
        self.root = Path(data_root).resolve()
        self.store = store
        self.storage = ProjectStorage(self.root, store)
        self._project_scoped = True
        self.datasets = self.root / "datasets"
        self.runs = self.root / "runs"
        self.models = self.root / "models"
        self.model_exports = self.root / "model-exports"
        self.predictions = self.root / "predictions"
        self.migration_reports = self.storage.migrate()
        self.registry = ModelRegistry(Path(__file__).resolve().parents[1], python_executable)
        self.python = str(self.registry.training_python)
        self.lock = threading.RLock()
        self.processes = {}
        self._review_yolo_cache = {}
        self._maskrcnn_available = self._probe_maskrcnn()
        self._reconcile_runs()
        self.catalog_issues = {}
        for project in self.store.list_projects():
            self._sync_catalog(project["id"])

    def _area(self, project_id, name, *, create=False):
        if getattr(self, "_project_scoped", False):
            return self.storage.area(project_id, name, create=create)
        legacy = getattr(self, name.replace("-", "_")) / project_id
        if create:
            legacy.mkdir(parents=True, exist_ok=True)
        return legacy

    def datasets_dir(self, project_id, *, create=False):
        return self._area(project_id, "datasets", create=create)

    def runs_dir(self, project_id, *, create=False):
        return self._area(project_id, "runs", create=create)

    def models_dir(self, project_id, *, create=False):
        return self._area(project_id, "models", create=create)

    def model_exports_dir(self, project_id, *, create=False):
        return self._area(project_id, "model-exports", create=create)

    def predictions_dir(self, project_id, *, create=False):
        return self._area(project_id, "predictions", create=create)

    def _probe_maskrcnn(self):
        environment = Path(self.python).parent.parent
        packages = environment / "Lib" / "site-packages"
        if (packages / "torch" / "__init__.py").is_file() and (packages / "torchvision" / "__init__.py").is_file():
            return True
        try:
            result = subprocess.run([self.python, "-c", "import torch,torchvision"], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    @staticmethod
    def _pid_alive(pid):
        try:
            pid = int(pid)
            if pid <= 0:
                return False
            if os.name == "nt":
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if not handle:
                    return False
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            os.kill(pid, 0)
            return True
        except (OSError, TypeError, ValueError):
            return False

    def _reconcile_runs(self):
        active = {"queued", "preparing", "running", "stopping"}
        paths = (path for project in self.store.list_projects()
                 for path in self.runs_dir(project["id"]).glob("*/run.json"))
        for path in paths:
            try:
                run = read_json(path)
                if run.get("status") in active and not self._pid_alive(run.get("worker_pid")):
                    run.update(status="failed", progress=None, message="上次訓練程序已不在執行",
                               error="工作台重新開啟時找不到原訓練程序", completed_at=time.time(), updated_at=time.time())
                    atomic_json(path, run)
            except (OSError, ValueError, json.JSONDecodeError):
                continue

    def capabilities(self):
        catalog = self.registry.snapshot()
        self.python = catalog["worker_python"]
        self._maskrcnn_available = bool(next((model["train"] for model in catalog["models"]
                                              if model["key"] == MASKRCNN_KEY), False))
        return {"available": True, "worker_python": self.python,
                "engines": [{**model, "parameters": parameter_schema(model)} for model in catalog["models"]],
                "components": catalog["components"], "tasks": catalog["tasks"],
                "refreshed_at": catalog["refreshed_at"]}

    def refresh_components(self):
        self.registry.snapshot(refresh=True)
        return self.capabilities()

    def install_component(self, component_id, progress):
        active = self.active_runs()
        if active:
            raise ValueError("訓練正在執行，完成或停止後才能變更模型環境")
        result = self.registry.install(component_id, progress)
        self.refresh_components()
        return result

    def readiness(self, project_id):
        project = self.store.snapshot(project_id)
        approved = [asset for asset in project["assets"] if asset["review_state"] == "approved"]
        blockers, warnings = [], []
        if not approved:
            blockers.append({"code": "no_approved_assets", "message": "沒有已核准圖片", "action": "前往資料審核"})
        if not project.get("classes"):
            blockers.append({"code": "no_classes", "message": "專案尚未建立類別", "action": "管理物件類別"})
        missing_split = [asset for asset in approved if asset.get("split") not in {"train", "val", "test"}]
        if missing_split:
            blockers.append({"code": "missing_split", "message": f"{len(missing_split)} 張已核准圖片尚未指定資料分割",
                             "action": "設定資料分割"})
        splits = {name: sum(asset.get("split") == name for asset in approved) for name in ("train", "val", "test")}
        purpose = split_purpose(project.get('split_plan'), approved)
        if approved and not splits["train"]:
            blockers.append({"code": "no_train_split", "message": "Train 沒有已核准圖片", "action": "設定資料分割"})
        if approved and not splits["val"] and purpose != 'all_train':
            blockers.append({"code": "no_validation_split", "message": "Validation 沒有已核准圖片，不能安全選模或調整門檻",
                             "action": "設定獨立的 Validation；Test 僅供最後評估，不會替代 Validation"})
        duplicate_splits = {}
        for asset in approved:
            if asset.get('sha256'):
                duplicate_splits.setdefault(asset['sha256'], set()).add(asset.get('split'))
        leaked_duplicates = [sha for sha, groups in duplicate_splits.items() if len(groups) > 1]
        if leaked_duplicates:
            blockers.append({'code': 'duplicate_split_leak',
                             'message': f'{len(leaked_duplicates)} 組完全相同圖片跨越不同集合',
                             'action': '使用資料分割管理重新分配；相同圖片必須留在同一集合'})
        class_counts = {label: 0 for label in project.get("classes", [])}
        unsupported = []
        for asset in approved:
            for shape in asset.get("shapes", []):
                if shape.get("type") not in AREA_SHAPES:
                    unsupported.append((asset["id"], shape.get("type")))
                if shape.get("label") in class_counts:
                    class_counts[shape["label"]] += 1
                if shape.get("type") == "mask":
                    # Decode now so malformed RLE never reaches a worker.
                    decode_rle(shape.get("counts"), asset["width"], asset["height"])
        if unsupported:
            kinds = "、".join(sorted({str(kind) for _aid, kind in unsupported}))
            blockers.append({"code": "unsupported_shapes", "message": f"{len(unsupported)} 個標註類型無法用於面積分割：{kinds}",
                             "action": "查看圖片"})
        empty = [label for label, count in class_counts.items() if count == 0]
        if empty:
            warnings.append({"code": "empty_classes", "message": f"沒有已核准實例的類別：{'、'.join(empty)}",
                             "action": "檢查類別"})
        batches = {}
        for asset in approved:
            batches.setdefault(asset.get("batch_id") or "", set()).add(asset.get("split"))
        leaked = [batch for batch, groups in batches.items() if batch and len(groups) > 1]
        if leaked and purpose == 'formal':
            warnings.append({"code": "source_group_leak", "message": f"{len(leaked)} 個拍攝批次跨越不同資料分割",
                             "action": "若批次內影像高度相似，建議調整分割後再建立資料版本"})
        coverage = split_class_coverage(approved, active_splits=[s for s, count in splits.items() if count],
                                        policy=purpose)
        blockers.extend(coverage["blockers"])
        warnings.extend(coverage["warnings"])
        return {"ready": not blockers, "project_id": project_id, "project_revision": project["revision"],
                'purpose': purpose,
                "stats": {"approved": len(approved), "excluded": project["stats"]["total"] - len(approved),
                          "splits": splits, "classes": class_counts, "class_counts": coverage["class_counts"]},
                "blockers": blockers, "warnings": warnings}

    def create_dataset_version(self, project_id, augmentation=None):
        report = self.readiness(project_id)
        if not report["ready"]:
            raise ValueError("；".join(item["message"] for item in report["blockers"]))
        project = self.store.snapshot(project_id)
        parent = self.datasets_dir(project_id, create=True)
        with self.lock:
            dataset_id = _next_id(parent, "D")
            target = parent / dataset_id
            temporary = parent / f".{dataset_id}-{uuid.uuid4().hex}.tmp"
            (temporary / "images").mkdir(parents=True)
            records = []
            try:
                for asset in project["assets"]:
                    if asset["review_state"] != "approved":
                        continue
                    source = Path(asset["image_path"])
                    suffix = source.suffix.lower() or ".png"
                    destination = temporary / "images" / f"{asset['id']}{suffix}"
                    _link_or_copy(source, destination)
                    raw = destination.read_bytes()
                    if sha256(raw).hexdigest() != asset["sha256"]:
                        raise ValueError(f"建立資料版本時圖片雜湊不符：{asset['name']}")
                    shapes = json.loads(dump(asset["shapes"]))
                    objects = []
                    for shape in shapes:
                        box = bounds(shape, asset["width"], asset["height"])
                        objects.append({"shape_id": shape["id"], "class_id": project["classes"].index(shape["label"]),
                                        "label": shape["label"], "type": shape["type"], "bbox_xywh": box})
                    records.append({"asset_id": asset["id"], "name": asset["name"], "width": asset["width"],
                                    "height": asset["height"], "sha256": asset["sha256"],
                                    "image_file": f"images/{destination.name}", "annotation_revision": asset["revision"],
                                    "annotation_sha256": _canonical_hash(shapes), "split": asset["split"],
                                    "batch_id": asset["batch_id"], "source": asset["source"],
                                    "shapes": shapes, "objects": objects})
                augmentation = normalize_augmentation(augmentation)
                manifest = {"schema_version": 1, "dataset_version_id": dataset_id, "project_id": project_id,
                            "project_name": project["name"], "project_revision": project["revision"],
                            "created_at": timestamp(), "classes": list(project["classes"]),
                            "class_mapping": [{"class_id": index, "name": name} for index, name in enumerate(project["classes"])],
                            "readiness": report, "assets": records,
                            "augmentation": augmentation,
                            "split_plan": project.get("split_plan")}
                purpose = split_purpose(project.get('split_plan'), records)
                labels = {
                    'formal': '正式獨立來源評估',
                    'reviewed_independent': '人工確認獨立／圖片層級平衡',
                    'experimental': '同來源寬鬆實驗',
                    'all_train': '全資料最終訓練（無獨立評估）',
                }
                if purpose == 'formal' and not (project.get('split_plan') or {}).get('source_isolation'):
                    labels['formal'] = '嚴格類別覆蓋／來源獨立性未宣告'
                manifest['data_quality'] = {
                    'purpose': purpose, 'label': labels[purpose],
                    'independent_sources': (True if purpose == 'formal' and (project.get('split_plan') or {}).get('source_isolation')
                                            else False if purpose != 'formal' else None),
                    'reviewed_independent': purpose == 'reviewed_independent',
                    'evaluation_available': purpose != 'all_train',
                    'independence_review': project.get('independence_review') if purpose == 'reviewed_independent' else None,
                    'split_policy': {'algorithm_version': (project.get('split_plan') or {}).get('algorithm_version'),
                                     'seed': (project.get('split_plan') or {}).get('seed'),
                                     'balance_mode': (project.get('split_plan') or {}).get('balance_mode'),
                                     'source_isolation': (project.get('split_plan') or {}).get('source_isolation')},
                    'warnings': [item['message'] for item in report['warnings']]}
                manifest["manifest_sha256"] = _canonical_hash(manifest)
                atomic_json(temporary / "manifest.json", manifest)
                temporary.replace(target)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        self._sync_catalog(project_id)
        return self.dataset(project_id, dataset_id)

    def dataset(self, project_id, dataset_id):
        dataset_id = _safe_id(dataset_id, "D")
        path = self.datasets_dir(project_id) / dataset_id / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError("找不到訓練資料版本")
        manifest = read_json(path)
        augmentation = normalize_augmentation(manifest.get("augmentation"))
        train_count = sum(a["split"] == "train" for a in manifest["assets"])
        return {"id": dataset_id, "project_id": project_id, "project_revision": manifest["project_revision"],
                "created_at": manifest["created_at"], "classes": manifest["classes"],
                "asset_count": len(manifest["assets"]), "splits": {name: sum(a["split"] == name for a in manifest["assets"])
                for name in ("train", "val", "test")}, "manifest_sha256": manifest["manifest_sha256"],
                "data_quality": manifest.get("data_quality"),
                "augmentation": augmentation,
                "training_events": augmentation_event_layout(train_count, augmentation),
                "readiness": dataset_readiness(manifest)}

    def create_diagnostic_dataset_version(self, project_id, source_dataset_id):
        """Create a separately labelled copy without modifying project splits or history."""
        from .diagnostic_split import diagnostic_temporal_split
        source_dataset_id = _safe_id(source_dataset_id, "D")
        parent = self.datasets_dir(project_id, create=True)
        source_root = parent / source_dataset_id
        source = read_json(source_root / "manifest.json")
        plan = diagnostic_temporal_split(source["assets"])
        with self.lock:
            dataset_id = _next_id(parent, "D")
            temporary = parent / f".{dataset_id}-{uuid.uuid4().hex}.tmp"
            (temporary / "images").mkdir(parents=True)
            try:
                manifest = json.loads(dump(source))
                manifest["assets"] = [a for a in manifest["assets"] if a["asset_id"] in plan["assignments"]]
                for asset in manifest["assets"]:
                    asset["split"] = plan["assignments"][asset["asset_id"]]
                    origin = (source_root / asset["image_file"]).resolve()
                    if not origin.is_relative_to(source_root.resolve()):
                        raise ValueError("圖片路徑超出原始資料版本")
                    destination = temporary / "images" / origin.name
                    _link_or_copy(origin, destination)
                    if sha256(destination.read_bytes()).hexdigest() != asset["sha256"]:
                        raise ValueError("來源圖片雜湊不符")
                    asset["image_file"] = f"images/{destination.name}"
                quality = {**plan["data_quality"], "source_dataset_version_id": source_dataset_id,
                           "source_manifest_sha256": source["manifest_sha256"]}
                manifest.update(dataset_version_id=dataset_id, created_at=timestamp(), data_quality=quality,
                                split_plan={"strategy": "diagnostic_temporal", "blocks": plan["blocks"]},
                                readiness={**plan["coverage"], "stats": {"splits": plan["coverage"]["image_counts"]}})
                manifest.pop("manifest_sha256", None)
                manifest["manifest_sha256"] = _canonical_hash(manifest)
                atomic_json(temporary / "manifest.json", manifest)
                temporary.replace(parent / dataset_id)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        self._sync_catalog(project_id)
        return self.dataset(project_id, dataset_id)

    def list_datasets(self, project_id):
        parent = self.datasets_dir(project_id)
        return [self.dataset(project_id, item.name) for item in sorted(parent.iterdir(), reverse=True)
                if item.is_dir() and item.name.startswith("D")] if parent.is_dir() else []

    def _run_path(self, project_id, run_id):
        return self.runs_dir(project_id) / _safe_id(run_id, "R") / "run.json"

    def _model_path(self, project_id, model_id):
        return self.models_dir(project_id) / _safe_id(model_id, "M") / "model.json"

    def start_run(self, project_id, dataset_id, config):
        if not isinstance(config, dict):
            raise ValueError("訓練參數必須是物件")
        dataset_id = _safe_id(dataset_id, "D")
        manifest = self.datasets_dir(project_id) / dataset_id / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError("找不到訓練資料版本")
        if any(run.get("status") in {"queued", "preparing", "running", "stopping"} for run in self.list_runs(project_id)):
            raise ValueError("此專案已有訓練正在執行")
        engine = str(config.get("engine") or (MASKRCNN_KEY if self._maskrcnn_available else ENGINE_KEY))
        definition = self.registry.model(engine)
        if definition is None:
            raise ValueError("找不到所選訓練引擎")
        if not definition["train"]:
            raise ValueError(definition.get("unavailable_reason") or "所選訓練引擎尚未安裝")
        effective_config = validate_config(definition, config)
        immutable = read_json(manifest)
        if "augmentation" in immutable:
            effective_config["augmentation"] = normalize_augmentation(immutable.get("augmentation"))
        coverage = dataset_readiness(immutable)
        if not coverage["ready"]:
            raise ValueError("固定資料版本的分割不適合訓練：" + "；".join(item["message"] for item in coverage["blockers"]))
        if effective_config.get("scheduler") == "plateau" and not any(a["split"] == "val" for a in immutable.get("assets", [])):
            raise ValueError("Validation 停滯下降需要獨立 Validation 集合，不能使用 Test 調整學習率")
        if coverage.get('purpose') == 'all_train' and engine == ENGINE_KEY:
            raise ValueError('全資料最終訓練不支援像素原型基準；請先用有 Validation 的資料選模，再以深度學習模型執行最終訓練')
        if engine in CLASSIFICATION_ENGINES:
            if len(immutable["classes"]) < 2:
                raise ValueError("影像分類至少需要兩個類別")
            ambiguous = []
            for asset in immutable["assets"]:
                labels = {shape.get("label") for shape in asset.get("shapes", []) if shape.get("label") in immutable["classes"]}
                if len(labels) != 1:
                    ambiguous.append(asset.get("name") or asset["asset_id"])
            if ambiguous:
                raise ValueError(f"影像分類要求每張圖片只有一個標註類別；請修正 {len(ambiguous)} 張圖片")
            train_labels = {shape.get("label") for asset in immutable["assets"] if asset["split"] == "train"
                            for shape in asset.get("shapes", [])}
            missing_classes = [name for name in immutable["classes"] if name not in train_labels]
            if missing_classes and not loose_split_applies(immutable.get('split_plan'), immutable['assets']):
                raise ValueError(f"Train 缺少分類樣本：{'、'.join(missing_classes)}")
        compatibility = None
        if engine.startswith("yolo26") and definition.get('task') == 'instance_segmentation':
            compatibility = analyze_manifest(immutable, effective_config)
            if not compatibility["compatible"]:
                raise ValueError(blocker_message(compatibility))
        project_runs = self.runs_dir(project_id, create=True)
        project_models = self.models_dir(project_id, create=True)
        with self.lock:
            run_id, model_id = _next_id(project_runs, "R"), _next_id(project_models, "M")
            run_dir, model_dir = project_runs / run_id, project_models / model_id
            run_dir.mkdir()
            model_dir.mkdir()
            engine_name = definition["name"]
            run = {"schema_version": 1, "run_id": run_id, "project_id": project_id,
                   "dataset_version_id": dataset_id, "model_version_id": model_id,
                   "engine": engine, "engine_name": engine_name,
                   "config": effective_config,
                   "status": "queued", "message": "等待訓練程序", "progress": 0,
                   "created_at": time.time(), "updated_at": time.time()}
            if immutable.get("data_quality"):
                run["data_quality"] = immutable["data_quality"]
                run['data_purpose'] = immutable['data_quality'].get('purpose')
            run['training_events'] = augmentation_event_layout(
                sum(asset.get('split') == 'train' for asset in immutable.get('assets', [])),
                effective_config.get('augmentation'))
            if coverage["warnings"]:
                run["split_warnings"] = coverage["warnings"]
            if compatibility is not None:
                run["yolo_compatibility"] = compatibility
            atomic_json(run_dir / "run.json", run)
            worker_python = str(self.registry.component_python(definition["component"]))
            run["runtime"] = {"component": definition["component"], "python": worker_python}
            atomic_json(run_dir / "run.json", run)
            command = [worker_python, "-m", "workbench.training_worker", "--dataset", str(manifest),
                       "--run-dir", str(run_dir), "--model-dir", str(model_dir)]
            kwargs = {"cwd": str(Path(__file__).resolve().parents[1]), "stdin": subprocess.DEVNULL}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            stdout = (run_dir / "worker.stdout.log").open("wb")
            stderr = (run_dir / "worker.stderr.log").open("wb")
            try:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **kwargs)
            finally:
                stdout.close()
                stderr.close()
            self.processes[(project_id, run_id)] = process
            current = read_json(run_dir / "run.json")
            current["worker_pid"] = process.pid
            if current["status"] == "queued":
                current.update(status="preparing", message="訓練程序已啟動")
            atomic_json(run_dir / "run.json", current)
            threading.Thread(target=self._reap, args=(project_id, run_id, process),
                             name=f"training-{run_id}", daemon=True).start()
        self._sync_catalog(project_id)
        return current

    def yolo_compatibility(self, project_id, dataset_id, config):
        dataset_id = _safe_id(dataset_id, "D")
        manifest = self.datasets_dir(project_id) / dataset_id / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError("找不到訓練資料版本")
        if not isinstance(config, dict):
            raise ValueError("訓練參數必須是物件")
        engine = str(config.get("engine") or "")
        definition = self.registry.model(engine)
        if definition is None or not engine.startswith("yolo26") or definition.get('task') != 'instance_segmentation':
            raise ValueError("相容檢查只適用於 YOLO Seg")
        effective = validate_config(definition, config)
        return analyze_manifest(read_json(manifest), effective)

    def review_yolo_compatibility(self, project_id, config=None):
        """Check the latest review candidates before an immutable dataset exists."""
        project = self.store.snapshot(project_id)
        assets = [asset for asset in project.get("assets", [])
                  if asset.get("review_state") in {"pending", "approved"}]
        records = [{
            "asset_id": asset["id"], "name": asset["name"],
            "width": asset["width"], "height": asset["height"],
            "split": asset.get("split"), "shapes": asset.get("shapes", []),
        } for asset in assets]
        effective = config or {}
        config_key = _canonical_hash(effective)

        def inspect(record):
            key = (project_id, record["asset_id"], _canonical_hash(record["shapes"]), config_key)
            with self.lock:
                cached = self._review_yolo_cache.get(key)
            if cached is not None:
                return cached
            result = analyze_manifest({"assets": [record]}, effective)
            with self.lock:
                self._review_yolo_cache[key] = result
                if len(self._review_yolo_cache) > 1024:
                    self._review_yolo_cache = dict(list(self._review_yolo_cache.items())[-768:])
            return result

        workers = min(4, max(1, len(records)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="yolo-review") as pool:
            parts = list(pool.map(inspect, records))
        repairs = [item for part in parts for item in part.get("repairs", [])]
        blockers = [item for part in parts for item in part.get("blockers", [])]
        report = {
            "schema_version": 1, "compatible": not blockers,
            "policy": parts[0]["policy"] if parts else analyze_manifest({"assets": []}, effective)["policy"],
            "summary": {
                "assets_scanned": len(records),
                "shapes_scanned": sum(part["summary"]["shapes_scanned"] for part in parts),
                "affected_assets": len({item["asset_id"] for item in repairs}),
                "affected_shapes": len(repairs),
                "holes_repaired": sum(len(item.get("holes", [])) for item in repairs),
                "pixels_repaired": sum(item.get("hole_pixels", 0) for item in repairs),
                "blocked_assets": len({item["asset_id"] for item in blockers}),
            },
            "repairs": repairs, "blockers": blockers, "source_annotations_unchanged": True,
            "metric_label_space": "yolo_compatible_copy" if repairs else "source_annotations",
        }
        report.update(
            project_revision=project["revision"],
            review_scope={
                "pending": sum(asset.get("review_state") == "pending" for asset in assets),
                "approved": sum(asset.get("review_state") == "approved" for asset in assets),
                "excluded_rejected": project.get("stats", {}).get("rejected", 0),
            },
        )
        return report

    def _reap(self, project_id, run_id, process):
        return_code = process.wait()
        with self.lock:
            self.processes.pop((project_id, run_id), None)
            path = self._run_path(project_id, run_id)
            if path.is_file():
                run = read_json(path)
                if return_code and run.get("status") in {"queued", "preparing", "running", "stopping"}:
                    error_log = path.parent / "worker.stderr.log"
                    detail = error_log.read_text(encoding="utf-8", errors="replace").strip() if error_log.is_file() else ""
                    message = detail.splitlines()[-1] if detail else f"訓練程序異常結束（{return_code}）"
                    run.update(status="failed", message=message, error=message, progress=None,
                               completed_at=time.time(), updated_at=time.time())
                    atomic_json(path, run)

    def run(self, project_id, run_id):
        path = self._run_path(project_id, run_id)
        if not path.is_file():
            raise FileNotFoundError("找不到訓練紀錄")
        return _with_evaluation_reassessment(read_json(path), path.parent)

    def run_metrics(self, project_id, run_id):
        """Return canonical epoch rows, enriching Ultralytics history from results.csv."""
        run = self.run(project_id, run_id)
        directory = self._run_path(project_id, run_id).parent
        rows = {}
        if run.get('engine') in ULTRALYTICS_ENGINES:
            from .ultralytics_engine import read_ultralytics_results
            for row in read_ultralytics_results(directory / 'ultralytics' / 'fit' / 'results.csv'):
                # Native CSV rates are scheduled parameter-group values. The
                # Workbench series means a rate observed on a successful
                # optimizer update, so CSV values must not fill skipped gaps.
                row = {key: value for key, value in row.items()
                       if key != 'train/learning_rate' and not key.startswith('lr/group_')}
                rows[row['epoch']] = row
        metrics = directory / 'metrics.jsonl'
        if metrics.is_file():
            for line in metrics.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                epoch = row.get('epoch')
                if type(epoch) is int and epoch > 0:
                    rows[epoch] = {**rows.get(epoch, {'epoch': epoch}), **row}
        return {'run': run, 'metrics': [rows[epoch] for epoch in sorted(rows)]}

    def list_runs(self, project_id):
        parent = self.runs_dir(project_id)
        rows = [_with_evaluation_reassessment(read_json(item / "run.json"), item) for item in parent.iterdir()
                if item.is_dir() and (item / "run.json").is_file()] if parent.is_dir() else []
        return sorted(rows, key=lambda row: row.get("created_at", 0), reverse=True)

    def active_runs(self):
        return [run for project in self.store.list_projects()
                for run in self.list_runs(project["id"])
                if run.get("status") in {"queued", "preparing", "running", "stopping"}]

    def stop_run(self, project_id, run_id):
        run = self.run(project_id, run_id)
        if run["status"] not in {"queued", "preparing", "running"}:
            return run
        run_dir = self._run_path(project_id, run_id).parent
        (run_dir / "stop.requested").touch()
        run.update(status="stopping", message="正在安全停止…", updated_at=time.time())
        atomic_json(run_dir / "run.json", run)
        return run

    def list_models(self, project_id):
        parent = self.models_dir(project_id)
        rows = [_with_evaluation_reassessment(read_json(item / "model.json"), item, model=True) for item in parent.iterdir()
                if item.is_dir() and (item / "model.json").is_file()] if parent.is_dir() else []
        return sorted(rows, key=lambda row: row.get("created_at", 0), reverse=True)

    def model(self, project_id, model_id):
        path = self._model_path(project_id, model_id)
        if not path.is_file():
            raise FileNotFoundError("找不到模型版本")
        return _with_evaluation_reassessment(read_json(path), path.parent, model=True)

    def export_model(self, project_id, model_id, progress=lambda _message, _percent=None: None):
        """Create a self-describing, checksummed model bundle without dataset images."""
        model = self.model(project_id, model_id)
        project = self.store.get_project(project_id, include_assets=False)
        model_dir = self._model_path(project_id, model_id).parent
        run_id = _safe_id(model.get("run_id"), "R")
        run_dir = self._run_path(project_id, run_id).parent
        parent = self.model_exports_dir(project_id, create=True)
        with self.lock:
            export_id = _next_id(parent, "E")
            target = parent / export_id
            temporary = parent / f".{export_id}-{uuid.uuid4().hex}.tmp"
            temporary.mkdir()
            try:
                progress("整理模型、評估與訓練來源", 15)
                sources = []
                for path in sorted(model_dir.iterdir()):
                    if path.is_file() and not path.is_symlink():
                        sources.append((path, f"model/{path.name}"))
                for name in ("run.json", "metrics.jsonl", "evaluation.json", "evaluation.v2.json", "artifact-manifest.json"):
                    path = run_dir / name
                    if path.is_file() and not path.is_symlink():
                        sources.append((path, f"run/{name}"))
                if not any(name == "model/model.json" for _path, name in sources):
                    raise FileNotFoundError("模型描述檔不存在")
                artifacts = []
                for path, archive_name in sources:
                    raw = path.read_bytes()
                    artifacts.append({"path": archive_name, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)})
                created_at = timestamp()
                manifest = {"schema_version": 1, "format": "vision-workbench-model-bundle",
                            "export_id": export_id, "created_at": created_at,
                            "project_id": project_id, "project_name": project["name"],
                            "model_version_id": model_id, "run_id": run_id,
                            "dataset_version_id": model.get("dataset_version_id"),
                            "engine": model.get("engine"), "engine_name": model.get("engine_name"),
                            "classes": model.get("classes", []), "artifacts": artifacts}
                bundle = temporary / f"{project_id}-{model_id}-model.zip"
                progress("建立可攜式模型封裝", 55)
                with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                    for source, archive_name in sources:
                        archive.write(source, archive_name)
                    archive.writestr("export-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
                    archive.writestr("README.txt", "Vision Workbench model bundle\nOpen export-manifest.json to verify lineage and SHA-256 values.\n")
                raw = bundle.read_bytes()
                record = {**manifest, "path": str((target / bundle.name).resolve()),
                          "directory": str(target.resolve()), "bytes": len(raw), "sha256": sha256(raw).hexdigest()}
                atomic_json(temporary / "record.json", record)
                temporary.replace(target)
                progress("模型封裝已完成並驗證", 100)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        self._sync_catalog(project_id)
        return self.model_export(project_id, export_id)

    def model_export(self, project_id, export_id):
        export_id = _safe_id(export_id, "E")
        path = self.model_exports_dir(project_id) / export_id / "record.json"
        if not path.is_file():
            raise FileNotFoundError("找不到模型匯出版本")
        return read_json(path)

    def list_model_exports(self, project_id):
        parent = self.model_exports_dir(project_id)
        rows = [read_json(item / "record.json") for item in parent.iterdir()
                if item.is_dir() and (item / "record.json").is_file()] if parent.is_dir() else []
        return sorted(rows, key=lambda row: row.get("created_at", ""), reverse=True)

    def create_predictions(self, project_id, model_id, asset_ids=None):
        model = self.model(project_id, model_id)
        definition = self.registry.model(model.get("engine"), refresh=True)
        if not definition or not definition.get("predict"):
            if model.get("task") == "image_classification":
                raise ValueError("影像分類結果不會轉成整張圖片的 Bounding Box；目前僅提供訓練、評估與模型匯出")
            raise RuntimeError(definition.get("unavailable_reason") if definition else "此模型不支援預標註")
        project = self.store.snapshot(project_id)
        requested = set(asset_ids or [asset["id"] for asset in project["assets"] if asset["review_state"] != "approved"])
        if not requested:
            raise ValueError("沒有可產生預標註的圖片")
        selected = [asset for asset in project["assets"] if asset["id"] in requested]
        missing = requested.difference(asset["id"] for asset in selected)
        if missing:
            raise FileNotFoundError("部分預標註圖片不存在")
        generated = {}
        external_engines = {MASKRCNN_KEY, *DETECTION_ENGINES, *SEMANTIC_ENGINES, *ULTRALYTICS_ENGINES}
        if model.get("engine") in external_engines:
            if not definition.get("train"):
                raise RuntimeError(definition.get("unavailable_reason") or "模型執行環境不可用")
            token = uuid.uuid4().hex
            prediction_dir = self.predictions_dir(project_id, create=True)
            request_path = prediction_dir / f".{token}.request.json"
            output_path = prediction_dir / f".{token}.output.json"
            stderr_path = prediction_dir / f".{token}.stderr.log"
            atomic_json(request_path, {"model_path": str(self._model_path(project_id, model_id)), "device": "auto",
                        "assets": [{"asset_id": asset["id"], "image_path": asset["image_path"],
                                    "width": asset["width"], "height": asset["height"]} for asset in selected]})
            try:
                worker = ("workbench.maskrcnn_predict_worker" if model.get("engine") == MASKRCNN_KEY else
                          "workbench.ultralytics_predict_worker" if model.get("engine") in ULTRALYTICS_ENGINES else
                          "workbench.torchvision_predict_worker")
                worker_python = str(self.registry.component_python(definition["component"]))
                with stderr_path.open("wb") as stderr:
                    result = subprocess.run([worker_python, "-m", worker, "--request", str(request_path),
                                             "--output", str(output_path)], cwd=str(Path(__file__).resolve().parents[1]),
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if result.returncode or not output_path.is_file():
                    detail = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
                    raise RuntimeError(detail.splitlines()[-1] if detail else "TorchVision 推論程序失敗")
                generated = {row["asset_id"]: row["shapes"] for row in read_json(output_path)["assets"]}
            finally:
                request_path.unlink(missing_ok=True); output_path.unlink(missing_ok=True); stderr_path.unlink(missing_ok=True)
        records = []
        for asset in selected:
            if model.get("engine") in external_engines:
                shapes = generated.get(asset["id"], [])
            else:
                with Image.open(asset["image_path"]) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
                shapes = predict(rgb, model, asset["width"], asset["height"])
            records.append({"asset_id": asset["id"], "asset_name": asset["name"], "base_revision": asset["revision"],
                            "shapes": shapes, "status": "candidate" if shapes else "empty"})
        candidate_id = uuid.uuid4().hex
        record = {"schema_version": 1, "candidate_id": candidate_id, "project_id": project_id,
                  "model_version_id": model_id, "run_id": model["run_id"], "created_at": timestamp(), "assets": records}
        atomic_json(self.predictions_dir(project_id, create=True) / f"{candidate_id}.json", record)
        self._sync_catalog(project_id)
        return record

    def list_predictions(self, project_id):
        rows = []
        for path in self.predictions_dir(project_id).glob("*.json"):
            try:
                record = read_json(path)
                if record.get("project_id") == project_id:
                    rows.append(record)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row.get("created_at", ""), reverse=True)

    def accept_predictions(self, candidate_id, asset_ids=None):
        if not isinstance(candidate_id, str) or len(candidate_id) != 32:
            raise ValueError("候選標註 ID 無效")
        matches = [self.predictions_dir(project["id"]) / f"{candidate_id}.json"
                   for project in self.store.list_projects()]
        path = next((item for item in matches if item.is_file()), None)
        if path is None:
            raise FileNotFoundError("找不到候選標註")
        record = read_json(path)
        requested = set(asset_ids or [row["asset_id"] for row in record["assets"] if row["status"] == "candidate"])
        accepted = []
        for row in record["assets"]:
            if row["asset_id"] not in requested or row["status"] != "candidate":
                continue
            current = self.store.get_asset(record["project_id"], row["asset_id"], internal=True)
            if current["revision"] != row["base_revision"]:
                raise ValueError(f"{row['asset_name']} 已有新修訂，請重新產生預標註")
            saved = self.store.save_asset(record["project_id"], row["asset_id"], current["shapes"] + row["shapes"], current["revision"])
            row.update(status="accepted", accepted_revision=saved["revision"], accepted_at=timestamp())
            accepted.append(row["asset_id"])
        atomic_json(path, record)
        self._sync_catalog(record["project_id"])
        return {"candidate_id": candidate_id, "accepted": accepted,
                "project": self.store.get_project(record["project_id"])}

    def _sync_catalog(self, project_id):
        """Rebuild the SQL index from authoritative project-owned manifests."""
        if not getattr(self, "_project_scoped", False):
            return
        issues = []
        datasets, runs, models, exports, predictions = [], [], [], [], []
        dataset_ids, run_ids, model_ids = set(), set(), set()

        for directory in sorted(self.datasets_dir(project_id).glob("D*")):
            path = directory / "manifest.json"
            if not path.is_file():
                continue
            try:
                row = read_json(path)
                dataset_id = _safe_id(row.get("dataset_version_id"), "D")
                if dataset_id != directory.name:
                    raise ValueError("資料版本 ID 與資料夾不一致")
                source_id = (row.get("data_quality") or {}).get("source_dataset_version_id")
                datasets.append({
                    "id": dataset_id,
                    "project_revision": int(row["project_revision"]),
                    "created_at": str(row["created_at"]),
                    "manifest_sha256": str(row["manifest_sha256"]),
                    "asset_count": len(row.get("assets", [])),
                    "relative_path": f"datasets/{dataset_id}/manifest.json",
                    "source_dataset_id": source_id,
                })
                dataset_ids.add(dataset_id)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                issues.append({"path": str(path), "message": str(error)})

        # A diagnostic version may sort before its source only by an unusual ID.
        for row in datasets:
            if row["source_dataset_id"] not in dataset_ids:
                row["source_dataset_id"] = None

        for directory in sorted(self.runs_dir(project_id).glob("R*")):
            path = directory / "run.json"
            if not path.is_file():
                continue
            try:
                row = read_json(path)
                run_id = _safe_id(row.get("run_id"), "R")
                dataset_id = _safe_id(row.get("dataset_version_id"), "D")
                if run_id != directory.name or dataset_id not in dataset_ids:
                    raise ValueError("訓練 ID 或資料版本關聯無效")
                created = float(row.get("created_at", 0))
                runs.append({
                    "id": run_id,
                    "dataset_version_id": dataset_id,
                    "proposed_model_id": row.get("model_version_id"),
                    "engine": str(row["engine"]),
                    "status": str(row["status"]),
                    "created_at": created,
                    "updated_at": float(row.get("updated_at", created)),
                    "relative_path": f"runs/{run_id}/run.json",
                })
                run_ids.add(run_id)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                issues.append({"path": str(path), "message": str(error)})

        for directory in sorted(self.models_dir(project_id).glob("M*")):
            path = directory / "model.json"
            if not path.is_file():
                continue
            try:
                row = read_json(path)
                model_id = _safe_id(row.get("model_version_id"), "M")
                run_id = _safe_id(row.get("run_id"), "R")
                dataset_id = _safe_id(row.get("dataset_version_id"), "D")
                if model_id != directory.name or run_id not in run_ids or dataset_id not in dataset_ids:
                    raise ValueError("模型 ID 或來源關聯無效")
                created = row.get("created_at", 0)
                models.append({
                    "id": model_id, "run_id": run_id, "dataset_version_id": dataset_id,
                    "engine": str(row["engine"]), "created_at": float(created),
                    "relative_path": f"models/{model_id}/model.json",
                })
                model_ids.add(model_id)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                issues.append({"path": str(path), "message": str(error)})

        for directory in sorted(self.model_exports_dir(project_id).glob("E*")):
            path = directory / "record.json"
            if not path.is_file():
                continue
            try:
                row = read_json(path)
                export_id = _safe_id(row.get("export_id"), "E")
                model_id = _safe_id(row.get("model_version_id"), "M")
                run_id = _safe_id(row.get("run_id"), "R")
                dataset_id = _safe_id(row.get("dataset_version_id"), "D")
                if export_id != directory.name or model_id not in model_ids or run_id not in run_ids or dataset_id not in dataset_ids:
                    raise ValueError("模型匯出來源關聯無效")
                exports.append({
                    "id": export_id, "model_version_id": model_id, "run_id": run_id,
                    "dataset_version_id": dataset_id, "created_at": str(row["created_at"]),
                    "sha256": str(row["sha256"]), "byte_count": int(row["bytes"]),
                    "relative_path": f"model-exports/{export_id}/record.json",
                })
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                issues.append({"path": str(path), "message": str(error)})

        for path in sorted(self.predictions_dir(project_id).glob("*.json")):
            try:
                row = read_json(path)
                candidate_id = str(row["candidate_id"])
                model_id = _safe_id(row.get("model_version_id"), "M")
                run_id = _safe_id(row.get("run_id"), "R")
                if path.stem != candidate_id or model_id not in model_ids or run_id not in run_ids:
                    raise ValueError("候選標註 ID 或模型來源關聯無效")
                assets = row.get("assets", [])
                predictions.append({
                    "id": candidate_id, "model_version_id": model_id, "run_id": run_id,
                    "created_at": str(row["created_at"]),
                    "candidate_count": sum(item.get("status") == "candidate" for item in assets),
                    "accepted_count": sum(item.get("status") == "accepted" for item in assets),
                    "relative_path": f"predictions/{path.name}",
                })
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                issues.append({"path": str(path), "message": str(error)})

        self.store.replace_artifact_catalog(project_id, {
            "datasets": datasets, "runs": runs, "models": models,
            "model_exports": exports, "predictions": predictions,
        })
        self.catalog_issues[project_id] = issues

    def overview(self, project_id):
        self._sync_catalog(project_id)
        return {"readiness": self.readiness(project_id), "datasets": self.list_datasets(project_id),
                "runs": self.list_runs(project_id), "models": self.list_models(project_id),
                "model_exports": self.list_model_exports(project_id),
                "predictions": self.list_predictions(project_id), "capabilities": self.capabilities(),
                "storage": {"layout": "project-scoped", "root": str(self.store.directory(project_id)),
                            "database": self.store.database_integrity(project_id),
                            "catalog_issues": self.catalog_issues.get(project_id, [])}}

    def close(self):
        # Workers write only to immutable Run/Model directories.  They may finish
        # after the UI closes; reopening Workbench reads their persisted state.
        self.processes.clear()
