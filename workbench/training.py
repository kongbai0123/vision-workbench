"""Dataset versions, training runs, model versions, and prediction candidates."""
from __future__ import annotations

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

from composer_core.geometry import bounds, decode_rle, shape_polygons
from .store import dump, timestamp
from .training_engine import ENGINE_KEY, ENGINE_NAME, atomic_json, predict, read_json
from .maskrcnn_engine import ENGINE_KEY as MASKRCNN_KEY, ENGINE_NAME as MASKRCNN_NAME
from .model_registry import ModelRegistry
from .torchvision_engines import DETECTION_ENGINES, SEMANTIC_ENGINES
from .classification_engine import CLASSIFICATION_ENGINES
from .ultralytics_engine import ULTRALYTICS_ENGINES
from .training_parameters import parameter_schema, validate_config


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


class TrainingWorkspace:
    """Own training data below one Workbench data root.

    ProjectStore remains the only writer of project annotations.  This class
    creates immutable copies and persists all long-running run state as files.
    """
    def __init__(self, data_root: Path, store, python_executable=None):
        self.root = Path(data_root).resolve()
        self.store = store
        self.datasets = self.root / "datasets"
        self.runs = self.root / "runs"
        self.models = self.root / "models"
        self.model_exports = self.root / "model-exports"
        self.predictions = self.root / "predictions"
        for folder in (self.datasets, self.runs, self.models, self.model_exports, self.predictions):
            folder.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry(Path(__file__).resolve().parents[1], python_executable)
        self.python = str(self.registry.training_python)
        self.lock = threading.RLock()
        self.processes = {}
        self._maskrcnn_available = self._probe_maskrcnn()
        self._reconcile_runs()

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
        for path in self.runs.glob("*/*/run.json"):
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
        active = [run for project in self.runs.iterdir() if project.is_dir()
                  for run in self.list_runs(project.name)
                  if run.get("status") in {"queued", "preparing", "running", "stopping"}]
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
        if approved and not splits["train"]:
            blockers.append({"code": "no_train_split", "message": "Train 沒有已核准圖片", "action": "設定資料分割"})
        if approved and not (splits["val"] or splits["test"]):
            blockers.append({"code": "no_evaluation_split", "message": "Validation 或 Test 至少需要一張圖片", "action": "設定資料分割"})
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
        if leaked:
            warnings.append({"code": "source_group_leak", "message": f"{len(leaked)} 個拍攝批次跨越不同資料分割",
                             "action": "若批次內影像高度相似，建議調整分割後再建立資料版本"})
        return {"ready": not blockers, "project_id": project_id, "project_revision": project["revision"],
                "stats": {"approved": len(approved), "excluded": project["stats"]["total"] - len(approved),
                          "splits": splits, "classes": class_counts}, "blockers": blockers, "warnings": warnings}

    def create_dataset_version(self, project_id):
        report = self.readiness(project_id)
        if not report["ready"]:
            raise ValueError("；".join(item["message"] for item in report["blockers"]))
        project = self.store.snapshot(project_id)
        parent = self.datasets / project_id
        parent.mkdir(parents=True, exist_ok=True)
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
                    shutil.copy2(source, destination)
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
                manifest = {"schema_version": 1, "dataset_version_id": dataset_id, "project_id": project_id,
                            "project_name": project["name"], "project_revision": project["revision"],
                            "created_at": timestamp(), "classes": list(project["classes"]),
                            "class_mapping": [{"class_id": index, "name": name} for index, name in enumerate(project["classes"])],
                            "readiness": report, "assets": records,
                            "split_plan": project.get("split_plan")}
                manifest["manifest_sha256"] = _canonical_hash(manifest)
                atomic_json(temporary / "manifest.json", manifest)
                temporary.replace(target)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return self.dataset(project_id, dataset_id)

    def dataset(self, project_id, dataset_id):
        dataset_id = _safe_id(dataset_id, "D")
        path = self.datasets / project_id / dataset_id / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError("找不到訓練資料版本")
        manifest = read_json(path)
        return {"id": dataset_id, "project_id": project_id, "project_revision": manifest["project_revision"],
                "created_at": manifest["created_at"], "classes": manifest["classes"],
                "asset_count": len(manifest["assets"]), "splits": {name: sum(a["split"] == name for a in manifest["assets"])
                for name in ("train", "val", "test")}, "manifest_sha256": manifest["manifest_sha256"]}

    def list_datasets(self, project_id):
        parent = self.datasets / project_id
        return [self.dataset(project_id, item.name) for item in sorted(parent.iterdir(), reverse=True)
                if item.is_dir() and item.name.startswith("D")] if parent.is_dir() else []

    def _run_path(self, project_id, run_id):
        return self.runs / project_id / _safe_id(run_id, "R") / "run.json"

    def _model_path(self, project_id, model_id):
        return self.models / project_id / _safe_id(model_id, "M") / "model.json"

    def start_run(self, project_id, dataset_id, config):
        if not isinstance(config, dict):
            raise ValueError("訓練參數必須是物件")
        dataset_id = _safe_id(dataset_id, "D")
        manifest = self.datasets / project_id / dataset_id / "manifest.json"
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
        if effective_config.get("scheduler") == "plateau" and not any(a["split"] == "val" for a in immutable.get("assets", [])):
            raise ValueError("Validation 停滯下降需要獨立 Validation 集合，不能使用 Test 調整學習率")
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
            if missing_classes:
                raise ValueError(f"Train 缺少分類樣本：{'、'.join(missing_classes)}")
        if engine.startswith("yolo26"):
            for asset in immutable["assets"]:
                for shape in asset.get("shapes", []):
                    polygons, diagnostics = shape_polygons(shape, int(asset["width"]), int(asset["height"]), tolerance=0)
                    if diagnostics.get("holes_omitted") or len(polygons) != 1 or diagnostics.get("pixel_iou", 1) < .999:
                        raise ValueError(f"{asset['name']} 含有 YOLO Seg 無法無損表示的複合遮罩")
        project_runs = self.runs / project_id
        project_models = self.models / project_id
        project_runs.mkdir(parents=True, exist_ok=True)
        project_models.mkdir(parents=True, exist_ok=True)
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
        return current

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
        return read_json(path)

    def list_runs(self, project_id):
        parent = self.runs / project_id
        rows = [read_json(item / "run.json") for item in parent.iterdir()
                if item.is_dir() and (item / "run.json").is_file()] if parent.is_dir() else []
        return sorted(rows, key=lambda row: row.get("created_at", 0), reverse=True)

    def active_runs(self):
        if not self.runs.is_dir():
            return []
        return [run for project in self.runs.iterdir() if project.is_dir()
                for run in self.list_runs(project.name)
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
        parent = self.models / project_id
        rows = [read_json(item / "model.json") for item in parent.iterdir()
                if item.is_dir() and (item / "model.json").is_file()] if parent.is_dir() else []
        return sorted(rows, key=lambda row: row.get("created_at", 0), reverse=True)

    def model(self, project_id, model_id):
        path = self._model_path(project_id, model_id)
        if not path.is_file():
            raise FileNotFoundError("找不到模型版本")
        return read_json(path)

    def export_model(self, project_id, model_id, progress=lambda _message, _percent=None: None):
        """Create a self-describing, checksummed model bundle without dataset images."""
        model = self.model(project_id, model_id)
        project = self.store.get_project(project_id, include_assets=False)
        model_dir = self._model_path(project_id, model_id).parent
        run_id = _safe_id(model.get("run_id"), "R")
        run_dir = self._run_path(project_id, run_id).parent
        parent = self.model_exports / project_id
        parent.mkdir(parents=True, exist_ok=True)
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
                for name in ("run.json", "metrics.jsonl", "evaluation.json", "artifact-manifest.json"):
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
        return self.model_export(project_id, export_id)

    def model_export(self, project_id, export_id):
        export_id = _safe_id(export_id, "E")
        path = self.model_exports / project_id / export_id / "record.json"
        if not path.is_file():
            raise FileNotFoundError("找不到模型匯出版本")
        return read_json(path)

    def list_model_exports(self, project_id):
        parent = self.model_exports / project_id
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
            request_path = self.predictions / f".{token}.request.json"
            output_path = self.predictions / f".{token}.output.json"
            stderr_path = self.predictions / f".{token}.stderr.log"
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
        atomic_json(self.predictions / f"{candidate_id}.json", record)
        return record

    def list_predictions(self, project_id):
        rows = []
        for path in self.predictions.glob("*.json"):
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
        path = self.predictions / f"{candidate_id}.json"
        if not path.is_file():
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
        return {"candidate_id": candidate_id, "accepted": accepted,
                "project": self.store.get_project(record["project_id"])}

    def overview(self, project_id):
        return {"readiness": self.readiness(project_id), "datasets": self.list_datasets(project_id),
                "runs": self.list_runs(project_id), "models": self.list_models(project_id),
                "model_exports": self.list_model_exports(project_id),
                "predictions": self.list_predictions(project_id), "capabilities": self.capabilities()}

    def close(self):
        # Workers write only to immutable Run/Model directories.  They may finish
        # after the UI closes; reopening Workbench reads their persisted state.
        self.processes.clear()
