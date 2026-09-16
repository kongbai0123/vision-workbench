"""Transactional project state shared by capture, editor, review and export.

Image bytes are immutable. SQLite commits annotations, revisions and review
events together. Each operation opens its own connection for worker safety.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import uuid

from PIL import Image

from composer_core.geometry import decode_rle, encode_rle
from composer_core.mask_cleanup import repair_tiny_holes


class ConflictError(ValueError):
    pass


CATALOG_SCHEMA_VERSION = 3


def ensure_catalog_schema(db):
    """Upgrade a project database to the filesystem artifact catalog schema."""
    current = db.execute("PRAGMA user_version").fetchone()[0]
    if current < CATALOG_SCHEMA_VERSION:
        # Catalog rows are a rebuildable index of immutable filesystem records.
        # Recreate them when constraints change; project annotations are never
        # part of these tables.
        for table in ("prediction_candidates", "model_exports", "model_versions",
                      "training_runs", "dataset_versions"):
            db.execute(f"DROP TABLE IF EXISTS {table}")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS dataset_versions(
            id TEXT PRIMARY KEY NOT NULL CHECK(id GLOB 'D[0-9]*'),
            project_revision INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
            asset_count INTEGER NOT NULL CHECK(asset_count >= 0),
            relative_path TEXT NOT NULL UNIQUE,
            source_dataset_id TEXT,
            FOREIGN KEY(source_dataset_id) REFERENCES dataset_versions(id)
        );
        CREATE TABLE IF NOT EXISTS training_runs(
            id TEXT PRIMARY KEY NOT NULL CHECK(id GLOB 'R[0-9]*'),
            dataset_version_id TEXT NOT NULL,
            proposed_model_id TEXT,
            engine TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('queued','preparing','running','stopping','completed','failed','stopped')),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id)
        );
        CREATE TABLE IF NOT EXISTS model_versions(
            id TEXT PRIMARY KEY NOT NULL CHECK(id GLOB 'M[0-9]*'),
            run_id TEXT NOT NULL UNIQUE,
            dataset_version_id TEXT NOT NULL,
            engine TEXT NOT NULL,
            created_at REAL NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            FOREIGN KEY(run_id) REFERENCES training_runs(id),
            FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id)
        );
        CREATE TABLE IF NOT EXISTS model_exports(
            id TEXT PRIMARY KEY NOT NULL CHECK(id GLOB 'E[0-9]*'),
            model_version_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            dataset_version_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
            byte_count INTEGER NOT NULL CHECK(byte_count >= 0),
            relative_path TEXT NOT NULL UNIQUE,
            FOREIGN KEY(model_version_id) REFERENCES model_versions(id),
            FOREIGN KEY(run_id) REFERENCES training_runs(id),
            FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id)
        );
        CREATE TABLE IF NOT EXISTS prediction_candidates(
            id TEXT PRIMARY KEY NOT NULL CHECK(length(id) = 32),
            model_version_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            candidate_count INTEGER NOT NULL CHECK(candidate_count >= 0),
            accepted_count INTEGER NOT NULL CHECK(accepted_count >= 0),
            relative_path TEXT NOT NULL UNIQUE,
            FOREIGN KEY(model_version_id) REFERENCES model_versions(id),
            FOREIGN KEY(run_id) REFERENCES training_runs(id)
        );
        CREATE INDEX IF NOT EXISTS idx_training_runs_dataset ON training_runs(dataset_version_id);
        CREATE INDEX IF NOT EXISTS idx_models_dataset ON model_versions(dataset_version_id);
        CREATE INDEX IF NOT EXISTS idx_exports_model ON model_exports(model_version_id);
        CREATE INDEX IF NOT EXISTS idx_predictions_model ON prediction_candidates(model_version_id);
    """)
    db.execute(f"PRAGMA user_version={CATALOG_SCHEMA_VERSION}")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("專案或圖片識別碼無效")
    return value


def clean_label(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 200 or any(ord(c) < 32 for c in value):
        raise ValueError("類別名稱必須為 1–200 個可見字元")
    return value.strip()


def clean_shapes(shapes, width, height):
    if not isinstance(shapes, list) or len(shapes) > 20000:
        raise ValueError("標註清單無效或超過 20,000 個物件")
    cleaned, seen = [], set()
    for item in shapes:
        if not isinstance(item, dict):
            raise ValueError("標註必須為物件")
        shape = json.loads(dump(item))
        shape["label"] = clean_label(shape.get("label"))
        shape_id = shape.get("id") or uuid.uuid4().hex
        if not isinstance(shape_id, str) or len(shape_id) > 128 or shape_id in seen:
            raise ValueError("標註識別碼重複或無效")
        seen.add(shape_id)
        shape["id"] = shape_id
        kind = shape.setdefault("type", "rectangle")
        shape["hidden"] = shape.get("hidden") is True
        finite = lambda n: type(n) in (int, float) and math.isfinite(n)
        if kind == "mask":
            counts = shape.get("counts")
            if (width * height > 16777216 or not isinstance(counts, list)
                    or len(counts) > width * height * 2 + 1
                    or any(type(n) is not int or n < 0 for n in counts)
                    or sum(counts) != width * height):
                raise ValueError("遮罩 RLE 與原圖尺寸不一致或超過 1,677 萬像素")
            shape.update(x=0, y=0, width=width, height=height)
        elif kind in {"polygon", "linestrip", "point", "obb"}:
            points = shape.get("points")
            minimum = 1 if kind == "point" else 2 if kind == "linestrip" else 3
            if (not isinstance(points, list) or not minimum <= len(points) <= 200000
                    or (kind == "point" and len(points) != 1)
                    or (kind == "obb" and len(points) != 4)):
                raise ValueError("標註頂點數量無效")
            if any(not isinstance(p, list) or len(p) != 2 or not all(finite(n) for n in p)
                   or not 0 <= p[0] <= width or not 0 <= p[1] <= height for p in points):
                raise ValueError("標註頂點超出原圖範圍")
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            shape.update(x=min(xs), y=min(ys), width=max(xs)-min(xs), height=max(ys)-min(ys))
        elif kind == "rectangle":
            values = [shape.get(key) for key in ("x", "y", "width", "height")]
            if not all(finite(n) for n in values):
                raise ValueError("矩形座標無效")
            x, y, w, h = values
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > width+1e-6 or y+h > height+1e-6:
                raise ValueError("矩形超出原圖範圍")
        else:
            raise ValueError(f"不支援的標註類型：{kind}")
        cleaned.append(shape)
    return cleaned


def repair_saved_mask_holes(shapes, width, height):
    """Apply the same conservative cleanup after editor changes."""
    repaired_shapes, audits = [], []
    for shape in shapes:
        if shape.get("type") != "mask":
            repaired_shapes.append(shape)
            continue
        metadata = shape.get("metadata") if isinstance(shape.get("metadata"), dict) else {}
        mask = decode_rle(shape["counts"], width, height)
        repaired, report = repair_tiny_holes(
            mask,
            protected_background_points=metadata.get("negative_points", []),
        )
        if not report["pixels_filled"]:
            repaired_shapes.append(shape)
            continue
        updated = json.loads(dump(shape))
        updated["counts"] = encode_rle(repaired)
        updated_metadata = updated.setdefault("metadata", {})
        updated_metadata["mask_cleanup"] = {**report, "stage": "editor_save"}
        audits.append({
            "shape_id": updated["id"],
            "label": updated["label"],
            **report,
        })
        repaired_shapes.append(updated)
    return repaired_shapes, audits


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Finish removals that Windows previously postponed because another
        # process briefly held an image or SQLite file open.
        for folder in self.root.glob(".deleting-*"):
            if folder.is_dir() and not folder.is_symlink():
                shutil.rmtree(folder, ignore_errors=True)

    def directory(self, project_id):
        project_id = identifier(project_id)
        direct = self.root / project_id
        candidates = [direct] if direct.exists() else [folder for folder in self.root.iterdir()
            if folder.is_dir() and folder.name.endswith("__" + project_id[:8])]
        result = candidates[0] if len(candidates) == 1 else direct
        if result.is_symlink() or result.resolve().parent != self.root:
            raise ValueError("專案路徑無效")
        return result

    @staticmethod
    def _folder_name(name, project_id):
        readable = re.sub(r'[<>:"/\\|?*／＼：＊？＂＜＞｜\x00-\x1f]+', "_", name).strip(" ._")[:60] or "project"
        return f"{readable}__{project_id[:8]}"

    def _rename_folder(self, project_id, name):
        current = self.directory(project_id)
        target = self.root / self._folder_name(name, project_id)
        if current == target or not current.exists(): return target
        # Windows can temporarily deny a directory rename while an image,
        # SQLite file, antivirus scan, or Explorer window has a handle open.
        # The readable folder name is cosmetic, so keep using the current
        # folder and retry on a later project listing instead of hiding the
        # project or failing an otherwise successful rename operation.
        if target.exists(): return current
        try:
            current.rename(target)
        except OSError:
            return current
        return target

    @contextmanager
    def connection(self, project_id, *, write=False):
        path = self.directory(project_id) / "project.sqlite3"
        if not path.is_file():
            raise FileNotFoundError("找不到專案")
        db = sqlite3.connect(path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        if db.execute("PRAGMA user_version").fetchone()[0] < CATALOG_SCHEMA_VERSION:
            ensure_catalog_schema(db)
            db.commit()
        try:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create_project(self, name):
        name = self._name(name)
        pid = uuid.uuid4().hex
        folder = self.root / self._folder_name(name, pid)
        folder.mkdir()
        (folder / "images").mkdir()
        db = sqlite3.connect(folder / "project.sqlite3")
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE project(id TEXT PRIMARY KEY,name TEXT NOT NULL,revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,updated_at TEXT NOT NULL,classes TEXT NOT NULL);
                CREATE TABLE assets(id TEXT PRIMARY KEY,name TEXT NOT NULL,width INTEGER NOT NULL,
                    height INTEGER NOT NULL,sha256 TEXT NOT NULL UNIQUE,image_file TEXT NOT NULL,
                    batch_id TEXT NOT NULL,split TEXT NOT NULL,source TEXT NOT NULL,
                    revision INTEGER NOT NULL,review_state TEXT NOT NULL,shapes TEXT NOT NULL,
                    created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
                CREATE TABLE history(id INTEGER PRIMARY KEY,asset_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,action TEXT NOT NULL,created_at TEXT NOT NULL,data TEXT NOT NULL);
                CREATE TABLE exports(id TEXT PRIMARY KEY,created_at TEXT NOT NULL,data TEXT NOT NULL);
            """)
            ensure_catalog_schema(db)
            now = timestamp()
            db.execute("INSERT INTO project VALUES(?,?,?,?,?,?)", (pid, name, 1, now, now, "[]"))
            db.commit()
        finally:
            db.close()
        return self.get_project(pid)

    @staticmethod
    def _catalog_relative_path(value):
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError("資產索引必須使用專案內的相對 POSIX 路徑")
        path = Path(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("資產索引路徑超出專案資料夾")
        return value

    def replace_artifact_catalog(self, project_id, catalog):
        """Atomically mirror validated on-disk manifests into relational rows."""
        required = {"datasets", "runs", "models", "model_exports", "predictions"}
        if not isinstance(catalog, dict) or not required.issubset(catalog):
            raise ValueError("專案資產索引無效")
        datasets = {row["id"]: row for row in catalog["datasets"]}
        runs = {row["id"]: row for row in catalog["runs"]}
        models = {row["id"]: row for row in catalog["models"]}
        for row in runs.values():
            if row["dataset_version_id"] not in datasets:
                raise ValueError(f"訓練 {row['id']} 指向不存在的資料版本")
        for row in models.values():
            if row["run_id"] not in runs or row["dataset_version_id"] not in datasets:
                raise ValueError(f"模型 {row['id']} 的來源關聯不存在")
        for collection in (catalog["model_exports"], catalog["predictions"]):
            for row in collection:
                if row["model_version_id"] not in models or row["run_id"] not in runs:
                    raise ValueError(f"資產 {row['id']} 的模型或訓練來源不存在")

        with self.connection(project_id, write=True) as db:
            for table in ("prediction_candidates", "model_exports", "model_versions", "training_runs", "dataset_versions"):
                db.execute(f"DELETE FROM {table}")
            db.executemany("""INSERT INTO dataset_versions
                (id,project_revision,created_at,manifest_sha256,asset_count,relative_path,source_dataset_id)
                VALUES(?,?,?,?,?,?,?)""", [(
                    row["id"], row["project_revision"], row["created_at"], row["manifest_sha256"],
                    row["asset_count"], self._catalog_relative_path(row["relative_path"]), row.get("source_dataset_id")
                ) for row in catalog["datasets"]])
            db.executemany("""INSERT INTO training_runs
                (id,dataset_version_id,proposed_model_id,engine,status,created_at,updated_at,relative_path)
                VALUES(?,?,?,?,?,?,?,?)""", [(
                    row["id"], row["dataset_version_id"], row.get("proposed_model_id"), row["engine"],
                    row["status"], row["created_at"], row["updated_at"], self._catalog_relative_path(row["relative_path"])
                ) for row in catalog["runs"]])
            db.executemany("""INSERT INTO model_versions
                (id,run_id,dataset_version_id,engine,created_at,relative_path) VALUES(?,?,?,?,?,?)""", [(
                    row["id"], row["run_id"], row["dataset_version_id"], row["engine"], row["created_at"],
                    self._catalog_relative_path(row["relative_path"])
                ) for row in catalog["models"]])
            db.executemany("""INSERT INTO model_exports
                (id,model_version_id,run_id,dataset_version_id,created_at,sha256,byte_count,relative_path)
                VALUES(?,?,?,?,?,?,?,?)""", [(
                    row["id"], row["model_version_id"], row["run_id"], row["dataset_version_id"], row["created_at"],
                    row["sha256"], row["byte_count"], self._catalog_relative_path(row["relative_path"])
                ) for row in catalog["model_exports"]])
            db.executemany("""INSERT INTO prediction_candidates
                (id,model_version_id,run_id,created_at,candidate_count,accepted_count,relative_path)
                VALUES(?,?,?,?,?,?,?)""", [(
                    row["id"], row["model_version_id"], row["run_id"], row["created_at"],
                    row["candidate_count"], row["accepted_count"], self._catalog_relative_path(row["relative_path"])
                ) for row in catalog["predictions"]])

    def artifact_catalog(self, project_id):
        with self.connection(project_id) as db:
            return {table: [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY id")]
                    for table in ("dataset_versions", "training_runs", "model_versions",
                                  "model_exports", "prediction_candidates")}

    def database_integrity(self, project_id):
        with self.connection(project_id) as db:
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            foreign_keys = [dict(row) for row in db.execute("PRAGMA foreign_key_check")]
            version = db.execute("PRAGMA user_version").fetchone()[0]
        return {"ok": integrity == "ok" and not foreign_keys, "integrity": integrity,
                "foreign_key_errors": foreign_keys, "schema_version": version}

    @staticmethod
    def _name(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 100:
            raise ValueError("專案名稱必須為 1–100 個字元")
        return value.strip()

    def list_projects(self):
        rows = []
        for folder in list(self.root.iterdir()):
            if (folder.name.startswith(".deleting-") or not folder.is_dir()
                    or folder.is_symlink() or not (folder / "project.sqlite3").is_file()):
                continue
            try:
                # sqlite3.Connection's context manager commits or rolls back,
                # but does not close the database handle. Close it before a
                # possible folder rename so Windows does not reject the move.
                with closing(sqlite3.connect(folder / "project.sqlite3")) as db:
                    row = db.execute("SELECT id,name FROM project").fetchone()
                if not row or not re.fullmatch(r"[a-f0-9]{32}", row[0]): continue
                if folder.name != self._folder_name(row[1], row[0]): self._rename_folder(row[0], row[1])
                project = self.get_project(row[0], include_assets=False)
                rows.append(project)
            except (sqlite3.Error, ValueError, FileNotFoundError):
                continue
        return sorted(rows, key=lambda p: p["updated_at"], reverse=True)

    @staticmethod
    def _touch(db):
        db.execute("UPDATE project SET revision=revision+1,updated_at=?", (timestamp(),))

    def _asset(self, row, pid, *, detail=False, internal=False):
        asset = dict(row)
        shapes = json.loads(asset.pop("shapes"))
        asset["source"] = json.loads(asset["source"])
        asset["shape_count"] = len(shapes)
        asset["class_counts"] = dict(Counter(
            str(shape.get("label")) for shape in shapes if shape.get("label")
        ))
        asset["url"] = f"/api/projects/{pid}/assets/{asset['id']}/image"
        image_file = asset.pop("image_file")
        if detail:
            asset["shapes"] = shapes
        if internal:
            asset["image_path"] = str(self.directory(pid) / "images" / image_file)
        return asset

    def get_project(self, project_id, *, include_assets=True, internal=False):
        with self.connection(project_id) as db:
            project = dict(db.execute("SELECT * FROM project").fetchone())
            project["classes"] = json.loads(project["classes"])
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='split_plans'").fetchone():
                plan = db.execute("SELECT data FROM split_plans WHERE id=1").fetchone()
                if plan:
                    project["split_plan"] = json.loads(plan["data"])
                    project["split_plan"]["current"] = project["split_plan"]["project_revision"] == project["revision"]
            stats = {"total": 0, "pending": 0, "approved": 0, "rejected": 0}
            for row in db.execute("SELECT review_state,COUNT(*) AS n FROM assets GROUP BY review_state"):
                stats[row["review_state"]] = row["n"]
                stats["total"] += row["n"]
            project["stats"] = stats
            project.update(stats)
            if include_assets:
                project["assets"] = [self._asset(row, project_id, detail=internal, internal=internal)
                                     for row in db.execute("SELECT * FROM assets ORDER BY created_at,rowid")]
            project["exports"] = [dict(json.loads(row["data"]), id=row["id"], created_at=row["created_at"])
                                  for row in db.execute("SELECT * FROM exports ORDER BY created_at DESC")]
            if internal:
                project["schema_version"] = 1
                project["snapshot_at"] = timestamp()
            return project

    def snapshot(self, project_id):
        return self.get_project(project_id, internal=True)

    def update_project(self, project_id, *, name=None, classes=None):
        with self.connection(project_id, write=True) as db:
            if name is not None:
                db.execute("UPDATE project SET name=?", (self._name(name),))
            if classes is not None:
                if not isinstance(classes, list):
                    raise ValueError("類別清單無效")
                names = list(dict.fromkeys(clean_label(s) for s in classes))
                used = set()
                for row in db.execute("SELECT shapes FROM assets"):
                    for shape in json.loads(row[0]):
                        used.add(shape["label"])
                removed_used = sorted(used.difference(names))
                if removed_used:
                    joined = "、".join(removed_used[:5])
                    raise ValueError(f"類別「{joined}」仍有標註使用；請在類別管理中指定替代類別")
                db.execute("UPDATE project SET classes=?", (dump(names),))
            self._touch(db)
        if name is not None:
            self._rename_folder(project_id, self._name(name))
        return self.get_project(project_id)

    def class_usage(self, project_id):
        """Return the persisted class list and actual annotation usage."""
        with self.connection(project_id) as db:
            project = db.execute("SELECT revision,classes FROM project").fetchone()
            names = json.loads(project["classes"])
            counts = {name: {"object_count": 0, "asset_ids": set()} for name in names}
            for row in db.execute("SELECT id,shapes FROM assets"):
                for shape in json.loads(row["shapes"]):
                    item = counts.setdefault(shape["label"], {"object_count": 0, "asset_ids": set()})
                    item["object_count"] += 1
                    item["asset_ids"].add(row["id"])
            ordered = names + sorted(name for name in counts if name not in names)
            return {"revision": project["revision"], "classes": [
                {"name": name, "object_count": counts[name]["object_count"],
                 "image_count": len(counts[name]["asset_ids"])} for name in ordered]}

    def require_class(self, project_id, label):
        label = clean_label(label)
        with self.connection(project_id) as db:
            names = json.loads(db.execute("SELECT classes FROM project").fetchone()[0])
        if label not in names:
            raise ValueError(f"類別「{label}」尚未由使用者建立；請先到類別管理新增")
        return label

    def manage_classes(self, project_id, classes, replacements, revision, delete_objects=None):
        """Persist classes and resolve every used removal by relabeling or deletion."""
        if not isinstance(classes, list) or not isinstance(replacements, dict) or not isinstance(delete_objects or [], list):
            raise ValueError("類別管理內容無效")
        names = list(dict.fromkeys(clean_label(value) for value in classes))
        delete_set = {clean_label(value) for value in (delete_objects or [])}
        if delete_set.intersection(names):
            raise ValueError("保留的類別不能同時刪除其標註物件")
        replacement_map = {}
        for source, target in replacements.items():
            source, target = clean_label(source), clean_label(target)
            if target not in names:
                raise ValueError(f"替代類別「{target}」不在儲存後的類別清單中")
            if source in names:
                raise ValueError(f"保留的類別「{source}」不需要指定替代類別")
            if source != target:
                replacement_map[source] = target

        affected_assets, changed_objects, deleted_objects = [], 0, 0
        with self.connection(project_id, write=True) as db:
            project = db.execute("SELECT revision,classes FROM project").fetchone()
            if type(revision) is not int or project["revision"] != revision:
                raise ConflictError("專案類別已有新版本；請重新開啟類別管理")
            old_names = json.loads(project["classes"])
            rows = list(db.execute("SELECT id,width,height,revision,review_state,shapes FROM assets"))
            used = {shape["label"] for row in rows for shape in json.loads(row["shapes"])}
            removed_used = sorted(used.difference(names))
            missing = [name for name in removed_used if name not in replacement_map and name not in delete_set]
            if missing:
                joined = "、".join(missing[:5])
                raise ValueError(f"類別「{joined}」仍有標註使用，請指定替代類別")

            for row in rows:
                shapes = json.loads(row["shapes"])
                changed = deleted = 0
                kept = []
                for shape in shapes:
                    if shape["label"] in delete_set:
                        deleted += 1
                        continue
                    if shape["label"] in replacement_map:
                        shape["label"] = replacement_map[shape["label"]]
                        changed += 1
                    kept.append(shape)
                if not changed and not deleted:
                    continue
                shapes = clean_shapes(kept, row["width"], row["height"])
                next_revision = row["revision"] + 1
                db.execute("UPDATE assets SET shapes=?,revision=?,review_state='pending',updated_at=? WHERE id=?",
                           (dump(shapes), next_revision, timestamp(), row["id"]))
                action = "class_delete" if deleted and not changed else "class_relabel" if changed and not deleted else "class_management"
                self._history(db, row["id"], next_revision, action,
                              {"shapes": shapes, "review_state": "pending",
                               "replacements": replacement_map,
                               "deleted_classes": sorted(delete_set)})
                affected_assets.append(row["id"])
                changed_objects += changed
                deleted_objects += deleted

            db.execute("UPDATE project SET classes=?", (dump(names),))
            self._touch(db)

        result = self.get_project(project_id)
        result["class_update"] = {
            "added": [name for name in names if name not in old_names],
            "removed": [name for name in old_names if name not in names],
            "changed_objects": changed_objects,
            "deleted_objects": deleted_objects,
            "affected_images": len(affected_assets),
            "asset_ids": affected_assets,
        }
        return result

    def delete_project(self, project_id, revision):
        """Remove a complete local project after checking its visible revision."""
        project_id = identifier(project_id)
        folder = self.directory(project_id)
        with self.connection(project_id) as db:
            row = db.execute("SELECT name,revision FROM project").fetchone()
            if row is None:
                raise FileNotFoundError("找不到專案")
            if type(revision) is not int or row["revision"] != revision:
                raise ConflictError("專案已有新版本；請重新載入後再刪除")
            name = row["name"]

        tombstone = self.root / f".deleting-{project_id}-{uuid.uuid4().hex[:8]}"
        if tombstone.resolve().parent != self.root:
            raise ValueError("專案刪除路徑無效")
        try:
            folder.rename(tombstone)
        except OSError as exc:
            raise ValueError("專案檔案正被其他程式使用，請關閉相關檔案或資料夾後再試一次") from exc

        cleanup_pending = False
        try:
            shutil.rmtree(tombstone)
        except OSError:
            # The project is already detached from the library. A later app
            # start retries physical cleanup after transient Windows locks end.
            cleanup_pending = True
        return {"deleted": True, "project_id": project_id, "name": name,
                "cleanup_pending": cleanup_pending}

    def get_asset(self, project_id, asset_id, *, internal=False):
        with self.connection(project_id) as db:
            row = db.execute("SELECT * FROM assets WHERE id=?", (identifier(asset_id),)).fetchone()
            if row is None:
                raise FileNotFoundError("找不到圖片")
            return self._asset(row, project_id, detail=True, internal=internal)

    def image_path(self, project_id, asset_id):
        return Path(self.get_asset(project_id, asset_id, internal=True)["image_path"])

    def delete_asset(self, project_id, asset_id, revision):
        return self.delete_assets(project_id, [asset_id], {asset_id: revision})

    def delete_assets(self, project_id, asset_ids, revisions):
        """Atomically remove project assets, then unlink their private copies."""
        if not isinstance(asset_ids, list) or not asset_ids or len(asset_ids) > 10000:
            raise ValueError("請選擇 1–10,000 張要刪除的圖片")
        if not isinstance(revisions, dict):
            raise ValueError("刪除圖片需要有效的修訂版本")
        ids = list(dict.fromkeys(identifier(value) for value in asset_ids))
        image_files = []
        with self.connection(project_id, write=True) as db:
            for asset_id in ids:
                row = db.execute("SELECT revision,image_file FROM assets WHERE id=?", (asset_id,)).fetchone()
                if row is None:
                    raise FileNotFoundError("找不到要刪除的圖片")
                if type(revisions.get(asset_id)) is not int or row["revision"] != revisions[asset_id]:
                    raise ConflictError("部分圖片已有新版本；請重新載入後再刪除")
                image_files.append(row["image_file"])
            placeholders = ",".join("?" for _ in ids)
            db.execute(f"DELETE FROM history WHERE asset_id IN ({placeholders})", ids)
            db.execute(f"DELETE FROM assets WHERE id IN ({placeholders})", ids)
            self._touch(db)
        images = (self.directory(project_id) / "images").resolve()
        for image_file in image_files:
            image = (images / image_file).resolve()
            if image.parent == images and not image.is_symlink():
                try:
                    image.unlink(missing_ok=True)
                except OSError:
                    # The database no longer references this immutable file. It
                    # may remain orphaned until a later cleanup if Windows holds it.
                    pass
        return dict(deleted=len(ids), asset_ids=ids, project=self.get_project(project_id))

    def add_assets(self, project_id, records, *, progress=None):
        """Validate every record before one transaction publishes the batch.

        Files can remain unreferenced after an interrupted batch, but a partial
        project/annotation batch is never visible. No original file is changed.
        """
        folder = self.directory(project_id)
        if not (folder / "project.sqlite3").is_file():
            raise FileNotFoundError("找不到專案")
        prepared = []
        default_batch = uuid.uuid4().hex
        for index, record in enumerate(records):
            path = Path(record["path"])
            if not path.is_file() or path.stat().st_size > 128*1024*1024:
                raise ValueError(f"圖片不存在或超過 128 MiB：{path.name}")
            raw = path.read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                image.verify()
                extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif"}.get(image.format)
            if not extension:
                raise ValueError(f"不支援的圖片格式：{path.name}")
            shapes = clean_shapes(record.get("shapes", []), width, height)
            digest = hashlib.sha256(raw).hexdigest()
            target = folder / "images" / (digest + extension)
            if target.parent.is_symlink() or target.is_symlink():
                raise ValueError("圖片儲存路徑不得為連結")
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError("已存原圖雜湊不符，請先還原備份")
            else:
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as f:
                    temporary = Path(f.name)
                    f.write(raw)
                    f.flush()
                    import os
                    os.fsync(f.fileno())
                temporary.replace(target)
            state = record.get("review_state", "pending")
            if state not in {"pending", "approved", "rejected"}:
                raise ValueError("圖片審核狀態無效")
            split = record.get("split") or ""
            if split not in {"", "train", "val", "test"}:
                raise ValueError("資料集分組無效")
            source = record.get("source") or {"path": str(path.resolve())}
            prepared.append(dict(id=uuid.uuid4().hex, name=record.get("name") or path.name,
                                 width=width, height=height, sha256=digest, image_file=target.name,
                                 batch_id=str(record.get("batch_id") or default_batch), split=split,
                                 source=dump(source), review_state=state, shapes=shapes))
            if progress:
                progress(index+1, len(records))
        added, duplicates, conflicts = [], [], []
        with self.connection(project_id, write=True) as db:
            names = json.loads(db.execute("SELECT classes FROM project").fetchone()[0])
            for asset in prepared:
                existing = db.execute("SELECT id,name,shapes FROM assets WHERE sha256=?", (asset["sha256"],)).fetchone()
                if existing:
                    duplicates.append(existing["id"])
                    # Same pixels cannot silently replace a user's annotations.
                    def semantic(rows):
                        return [{k:v for k,v in shape.items() if k not in {"id", "hidden"}} for shape in rows]
                    if semantic(json.loads(existing["shapes"])) != semantic(asset["shapes"]):
                        conflicts.append(f"{asset['name']}：原圖已存在且標註不同，保留已存版本")
                    continue
                now = timestamp()
                db.execute("INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (asset["id"], asset["name"], asset["width"], asset["height"], asset["sha256"],
                            asset["image_file"], asset["batch_id"], asset["split"], asset["source"],
                            1, asset["review_state"], dump(asset["shapes"]), now, now))
                self._history(db, asset["id"], 1, "import", {"shapes":asset["shapes"], "review_state":asset["review_state"], "source":json.loads(asset["source"])})
                for shape in asset["shapes"]:
                    if shape["label"] not in names:
                        names.append(shape["label"])
                added.append(asset["id"])
            db.execute("UPDATE project SET classes=?", (dump(names),))
            self._touch(db)
        return {"added":len(added), "asset_ids":added, "duplicates":len(duplicates), "conflicts":conflicts,
                "project":self.get_project(project_id)}

    @staticmethod
    def _history(db, asset_id, revision, action, data):
        db.execute("INSERT INTO history(asset_id,revision,action,created_at,data) VALUES(?,?,?,?,?)",
                   (asset_id, revision, action, timestamp(), dump(data)))

    def save_asset(self, project_id, asset_id, shapes, revision):
        with self.connection(project_id, write=True) as db:
            asset = db.execute("SELECT * FROM assets WHERE id=?", (identifier(asset_id),)).fetchone()
            if not asset:
                raise FileNotFoundError("找不到圖片")
            if type(revision) is not int or revision != asset["revision"]:
                raise ConflictError("圖片已有新版本；保留目前編輯，請重新載入後整合修改")
            cleaned = clean_shapes(shapes, asset["width"], asset["height"])
            cleaned, mask_repairs = repair_saved_mask_holes(
                cleaned, asset["width"], asset["height"],
            )
            names = json.loads(db.execute("SELECT classes FROM project").fetchone()[0])
            unknown = list(dict.fromkeys(shape["label"] for shape in cleaned if shape["label"] not in names))
            if unknown:
                raise ValueError(f"類別「{'、'.join(unknown[:5])}」尚未由使用者建立；請先到類別管理新增")
            serialized = dump(cleaned)
            if serialized != asset["shapes"]:
                next_revision = revision+1
                db.execute("UPDATE assets SET shapes=?,revision=?,review_state='pending',updated_at=? WHERE id=?",
                           (serialized, next_revision, timestamp(), asset_id))
                self._history(db, asset_id, next_revision, "edit", {"shapes":cleaned,"review_state":"pending"})
                self._touch(db)
        result = self.get_asset(project_id, asset_id)
        if mask_repairs:
            result["mask_cleanup"] = {
                "shapes_repaired": len(mask_repairs),
                "holes_filled": sum(item["holes_filled"] for item in mask_repairs),
                "pixels_filled": sum(item["pixels_filled"] for item in mask_repairs),
                "repairs": mask_repairs,
            }
        return result

    def review(self, project_id, asset_ids, state, revisions=None, reason='', note=''):
        if not isinstance(reason, str) or not isinstance(note, str) or len(reason) > 100 or len(note) > 2000:
            raise ValueError('審核原因或備註無效')
        if state not in {"pending", "approved", "rejected"} or not isinstance(asset_ids, list) or not asset_ids:
            raise ValueError("請選擇圖片及有效的審核狀態")
        with self.connection(project_id, write=True) as db:
            for aid in dict.fromkeys(asset_ids):
                row = db.execute("SELECT * FROM assets WHERE id=?", (identifier(aid),)).fetchone()
                if not row:
                    raise FileNotFoundError("找不到待審核圖片")
                if revisions is not None and revisions.get(aid) != row["revision"]:
                    raise ConflictError("圖片已被修改，請載入最新標註後重新核准")
                shapes = clean_shapes(json.loads(row["shapes"]), row["width"], row["height"])
                if state == "approved":
                    from composer_core.geometry import validate_shape
                    for shape in shapes:
                        validate_shape(shape, row["width"], row["height"])
                # Review is itself a revision; a concurrent stale edit cannot undo it.
                next_revision = row["revision"]+1
                source = json.loads(row['source'])
                source['review'] = {'reason': reason if state != 'approved' else '', 'note': note,
                                    'needs_correction': state == 'pending' and reason == '待修正'}
                db.execute("UPDATE assets SET review_state=?,revision=?,updated_at=?,source=? WHERE id=?",
                           (state, next_revision, timestamp(), dump(source), aid))
                self._history(db, aid, next_revision, "review", {"review_state":state,"shapes":json.loads(row["shapes"]), 'review': source['review']})
            self._touch(db)
        return self.get_project(project_id)

    def history(self, project_id, asset_id):
        with self.connection(project_id) as db:
            return [dict(row, data=json.loads(row["data"])) for row in db.execute(
                "SELECT * FROM history WHERE asset_id=? ORDER BY id", (identifier(asset_id),))]

    def restore(self, project_id, asset_id, history_id, revision):
        history = self.history(project_id, asset_id)
        target = next((item for item in history if item["id"] == history_id), None)
        if target is None:
            raise FileNotFoundError("找不到歷史版本")
        return self.save_asset(project_id, asset_id, target["data"]["shapes"], revision)

    def record_export(self, project_id, result):
        eid = uuid.uuid4().hex
        data = json.loads(dump(result))
        with self.connection(project_id, write=True) as db:
            db.execute("INSERT INTO exports VALUES(?,?,?)", (eid, timestamp(), dump(data)))
            self._touch(db)
        return dict(data, id=eid)

    def assign(self, project_id, asset_ids, *, batch_id=None, split=None):
        if not isinstance(asset_ids, list) or not asset_ids:
            raise ValueError("請先選擇圖片")
        if batch_id is not None and (not isinstance(batch_id, str) or not batch_id.strip() or len(batch_id)>200):
            raise ValueError("拍攝批次名稱無效")
        if split is not None and split not in {"", "train", "val", "test"}:
            raise ValueError("分組必須為 train、val 或 test")
        with self.connection(project_id, write=True) as db:
            for aid in dict.fromkeys(asset_ids):
                row = db.execute("SELECT * FROM assets WHERE id=?", (identifier(aid),)).fetchone()
                if not row:
                    raise FileNotFoundError("找不到圖片")
                batch = batch_id.strip() if batch_id is not None else row["batch_id"]
                group = split if split is not None else row["split"]
                db.execute("UPDATE assets SET batch_id=?,split=?,revision=revision+1,updated_at=? WHERE id=?",
                           (batch, group, timestamp(), aid))
                self._history(db, aid, row["revision"]+1, "assign", {"shapes":json.loads(row["shapes"]),
                              "review_state":row["review_state"],"batch_id":batch,"split":group})
            self._touch(db)
        return self.get_project(project_id)

    def split_info(self, project_id):
        from .smart_splitting import source_groups
        project = self.snapshot(project_id)
        assets = [a for a in project["assets"] if a["review_state"] == "approved"]
        ids = {a["id"] for a in assets}
        overrides = {k:v for k,v in project.get("split_plan",{}).get("options",{}).get("group_overrides",{}).items() if k in ids}
        return {"project_revision":project["revision"], "groups":source_groups(assets,overrides),
                "assets":[{k:a[k] for k in ("id","name","batch_id","url")} for a in assets],
                "previous":project.get("split_plan")}

    def preview_split(self, project_id, options=None):
        from .smart_splitting import smart_split
        project = self.snapshot(project_id)
        assets = [a for a in project["assets"] if a["review_state"] == "approved"]
        plan = smart_split(assets, options)
        plan["project_revision"] = project["revision"]
        plan["fingerprint"] = hashlib.sha256(dump(plan).encode()).hexdigest()
        plan["assets"] = [{k: a[k] for k in ("id", "name", "batch_id", "url")} for a in assets]
        return plan

    def apply_split(self, project_id, options, revision, fingerprint):
        plan = self.preview_split(project_id, options)
        if type(revision) is not int or revision != plan["project_revision"] or fingerprint != plan["fingerprint"]:
            raise ConflictError("資料或分割設定已變動，請重新預覽")
        if plan.get("blockers"):
            raise ValueError("；".join(item["message"] for item in plan["blockers"]))
        with self.connection(project_id, write=True) as db:
            current = db.execute("SELECT revision FROM project").fetchone()["revision"]
            if current != revision:
                raise ConflictError("預覽後資料已變動，請重新預覽")
            for aid, split in plan["assignments"].items():
                row = db.execute("SELECT * FROM assets WHERE id=?", (aid,)).fetchone()
                if row["split"] == split: continue
                next_revision = row["revision"] + 1
                db.execute("UPDATE assets SET split=?,revision=?,updated_at=? WHERE id=?", (split,next_revision,timestamp(),aid))
                self._history(db, aid, next_revision, "smart_split", {"shapes":json.loads(row["shapes"]),
                    "review_state":row["review_state"], "batch_id":row["batch_id"], "split":split})
            self._touch(db)
            plan["project_revision"] = revision + 1
            plan["applied_at"] = timestamp()
            plan.pop("assets", None)
            db.execute("CREATE TABLE IF NOT EXISTS split_plans(id INTEGER PRIMARY KEY,data TEXT NOT NULL)")
            db.execute("INSERT OR REPLACE INTO split_plans VALUES(1,?)", (dump(plan),))
        return {"project":self.get_project(project_id), "report":plan}

    def auto_split(self, project_id, ratios):
        from .splitting import stratified_split

        with self.connection(project_id, write=True) as db:
            rows = list(db.execute("SELECT * FROM assets WHERE review_state='approved' ORDER BY created_at,rowid"))
            assets = [dict(id=row["id"], sha256=row["sha256"], shapes=json.loads(row["shapes"])) for row in rows]
            assignments, report = stratified_split(assets, ratios)
            project_classes = json.loads(db.execute("SELECT classes FROM project").fetchone()["classes"])
            for class_name in project_classes:
                if not report["class_totals"].get(class_name):
                    report["warnings"].append(f"類別「{class_name}」沒有已核准標註，無法參與平衡分割")
            changed = 0
            now = timestamp()
            for row in rows:
                split = assignments[row["id"]]
                if row["split"] == split:
                    continue
                revision = row["revision"] + 1
                db.execute("UPDATE assets SET split=?,revision=?,updated_at=? WHERE id=?",
                           (split, revision, now, row["id"]))
                self._history(db, row["id"], revision, "auto_split", {
                    "shapes": json.loads(row["shapes"]), "review_state": row["review_state"],
                    "batch_id": row["batch_id"], "split": split,
                })
                changed += 1
            if changed:
                self._touch(db)
        report["changed"] = changed
        return {"project": self.get_project(project_id), "report": report}

    def merge(self, project_id, source_ids):
        if not isinstance(source_ids, list) or not source_ids or project_id in source_ids:
            raise ValueError("請選擇其他專案作為整併來源")
        records = []
        for source_id in dict.fromkeys(source_ids):
            project = self.snapshot(source_id)
            for asset in project["assets"]:
                records.append(dict(path=asset["image_path"], name=asset["name"], shapes=asset["shapes"],
                    batch_id=asset["batch_id"], split=asset["split"], review_state=asset["review_state"],
                    source={"project_id":source_id,"asset_id":asset["id"],"revision":asset["revision"],
                            "original":asset["source"]}))
        return self.add_assets(project_id, records)
