"""Project-scoped filesystem layout and one-time legacy migration."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import file_digest
import json
import os
from pathlib import Path

from .file_lock import exclusive_file_lock
from .project_layout import AREA_PATHS, migrate_project_layout, project_area

PROJECT_AREAS = ("datasets", "runs", "models", "model-exports", "predictions")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ProjectStorage:
    """Resolve durable artifacts below the owning ProjectStore directory.

    Large images and model files stay on disk. SQLite keeps their relational
    catalog; relative paths always resolve inside this project container.
    """

    def __init__(self, data_root: Path, store):
        self.data_root = Path(data_root).resolve()
        self.store = store

    def root(self, project_id: str) -> Path:
        return self.store.directory(project_id)

    def area(self, project_id: str, name: str, *, create: bool = False) -> Path:
        path = project_area(self.root(project_id), name)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _move_tree(source: Path, destination: Path, report: dict) -> None:
        if not source.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                source.replace(destination)
            except OSError:
                if source.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    report["conflicts"].append({"from": str(source), "to": str(destination),
                                                "reason": "path_locked"})
                    return
            else:
                report["moved"].append({"from": str(source), "to": str(destination)})
                return
        if not source.is_dir() or not destination.is_dir():
            report["conflicts"].append({"from": str(source), "to": str(destination)})
            return
        for item in list(source.iterdir()):
            target = destination / item.name
            if target.exists():
                report["conflicts"].append({"from": str(item), "to": str(target)})
            else:
                try:
                    item.replace(target)
                    report["moved"].append({"from": str(item), "to": str(target)})
                except OSError:
                    report["conflicts"].append({"from": str(item), "to": str(target),
                                                "reason": "path_locked"})
        try:
            source.rmdir()
        except OSError:
            pass

    def _deduplicate_dataset_images(self, project_id: str, report: dict) -> None:
        """Replace verified legacy copies with hard links to immutable project images."""
        project_images = self.area(project_id, "images")
        if not project_images.is_dir():
            return
        by_hash = {path.stem: path for path in project_images.iterdir()
                   if path.is_file() and not path.is_symlink()}
        datasets = self.area(project_id, "datasets")
        for manifest_path in datasets.glob("D*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            dataset_root = manifest_path.parent.resolve()
            for asset in manifest.get("assets", []):
                digest = asset.get("sha256")
                source = by_hash.get(digest)
                destination = (dataset_root / str(asset.get("image_file", ""))).resolve()
                if (source is None or not destination.is_file()
                        or not destination.is_relative_to(dataset_root)):
                    continue
                temporary = None
                try:
                    if os.path.samefile(source, destination):
                        continue
                    with destination.open("rb") as stream:
                        actual = file_digest(stream, "sha256").hexdigest()
                    if actual != digest:
                        report["conflicts"].append({"path": str(destination), "reason": "sha256_mismatch"})
                        continue
                    with source.open("rb") as stream:
                        source_digest = file_digest(stream, "sha256").hexdigest()
                    if source_digest != digest:
                        report["conflicts"].append({"path": str(source), "reason": "source_sha256_mismatch"})
                        continue
                    temporary = destination.with_name(f".{destination.name}.link.tmp")
                    temporary.unlink(missing_ok=True)
                    os.link(source, temporary)
                    temporary.replace(destination)
                    report["deduplicated_files"] += 1
                    report["deduplicated_bytes"] += destination.stat().st_size
                except OSError:
                    if temporary is not None:
                        try:
                            temporary.unlink(missing_ok=True)
                        except OSError:
                            pass

    def migrate(self) -> list[dict]:
        """Move legacy project-owned artifacts without overwriting collisions."""
        projects = self.store.list_projects()
        reports = []
        legacy_links_path = self.data_root / "cvat" / "project-links.json"
        try:
            legacy_links = json.loads(legacy_links_path.read_text(encoding="utf-8"))
            if not isinstance(legacy_links, dict):
                legacy_links = {}
        except (OSError, ValueError):
            legacy_links = {}
        links_changed = False

        for project in projects:
            project_id = project["id"]
            report = {"project_id": project_id, "moved": [], "conflicts": [],
                      "deduplicated_files": 0, "deduplicated_bytes": 0}
            project_root = self.root(project_id)
            with (exclusive_file_lock(project_root / '.training-admission.lock'),
                  exclusive_file_lock(project_root / '.dataset-publish.lock'),
                  exclusive_file_lock(project_root / '.images.lock')):
                self._migrate_project_layout(project_root, report)
            for name in PROJECT_AREAS:
                destination = self.area(project_id, name)
                if name == "predictions":
                    destination.mkdir(parents=True, exist_ok=True)
                    legacy = self.data_root / name
                    if legacy.is_dir():
                        for item in list(legacy.glob("*.json")):
                            try:
                                record = json.loads(item.read_text(encoding="utf-8"))
                            except (OSError, ValueError):
                                continue
                            if record.get("project_id") == project_id:
                                self._move_tree(item, destination / item.name, report)
                    continue
                self._move_tree(self.data_root / name / project_id, destination, report)

            self._move_tree(self.data_root / "labelme" / project_id,
                            self.area(project_id, "integrations/labelme"), report)

            cvat_dir = self.area(project_id, "integrations/cvat", create=True)
            if project_id in legacy_links and not (cvat_dir / "project-link.json").exists():
                _write_json(cvat_dir / "project-link.json", legacy_links[project_id])
                task_id = legacy_links[project_id].get("task_id")
                backup_root = self.data_root / "cvat" / "annotation-backups"
                if task_id is not None and backup_root.is_dir():
                    for backup in list(backup_root.glob(f"task-{task_id}-*.json")):
                        self._move_tree(backup, cvat_dir / "annotation-backups" / backup.name, report)
                legacy_links.pop(project_id, None)
                links_changed = True
            self._move_tree(self.data_root / "cvat" / f"sync-{project_id}.json",
                            cvat_dir / "sync.json", report)

            for name in PROJECT_AREAS:
                self.area(project_id, name, create=True)
            self.area(project_id, "integrations/labelme", create=True)
            self._deduplicate_dataset_images(project_id, report)
            report["completed_at"] = datetime.now(timezone.utc).isoformat()
            layout_path = project_root / "storage-layout.json"
            try:
                previous = json.loads(layout_path.read_text(encoding="utf-8"))
                migrations = previous.get("migrations", []) if isinstance(previous, dict) else []
            except (OSError, ValueError):
                migrations = []
            if report["moved"] or report["conflicts"] or report["deduplicated_files"]:
                migrations = [*migrations, report]
            _write_json(layout_path, {
                "schema_version": 2,
                "layout": "project-scoped",
                "areas": {name: self.area(project_id, name).relative_to(project_root).as_posix()
                          for name in (*AREA_PATHS, "integrations/labelme", "integrations/cvat")},
                "last_verified_at": report["completed_at"],
                "migrations": migrations,
            })
            reports.append(report)

        if links_changed:
            _write_json(legacy_links_path, legacy_links)
        return reports

    @staticmethod
    def _migrate_project_layout(root, report):
        from .run_events import read_state

        if not any((root / name).exists() or (root / name).is_symlink() for name in AREA_PATHS):
            return
        active = {'queued', 'preparing', 'running', 'stopping'}
        for run_path in project_area(root, 'runs').glob('R*/run.json'):
            try:
                run = read_state(run_path)
                terminal = run_path.parent / 'control' / 'terminal.json'
                if run.get('status') in active and terminal.is_file():
                    run = json.loads(terminal.read_text(encoding='utf-8'))
                if run.get('status') in {'completed', 'failed', 'stopped'}:
                    continue
            except (OSError, ValueError, AttributeError):
                pass
            report['conflicts'].append({'path': str(run_path), 'reason': 'active_or_unverified_run'})
            return
        migrate_project_layout(root, report)
