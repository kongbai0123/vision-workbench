"""Safe repair for SAM2 sessions renamed as human-readable collections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .training_dataset import Sam2TrainingDatasetSession


SAM2_DATASET_TYPE = "sam2_training_pseudo_labels"


@dataclass(frozen=True)
class Sam2LayoutRepairPlan:
    """One flat, renamed session that can become collection/session_id."""

    source_directory: Path
    collection_name: str
    session_id: str
    target_directory: Path
    manifest_sha256: str


def _read_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid SAM2 manifest: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"SAM2 manifest must contain an object: {path}")
    return document, payload


def find_misnamed_sam2_sessions(
    root_directory: str | os.PathLike[str],
) -> tuple[Sam2LayoutRepairPlan, ...]:
    """Find direct child sessions whose readable folder replaced session_id."""

    root = Path(root_directory).expanduser().resolve()
    if not root.is_dir():
        return ()
    plans: list[Sam2LayoutRepairPlan] = []
    session_ids: set[str] = set()
    for source in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        manifest_path = source / "manifest.json"
        if not source.is_dir() or not manifest_path.is_file():
            continue
        document, payload = _read_manifest(manifest_path)
        if document.get("dataset_type") != SAM2_DATASET_TYPE:
            continue
        session_id = document.get("session_id")
        if (
            not isinstance(session_id, str)
            or not session_id.startswith("session_")
            or Path(session_id).name != session_id
        ):
            raise ValueError(f"invalid SAM2 session_id in {manifest_path}")
        if session_id in session_ids:
            raise ValueError(f"duplicate SAM2 session_id: {session_id}")
        session_ids.add(session_id)
        if source.name == session_id:
            continue
        target = (source / session_id).resolve()
        if target.exists():
            raise FileExistsError(f"SAM2 repair target already exists: {target}")
        if any(source.rglob(".review.lock")):
            raise RuntimeError(f"SAM2 session is currently in use: {source}")
        plans.append(
            Sam2LayoutRepairPlan(
                source_directory=source.resolve(),
                collection_name=source.name,
                session_id=session_id,
                target_directory=target,
                manifest_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return tuple(plans)


def _restore_repaired_plan(plan: Sam2LayoutRepairPlan) -> None:
    collection = plan.source_directory
    target = plan.target_directory
    if not target.is_dir():
        raise RuntimeError(f"cannot roll back missing repaired session: {target}")
    children = tuple(target.iterdir())
    for child in children:
        destination = collection / child.name
        if destination.exists():
            raise FileExistsError(f"cannot roll back over existing path: {destination}")
        os.replace(child, destination)
    target.rmdir()


def _apply_repair_plan(plan: Sam2LayoutRepairPlan) -> None:
    """Move session contents down one level without renaming the open parent."""

    source = plan.source_directory
    target = plan.target_directory
    children = tuple(source.iterdir())
    target.mkdir()
    moved: list[tuple[Path, Path]] = []
    try:
        for child in children:
            destination = target / child.name
            if destination.exists():
                raise FileExistsError(f"SAM2 repair destination exists: {destination}")
            os.replace(child, destination)
            moved.append((child, destination))
        Sam2TrainingDatasetSession.open(target)
    except BaseException:
        for original, destination in reversed(moved):
            if destination.exists():
                os.replace(destination, original)
        if target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        raise


def repair_misnamed_sam2_sessions(
    root_directory: str | os.PathLike[str],
) -> tuple[Path, ...]:
    """Transactionally organize flat renamed sessions and validate each result."""

    plans = find_misnamed_sam2_sessions(root_directory)
    completed: list[Sam2LayoutRepairPlan] = []
    try:
        for plan in plans:
            source = plan.source_directory
            manifest_path = source / "manifest.json"
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != plan.manifest_sha256:
                raise RuntimeError(f"SAM2 manifest changed during repair: {manifest_path}")
            _apply_repair_plan(plan)
            completed.append(plan)
    except BaseException:
        rollback_errors: list[str] = []
        for plan in reversed(completed):
            try:
                _restore_repaired_plan(plan)
            except BaseException as error:
                rollback_errors.append(f"{plan.collection_name}: {error}")
        if rollback_errors:
            raise RuntimeError(
                "SAM2 layout repair failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            )
        raise
    return tuple(plan.target_directory for plan in plans)


__all__ = [
    "Sam2LayoutRepairPlan",
    "find_misnamed_sam2_sessions",
    "repair_misnamed_sam2_sessions",
]
