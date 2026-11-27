"""Project-owned physical areas and resolution of stable historical references."""
from pathlib import Path, PurePath


AREA_PATHS = {
    "images": "inputs/images",
    "recordings": "inputs/videos",
    "imported-annotations": "inputs/annotations",
    "datasets": "artifacts/datasets",
    "runs": "artifacts/runs",
    "models": "artifacts/models",
    "predictions": "artifacts/predictions",
    "exports": "outputs/datasets",
    "model-exports": "outputs/models",
    "scratch": "work/scratch",
}


def _relative_parts(value):
    if not isinstance(value, (str, PurePath)):
        raise ValueError("專案儲存路徑無效")
    raw = value.as_posix() if isinstance(value, PurePath) else value.replace("\\", "/")
    parts = raw.split("/")
    if any(not part or part in {".", ".."} or ":" in part or "\0" in part for part in parts):
        raise ValueError("專案儲存路徑必須為安全的相對路徑")
    return parts


def _inside(root, path):
    resolved = path.resolve()
    if resolved != path or not resolved.is_relative_to(root):
        raise ValueError("專案儲存路徑不得為連結或超出專案資料夾")
    return resolved


def project_area(root, area):
    """Return the physical area, using an unmigrated folder when present."""
    root = Path(root).resolve()
    if area not in AREA_PATHS and not area.startswith("integrations/"):
        raise ValueError("專案儲存區名稱無效")
    canonical = root.joinpath(*_relative_parts(AREA_PATHS.get(area, area)))
    if area in AREA_PATHS and not canonical.exists():
        legacy = root / area
        if legacy.exists() or legacy.is_symlink():
            return _inside(root, legacy)
    return _inside(root, canonical)


def logical_reference(value):
    """Normalize a physical relative reference to a stable catalog prefix."""
    parts = _relative_parts(value)
    for alias, canonical in AREA_PATHS.items():
        prefix = canonical.split("/")
        if parts[:len(prefix)] == prefix:
            return "/".join((alias, *parts[len(prefix):]))
    return "/".join(parts)


def project_reference(root, value):
    """Resolve a project-relative reference from either layout."""
    root = Path(root).resolve()
    exact = root.joinpath(*_relative_parts(value))
    if exact.exists() or exact.is_symlink():
        return _inside(root, exact)
    area, *suffix = logical_reference(value).split("/")
    if area not in AREA_PATHS:
        return _inside(root, root.joinpath(area, *suffix))
    selected = _inside(root, project_area(root, area).joinpath(*suffix))
    if not selected.exists():
        legacy = root.joinpath(area, *suffix)
        if legacy.exists() or legacy.is_symlink():
            return _inside(root, legacy)
    return selected


def project_relative(root, value):
    return project_reference(root, value).relative_to(Path(root).resolve())


def migrate_project_layout(root, report):
    """Move whole areas on the same volume and leave conflicting areas readable."""
    root = Path(root).resolve()
    for alias, canonical in AREA_PATHS.items():
        source = root / alias
        if not source.exists() and not source.is_symlink():
            continue
        target = root.joinpath(*canonical.split("/"))
        entry = {"from": str(source), "to": str(target)}
        try:
            if source.is_symlink() or not source.is_dir() or _inside(root, source) != source:
                report["conflicts"].append({**entry, "reason": "invalid_legacy_area"})
                continue
            if _inside(root, target) != target or target.exists() or target.is_symlink():
                report["conflicts"].append({**entry, "reason": "destination_exists"})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        except (OSError, ValueError):
            report["conflicts"].append({**entry, "reason": "path_locked"})
        else:
            report["moved"].append(entry)
