"""Crash-resistant dataset sessions for segmentation error analysis.

The GUI can use :class:`SegmentationDatasetSession` without knowing anything
about file naming or manifest bookkeeping.  Every session owns a unique
directory, uses lossless PNG files, and publishes a record to ``manifest.json``
only after all of that record's assets have been written successfully.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Optional

import cv2
import numpy as np

from .models import SegmentationDiagnostics


MANIFEST_SCHEMA_VERSION = 1
_PNG_PARAMETERS = (cv2.IMWRITE_PNG_COMPRESSION, 3)


@dataclass(frozen=True)
class BackgroundSampleRecord:
    """The stable identifier and absolute path of one empty-scene sample."""

    record_id: str
    image_path: Path


@dataclass(frozen=True)
class SceneRecord:
    """Absolute paths belonging to one captured object scene."""

    record_id: str
    image_path: Path
    prediction_path: Path
    diagnostics_path: Path
    ground_truth_path: Optional[Path] = None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _session_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _validated_bgr(image: np.ndarray, name: str = "image_bgr") -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be a non-empty HxWx3 uint8 BGR image")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{name} must be a non-empty HxWx3 uint8 BGR image")
    # Own a snapshot: GUI capture buffers may be reused or mutated immediately
    # after this method returns to the caller.
    return np.array(image, dtype=np.uint8, order="C", copy=True)


def _validated_mask(
    mask: np.ndarray,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != expected_shape:
        raise ValueError(
            f"{name} must be a uint8 binary mask with shape {expected_shape}"
        )
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError(f"{name} must contain only 0 and 255")
    return np.array(mask, dtype=np.uint8, order="C", copy=True)


def _json_compatible(value: Any) -> Any:
    """Convert supported diagnostic values to strict JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("diagnostics and metadata keys must be strings")
            converted[key] = _json_compatible(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON compatible")


def _strict_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("data must be finite and JSON serializable") from exc
    return (text + "\n").encode("utf-8")


def _encoded_png(image: np.ndarray, name: str) -> bytes:
    success, encoded = cv2.imencode(".png", image, _PNG_PARAMETERS)
    if not success or encoded is None or encoded.size == 0:
        raise OSError(f"OpenCV could not encode {name} as PNG")
    return encoded.tobytes()


def _atomic_write(path: Path, payload: bytes, *, replace: bool = False) -> None:
    """Write one file in place without exposing a partially written target."""

    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite existing dataset file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class SegmentationDatasetSession:
    """Create and append to one fixed-camera segmentation dataset session.

    Constructing the class creates ``session_<UTC timestamp>`` below
    ``root_directory``.  Public save methods are thread-safe within the process;
    a session object should be shared by all GUI callbacks writing that session.
    """

    _SUBDIRECTORIES = (
        "background",
        "images",
        "predictions",
        "ground_truth",
        "diagnostics",
    )

    def __init__(
        self,
        root_directory: str | os.PathLike[str],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        converted_metadata = _json_compatible({} if metadata is None else metadata)
        if not isinstance(converted_metadata, dict):
            raise TypeError("metadata must be a mapping")
        # Validate user-controlled JSON before allocating a session directory.
        _strict_json_bytes(converted_metadata)

        root = Path(root_directory).expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise NotADirectoryError(f"dataset root is not a directory: {root}")
        root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self.root_directory = root
        self.session_directory = self._create_unique_session_directory(root)
        # Concise alias intended for status labels and file-dialog integration.
        self.session_dir = self.session_directory
        self.session_id = self.session_directory.name
        for name in self._SUBDIRECTORIES:
            (self.session_directory / name).mkdir()
        self.manifest_path = self.session_directory / "manifest.json"

        now = _utc_timestamp()
        self._manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at_utc": now,
            "updated_at_utc": now,
            "source_size": None,
            "metadata": converted_metadata,
            "background_samples": [],
            "scenes": [],
        }
        try:
            _atomic_write(self.manifest_path, _strict_json_bytes(self._manifest))
        except BaseException:
            # A constructor failure must not leave a directory that looks like a
            # usable session.  Only paths just created by this object are touched.
            for name in reversed(self._SUBDIRECTORIES):
                (self.session_directory / name).rmdir()
            self.session_directory.rmdir()
            raise

    @classmethod
    def open(
        cls,
        session_directory: str | os.PathLike[str],
    ) -> "SegmentationDatasetSession":
        """Open an existing valid session without creating a new directory."""

        directory = Path(session_directory).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"dataset session is not a directory: {directory}")
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"dataset manifest is not valid UTF-8 JSON: {manifest_path}") from exc

        cls._validate_existing_session(directory, manifest)
        instance = cls.__new__(cls)
        instance._lock = threading.RLock()
        instance.root_directory = directory.parent
        instance.session_directory = directory
        instance.session_dir = directory
        instance.session_id = directory.name
        instance.manifest_path = manifest_path
        instance._manifest = manifest
        return instance

    @classmethod
    def _validate_existing_session(cls, directory: Path, manifest: Any) -> None:
        if not isinstance(manifest, dict):
            raise ValueError("dataset manifest root must be a JSON object")
        required = {
            "schema_version",
            "session_id",
            "created_at_utc",
            "updated_at_utc",
            "source_size",
            "metadata",
            "background_samples",
            "scenes",
        }
        if not required.issubset(manifest):
            missing = ", ".join(sorted(required.difference(manifest)))
            raise ValueError(f"dataset manifest is missing required fields: {missing}")
        if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported dataset manifest schema: {manifest['schema_version']!r}"
            )
        if manifest["session_id"] != directory.name:
            raise ValueError("manifest session_id does not match its directory name")
        if not isinstance(manifest["metadata"], dict):
            raise ValueError("manifest metadata must be a JSON object")
        if not isinstance(manifest["background_samples"], list) or not isinstance(
            manifest["scenes"], list
        ):
            raise ValueError("manifest records must be JSON arrays")

        source_size = manifest["source_size"]
        valid_size = (
            isinstance(source_size, list)
            and len(source_size) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in source_size)
        )
        if source_size is not None and not valid_size:
            raise ValueError("manifest source_size must be null or two positive integers")
        if source_size is None and (manifest["background_samples"] or manifest["scenes"]):
            raise ValueError("a manifest with records must declare source_size")

        for name in cls._SUBDIRECTORIES:
            if not (directory / name).is_dir():
                raise ValueError(f"dataset session is missing the {name} directory")

        seen_ids: set[str] = set()
        for entry in manifest["background_samples"]:
            cls._validate_record_entry(entry, "background", seen_ids)
            cls._require_session_file(directory, entry["image"], "background")
        for entry in manifest["scenes"]:
            cls._validate_record_entry(entry, "scene", seen_ids)
            cls._require_session_file(directory, entry["image"], "images")
            cls._require_session_file(directory, entry["prediction"], "predictions")
            cls._require_session_file(directory, entry["diagnostics"], "diagnostics")
            ground_truth = entry["ground_truth"]
            if ground_truth is not None:
                cls._require_session_file(directory, ground_truth, "ground_truth")

    @staticmethod
    def _validate_record_entry(
        entry: Any,
        kind: str,
        seen_ids: set[str],
    ) -> None:
        required = (
            {"id", "created_at_utc", "image"}
            if kind == "background"
            else {
                "id",
                "created_at_utc",
                "image",
                "prediction",
                "diagnostics",
                "ground_truth",
            }
        )
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError(f"manifest contains an invalid {kind} record")
        record_id = entry["id"]
        expected_prefix = "background_" if kind == "background" else "scene_"
        if (
            not isinstance(record_id, str)
            or not record_id.startswith(expected_prefix)
            or not record_id[len(expected_prefix) :].isdigit()
            or record_id in seen_ids
        ):
            raise ValueError(f"manifest contains an invalid or duplicate {kind} id")
        seen_ids.add(record_id)

    @staticmethod
    def _require_session_file(directory: Path, relative_value: Any, folder: str) -> Path:
        if not isinstance(relative_value, str) or not relative_value:
            raise ValueError("manifest file paths must be non-empty strings")
        relative_path = Path(relative_value)
        if relative_path.is_absolute() or not relative_path.parts or relative_path.parts[0] != folder:
            raise ValueError(f"manifest path must remain inside {folder}/")
        resolved = (directory / relative_path).resolve()
        try:
            resolved.relative_to(directory)
        except ValueError as exc:
            raise ValueError("manifest path escapes the session directory") from exc
        if not resolved.is_file():
            raise ValueError(f"manifest references a missing dataset file: {relative_value}")
        return resolved

    @staticmethod
    def _create_unique_session_directory(root: Path) -> Path:
        stem = f"session_{_session_timestamp()}"
        for suffix in range(10_000):
            name = stem if suffix == 0 else f"{stem}_{suffix:02d}"
            candidate = root / name
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            return candidate
        raise FileExistsError("could not allocate a unique dataset session directory")

    @property
    def source_size(self) -> Optional[tuple[int, int]]:
        """The locked ``(width, height)`` or ``None`` before the first save."""

        with self._lock:
            size = self._manifest["source_size"]
            return None if size is None else (int(size[0]), int(size[1]))

    def read_manifest(self) -> dict[str, Any]:
        """Return a defensive snapshot of the in-memory manifest."""

        with self._lock:
            return deepcopy(self._manifest)

    def save_background(self, image_bgr: np.ndarray) -> BackgroundSampleRecord:
        """Save one empty-background BGR frame and return its stable record."""

        image = _validated_bgr(image_bgr, "image_bgr")
        payload = _encoded_png(image, "background sample")
        height, width = image.shape[:2]

        with self._lock:
            candidate = deepcopy(self._manifest)
            self._lock_or_check_source_size(candidate, width, height)
            index = len(candidate["background_samples"]) + 1
            record_id = f"background_{index:06d}"
            relative_path = Path("background") / f"{record_id}.png"
            absolute_path = self.session_directory / relative_path
            entry = {
                "id": record_id,
                "created_at_utc": _utc_timestamp(),
                "image": relative_path.as_posix(),
            }
            candidate["background_samples"].append(entry)
            candidate["updated_at_utc"] = entry["created_at_utc"]
            self._commit_record(candidate, ((absolute_path, payload),))
            return BackgroundSampleRecord(record_id, absolute_path)

    def save_background_sample(self, image_bgr: np.ndarray) -> BackgroundSampleRecord:
        """Backward-compatible descriptive alias for :meth:`save_background`."""

        return self.save_background(image_bgr)

    def save_scene(
        self,
        image_bgr: np.ndarray,
        prediction_mask: np.ndarray,
        diagnostics: SegmentationDiagnostics | Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        ground_truth_mask: Optional[np.ndarray] = None,
    ) -> SceneRecord:
        """Save one source image, prediction, diagnostics, and optional GT mask."""

        image = _validated_bgr(image_bgr, "image_bgr")
        height, width = image.shape[:2]
        prediction = _validated_mask(
            prediction_mask,
            (height, width),
            "prediction_mask",
        )
        ground_truth = None
        if ground_truth_mask is not None:
            ground_truth = _validated_mask(
                ground_truth_mask,
                (height, width),
                "ground_truth_mask",
            )
        converted_diagnostics = _json_compatible(diagnostics)
        if not isinstance(converted_diagnostics, dict):
            raise TypeError("diagnostics must be a mapping or dataclass")
        converted_metadata = _json_compatible({} if metadata is None else metadata)
        if not isinstance(converted_metadata, dict):
            raise TypeError("metadata must be a mapping")

        image_payload = _encoded_png(image, "scene image")
        prediction_payload = _encoded_png(prediction, "prediction mask")
        ground_truth_payload = (
            None
            if ground_truth is None
            else _encoded_png(ground_truth, "ground-truth mask")
        )

        with self._lock:
            candidate = deepcopy(self._manifest)
            self._lock_or_check_source_size(candidate, width, height)
            index = len(candidate["scenes"]) + 1
            record_id = f"scene_{index:06d}"
            created_at = _utc_timestamp()
            image_relative = Path("images") / f"{record_id}.png"
            prediction_relative = Path("predictions") / f"{record_id}.png"
            diagnostics_relative = Path("diagnostics") / f"{record_id}.json"
            ground_truth_relative = (
                None
                if ground_truth is None
                else Path("ground_truth") / f"{record_id}.png"
            )
            diagnostics_document = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "session_id": self.session_id,
                "record_id": record_id,
                "created_at_utc": created_at,
                "source_size": [width, height],
                "diagnostics": converted_diagnostics,
                "metadata": converted_metadata,
            }
            diagnostics_payload = _strict_json_bytes(diagnostics_document)

            entry = {
                "id": record_id,
                "created_at_utc": created_at,
                "image": image_relative.as_posix(),
                "prediction": prediction_relative.as_posix(),
                "diagnostics": diagnostics_relative.as_posix(),
                "ground_truth": (
                    None
                    if ground_truth_relative is None
                    else ground_truth_relative.as_posix()
                ),
            }
            candidate["scenes"].append(entry)
            candidate["updated_at_utc"] = created_at

            files = [
                (self.session_directory / image_relative, image_payload),
                (self.session_directory / prediction_relative, prediction_payload),
                (self.session_directory / diagnostics_relative, diagnostics_payload),
            ]
            if ground_truth_relative is not None and ground_truth_payload is not None:
                files.append(
                    (self.session_directory / ground_truth_relative, ground_truth_payload)
                )
            self._commit_record(candidate, tuple(files))
            return SceneRecord(
                record_id=record_id,
                image_path=self.session_directory / image_relative,
                prediction_path=self.session_directory / prediction_relative,
                diagnostics_path=self.session_directory / diagnostics_relative,
                ground_truth_path=(
                    None
                    if ground_truth_relative is None
                    else self.session_directory / ground_truth_relative
                ),
            )

    def save_ground_truth(
        self,
        record_id: str,
        ground_truth_mask: np.ndarray,
    ) -> SceneRecord:
        """Attach a mask to an existing scene that does not yet have one."""

        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        with self._lock:
            candidate = deepcopy(self._manifest)
            matches = [entry for entry in candidate["scenes"] if entry["id"] == record_id]
            if not matches:
                raise KeyError(f"unknown scene record: {record_id}")
            entry = matches[0]
            if entry["ground_truth"] is not None:
                raise FileExistsError(f"ground truth already exists for {record_id}")
            source_size = candidate["source_size"]
            assert source_size is not None
            width, height = int(source_size[0]), int(source_size[1])
            mask = _validated_mask(
                ground_truth_mask,
                (height, width),
                "ground_truth_mask",
            )
            payload = _encoded_png(mask, "ground-truth mask")
            relative_path = Path("ground_truth") / f"{record_id}.png"
            entry["ground_truth"] = relative_path.as_posix()
            candidate["updated_at_utc"] = _utc_timestamp()
            self._commit_record(
                candidate,
                ((self.session_directory / relative_path, payload),),
            )
            return self._scene_record_from_entry(entry)

    def register_existing_ground_truth(self, record_id: str) -> SceneRecord:
        """Validate and register ``ground_truth/<record_id>.png``.

        This is the synchronization point for an external annotation tool that
        writes the PNG itself.  The manifest is updated only after OpenCV can
        decode a same-resolution, binary uint8 mask from the expected path.
        """

        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        with self._lock:
            candidate = deepcopy(self._manifest)
            matches = [entry for entry in candidate["scenes"] if entry["id"] == record_id]
            if not matches:
                raise KeyError(f"unknown scene record: {record_id}")
            entry = matches[0]
            relative_path = Path("ground_truth") / f"{record_id}.png"
            registered_path = entry["ground_truth"]
            if registered_path not in (None, relative_path.as_posix()):
                raise ValueError(f"ground truth for {record_id} has an unexpected path")
            absolute_path = self.session_directory / relative_path
            decoded = cv2.imread(str(absolute_path), cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise ValueError(f"ground-truth PNG is missing or unreadable: {absolute_path}")
            source_size = candidate["source_size"]
            assert source_size is not None
            width, height = int(source_size[0]), int(source_size[1])
            _validated_mask(decoded, (height, width), "ground_truth_mask")
            entry["ground_truth"] = relative_path.as_posix()
            candidate["updated_at_utc"] = _utc_timestamp()
            self._commit_record(candidate, ())
            return self._scene_record_from_entry(entry)

    def _lock_or_check_source_size(
        self,
        candidate: dict[str, Any],
        width: int,
        height: int,
    ) -> None:
        expected = candidate["source_size"]
        actual = [width, height]
        if expected is None:
            candidate["source_size"] = actual
        elif expected != actual:
            raise ValueError(
                f"image size {(width, height)} does not match session source size "
                f"{tuple(expected)}"
            )

    def _commit_record(
        self,
        candidate_manifest: dict[str, Any],
        files: tuple[tuple[Path, bytes], ...],
    ) -> None:
        committed: list[Path] = []
        try:
            for path, payload in files:
                _atomic_write(path, payload)
                committed.append(path)
            _atomic_write(
                self.manifest_path,
                _strict_json_bytes(candidate_manifest),
                replace=True,
            )
        except BaseException:
            for path in reversed(committed):
                path.unlink(missing_ok=True)
            raise
        self._manifest = candidate_manifest

    def _scene_record_from_entry(self, entry: Mapping[str, Any]) -> SceneRecord:
        ground_truth = entry["ground_truth"]
        return SceneRecord(
            record_id=entry["id"],
            image_path=self.session_directory / entry["image"],
            prediction_path=self.session_directory / entry["prediction"],
            diagnostics_path=self.session_directory / entry["diagnostics"],
            ground_truth_path=(
                None if ground_truth is None else self.session_directory / ground_truth
            ),
        )


__all__ = [
    "BackgroundSampleRecord",
    "MANIFEST_SCHEMA_VERSION",
    "SceneRecord",
    "SegmentationDatasetSession",
]
