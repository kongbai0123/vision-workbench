"""Crash-resistant SAM2 pseudo-label datasets for later review and training.

Each sample publishes a lossless source image, binary mask, explicit contour
geometry, QA overlay, manifest entry, and a COCO annotation.  The COCO mask is
stored as uncompressed column-major RLE so it can be decoded exactly without
``pycocotools``.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from classical_segmentation import (
    NormalizedPoint,
    NormalizedRect,
    SegmentationPrompts,
    extract_contour_hierarchy,
)

from .models import Sam2SegmentationResult
from .postprocess import encode_sam2_prompts


TRAINING_MANIFEST_SCHEMA_VERSION = 1
COCO_SCHEMA_VERSION = "1.0"
LABEL_SOURCE = "sam2_pseudo_label"
REVIEW_STATUS = "pending"
HUMAN_VERIFIED_REVIEW_STATUS = "human_verified"
REJECTED_REVIEW_STATUS = "rejected"
MIXED_REVIEW_STATUS = "mixed"
SAMPLE_REVIEW_STATUSES = frozenset(
    {REVIEW_STATUS, HUMAN_VERIFIED_REVIEW_STATUS, REJECTED_REVIEW_STATUS}
)
SESSION_REVIEW_STATUSES = frozenset((*SAMPLE_REVIEW_STATUSES, MIXED_REVIEW_STATUS))
_PNG_PARAMETERS = (cv2.IMWRITE_PNG_COMPRESSION, 3)
_SUBDIRECTORIES = ("images", "masks", "contours", "overlays", "annotations")
_SESSION_LOCKS_GUARD = threading.Lock()
_SESSION_LOCKS: dict[str, threading.RLock] = {}
_SESSION_STATES: dict[str, "_SharedSessionState"] = {}


def _session_key(directory: Path) -> str:
    return os.path.normcase(str(directory.resolve()))


def _shared_session_lock(directory: Path) -> threading.RLock:
    """Share one append lock between separately opened in-process handles."""

    key = _session_key(directory)
    with _SESSION_LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _SESSION_LOCKS[key] = lock
        return lock


@dataclass(frozen=True)
class Sam2TrainingSampleRecord:
    """Stable identifiers and absolute paths for one exported training sample."""

    sample_id: str
    image_id: int
    annotation_id: int
    image_path: Path
    mask_path: Path
    contours_path: Path
    overlay_path: Path

    @property
    def record_id(self) -> str:
        """Alias matching the existing classical dataset record terminology."""

        return self.sample_id


@dataclass
class _SharedSessionState:
    """Mutable documents shared by all handles for one in-process writer."""

    manifest: dict[str, Any]
    instances: dict[str, Any]


def _shared_session_state(
    directory: Path,
    manifest: dict[str, Any],
    instances: dict[str, Any],
) -> _SharedSessionState:
    key = _session_key(directory)
    with _SESSION_LOCKS_GUARD:
        state = _SESSION_STATES.get(key)
        if state is None:
            state = _SharedSessionState(manifest=manifest, instances=instances)
            _SESSION_STATES[key] = state
        else:
            state.manifest = manifest
            state.instances = instances
        return state


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _session_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _validated_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validated_bgr(image_bgr: np.ndarray) -> np.ndarray:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if (
        image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.shape[0] <= 0
        or image_bgr.shape[1] <= 0
    ):
        raise ValueError("image_bgr must be a non-empty HxWx3 uint8 BGR image")
    return np.array(image_bgr, dtype=np.uint8, order="C", copy=True)


def _validated_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError("result.mask must be a numpy array")
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != expected_shape:
        raise ValueError(
            f"result.mask must be a uint8 binary mask with shape {expected_shape}"
        )
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError("result.mask must contain only 0 and 255")
    if not np.any(mask):
        raise ValueError("result.mask must contain foreground pixels")
    return np.array(mask, dtype=np.uint8, order="C", copy=True)


def _validated_review_mask(
    mask: np.ndarray,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError("corrected_mask must be a numpy array")
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != expected_shape:
        raise ValueError(
            "corrected_mask must be a uint8 binary mask with shape "
            f"{expected_shape}"
        )
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError("corrected_mask must contain only 0 and 255")
    if not np.any(mask):
        raise ValueError("corrected_mask must contain foreground pixels")
    return np.array(mask, dtype=np.uint8, order="C", copy=True)


def _json_compatible(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("metadata keys must be strings")
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


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _decode_png(payload: bytes, flags: int, description: str) -> np.ndarray:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{description} is not a PNG file")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
    if decoded is None or decoded.size == 0:
        raise ValueError(f"{description} is not a readable PNG file")
    return decoded


def _validate_saved_prompts(document: Any, width: int, height: int) -> tuple[int, int]:
    if not isinstance(document, dict):
        raise ValueError("saved prompts must be an object")

    def validate_points(value: Any, name: str) -> int:
        if not isinstance(value, list):
            raise ValueError(f"saved {name} must be an array")
        for point in value:
            if not isinstance(point, dict):
                raise ValueError(f"saved {name} contains an invalid point")
            normalized = point.get("normalized")
            pixel = point.get("pixel")
            if not isinstance(normalized, dict) or not isinstance(pixel, dict):
                raise ValueError(f"saved {name} point is incomplete")
            u, v = normalized.get("u"), normalized.get("v")
            if (
                isinstance(u, bool)
                or isinstance(v, bool)
                or not isinstance(u, (int, float))
                or not isinstance(v, (int, float))
                or not math.isfinite(float(u))
                or not math.isfinite(float(v))
                or not 0.0 <= float(u) <= 1.0
                or not 0.0 <= float(v) <= 1.0
            ):
                raise ValueError(f"saved {name} normalized point is invalid")
            x, y = pixel.get("x"), pixel.get("y")
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, int)
                or not isinstance(y, int)
                or not 0 <= x < width
                or not 0 <= y < height
            ):
                raise ValueError(f"saved {name} pixel point is invalid")
            expected = (
                min(width - 1, max(0, int(round(float(u) * (width - 1))))),
                min(height - 1, max(0, int(round(float(v) * (height - 1))))),
            )
            if (x, y) != expected:
                raise ValueError(f"saved {name} pixel and normalized point disagree")
        return len(value)

    foreground_count = validate_points(
        document.get("foreground_points"), "foreground points"
    )
    background_count = validate_points(
        document.get("background_points"), "background points"
    )
    roi = document.get("roi")
    if roi is not None:
        if not isinstance(roi, dict) or not isinstance(roi.get("normalized"), dict):
            raise ValueError("saved ROI is incomplete")
        normalized = roi["normalized"]
        try:
            rectangle = NormalizedRect(
                normalized["left"],
                normalized["top"],
                normalized["right"],
                normalized["bottom"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("saved normalized ROI is invalid") from exc
        pixels = roi.get("pixel_xyxy_exclusive")
        if pixels != list(rectangle.to_pixels(width, height)):
            raise ValueError("saved ROI pixel and normalized coordinates disagree")
    return foreground_count, background_count


def _encoded_png(image: np.ndarray, name: str) -> bytes:
    success, encoded = cv2.imencode(".png", image, _PNG_PARAMETERS)
    if not success or encoded is None or encoded.size == 0:
        raise OSError(f"OpenCV could not encode {name} as PNG")
    return encoded.tobytes()


def _atomic_write(path: Path, payload: bytes, *, replace: bool = False) -> None:
    """Publish one complete file without exposing a partially written target."""

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


def encode_coco_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask using COCO's column-major uncompressed RLE."""

    array = np.asarray(mask)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("mask must be a non-empty two-dimensional array")
    if not (
        array.dtype == np.bool_
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError("mask must contain boolean or numeric values")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("mask must contain finite values")
    if not np.all((array == 0) | (array == 1) | (array == 255)):
        raise ValueError("mask must contain only binary values (0, 1, or 255)")
    binary = np.asarray(array != 0, dtype=np.uint8)
    from composer_core.geometry import encode_rle
    counts = encode_rle(binary)
    return {
        "size": [int(binary.shape[0]), int(binary.shape[1])],
        "counts": counts,
    }


def decode_coco_uncompressed_rle(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode the uncompressed RLE format emitted by this module."""

    if not isinstance(rle, Mapping):
        raise TypeError("rle must be a mapping")
    size = rle.get("size")
    counts = rle.get("counts")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in size
        )
    ):
        raise ValueError("rle size must contain positive integer height and width")
    if not isinstance(counts, list) or not counts:
        raise ValueError("rle counts must be a non-empty integer list")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise ValueError("rle counts must contain non-negative integers")
    height, width = int(size[0]), int(size[1])
    if sum(counts) != height * width:
        raise ValueError("rle counts do not cover the declared mask size")
    from composer_core.geometry import decode_rle
    return decode_rle(counts, width, height)


def _mask_area_and_bbox(mask: np.ndarray) -> tuple[int, list[int]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("mask must contain foreground pixels")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return int(xs.size), [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _contour_points(contour: np.ndarray) -> list[list[int]]:
    points = np.asarray(contour).reshape(-1, 2)
    return [[int(x), int(y)] for x, y in points]


def _normalized_contour_points(
    pixel_points: Sequence[Sequence[int]],
    width: int,
    height: int,
) -> list[list[float]]:
    x_denominator = max(1, width - 1)
    y_denominator = max(1, height - 1)
    return [
        [float(point[0]) / x_denominator, float(point[1]) / y_denominator]
        for point in pixel_points
    ]


def _prompt_document(
    prompts: SegmentationPrompts,
    width: int,
    height: int,
) -> dict[str, Any]:
    encoding = encode_sam2_prompts(prompts, (width, height))

    def point_records(normalized_points, pixel_points):
        return [
            {
                "normalized": {"u": float(point.u), "v": float(point.v)},
                "pixel": {"x": int(pixel[0]), "y": int(pixel[1])},
            }
            for point, pixel in zip(normalized_points, pixel_points)
        ]

    roi = prompts.roi
    roi_document = None
    if roi is not None:
        assert isinstance(roi, NormalizedRect)
        x0, y0, x1, y1 = roi.to_pixels(width, height)
        roi_document = {
            "normalized": {
                "left": roi.left,
                "top": roi.top,
                "right": roi.right,
                "bottom": roi.bottom,
            },
            "pixel_xyxy_exclusive": [x0, y0, x1, y1],
        }
    return {
        "foreground_points": point_records(
            prompts.foreground_points,
            encoding.foreground_pixels,
        ),
        "background_points": point_records(
            prompts.background_points,
            encoding.background_pixels,
        ),
        "roi": roi_document,
    }


def _prompts_from_document(document: Mapping[str, Any]) -> SegmentationPrompts:
    """Rebuild prompts after the persisted representation has been validated."""

    def points(name: str) -> tuple[NormalizedPoint, ...]:
        return tuple(
            NormalizedPoint(
                float(entry["normalized"]["u"]),
                float(entry["normalized"]["v"]),
            )
            for entry in document[name]
        )

    roi_document = document.get("roi")
    roi = None
    if roi_document is not None:
        normalized = roi_document["normalized"]
        roi = NormalizedRect(
            float(normalized["left"]),
            float(normalized["top"]),
            float(normalized["right"]),
            float(normalized["bottom"]),
        )
    return SegmentationPrompts(
        foreground_points=points("foreground_points"),
        background_points=points("background_points"),
        roi=roi,
    )


def _aggregate_review_status(samples: Sequence[Mapping[str, Any]]) -> str:
    statuses = {sample.get("review_status") for sample in samples}
    if not statuses:
        return REVIEW_STATUS
    if len(statuses) == 1:
        status = next(iter(statuses))
        if status in SAMPLE_REVIEW_STATUSES:
            return str(status)
    return MIXED_REVIEW_STATUS


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contours_document(
    mask: np.ndarray,
    width: int,
    height: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, ...], int]:
    hierarchy = extract_contour_hierarchy(mask)
    if hierarchy.outer_index is None:
        raise ValueError("result mask has no outer contour")
    component_count, _labels = cv2.connectedComponents(
        np.asarray(mask != 0, dtype=np.uint8),
        connectivity=8,
    )
    if component_count != 2:
        raise ValueError("result mask must contain exactly one main connected object")
    top_level = [
        index
        for index in range(len(hierarchy.contours))
        if int(hierarchy.hierarchy[0, index, 3]) == -1
    ]
    if len(top_level) != 1:
        raise ValueError("result mask must contain exactly one main connected object")
    outer_index = hierarchy.outer_index

    def record(index: int) -> dict[str, Any]:
        pixels = _contour_points(hierarchy.contours[index])
        return {
            "hierarchy_index": int(index),
            "closed": True,
            "pixels": pixels,
            "normalized": _normalized_contour_points(pixels, width, height),
        }

    outer = record(outer_index)
    holes = [record(index) for index in hierarchy.hole_indices]
    return {
        "coordinate_convention": {
            "pixel": "inclusive_xy",
            "normalized": "x/(width-1), y/(height-1)",
        },
        "outer": outer,
        "holes": holes,
    }, hierarchy.contours, outer_index


def _qa_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    prompts: SegmentationPrompts,
) -> np.ndarray:
    height, width = mask.shape
    hierarchy = extract_contour_hierarchy(mask)
    overlay = image.copy()
    foreground = mask != 0
    tint = np.zeros_like(overlay)
    tint[:, :] = (0, 180, 0)
    overlay[foreground] = cv2.addWeighted(
        image[foreground],
        0.60,
        tint[foreground],
        0.40,
        0.0,
    )
    if hierarchy.outer_index is not None:
        cv2.drawContours(
            overlay,
            [hierarchy.contours[hierarchy.outer_index]],
            -1,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
    for index in hierarchy.hole_indices:
        cv2.drawContours(
            overlay,
            [hierarchy.contours[index]],
            -1,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    encoding = encode_sam2_prompts(prompts, (width, height))
    marker_size = max(10, int(round(min(width, height) * 0.025)))
    for point in encoding.foreground_pixels:
        cv2.drawMarker(
            overlay,
            point,
            (0, 255, 0),
            cv2.MARKER_CROSS,
            marker_size,
            2,
            cv2.LINE_AA,
        )
    for point in encoding.background_pixels:
        cv2.drawMarker(
            overlay,
            point,
            (0, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            marker_size,
            2,
            cv2.LINE_AA,
        )
    return overlay


class Sam2TrainingDatasetSession:
    """Create or append to one reviewable SAM2 pseudo-label dataset session.

    Appends are thread-safe across handles opened in the same Python process.
    Only one process may write a session at a time; external multi-process
    locking is deliberately left to the caller.  Mask PNG files are the
    canonical training labels.  COCO uncompressed RLE is an exact interchange
    representation and may need conversion for loaders that require compressed
    ``pycocotools`` RLE.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise TypeError("use Sam2TrainingDatasetSession.create() or .open()")

    @classmethod
    def create(
        cls,
        root_directory: str | os.PathLike[str],
        model_id: str,
        category_name: str = "workpiece",
        session_metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Sam2TrainingDatasetSession":
        model_id = _validated_nonempty_string(model_id, "model_id")
        category_name = _validated_nonempty_string(category_name, "category_name")
        converted_session_metadata = _json_compatible(
            {} if session_metadata is None else session_metadata
        )
        if not isinstance(converted_session_metadata, dict):
            raise TypeError("session_metadata must be a mapping")
        _strict_json_bytes(converted_session_metadata)
        root = Path(root_directory).expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise NotADirectoryError(f"dataset root is not a directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        session_directory = cls._create_unique_session_directory(root)
        created_paths: list[Path] = []
        try:
            for name in _SUBDIRECTORIES:
                directory = session_directory / name
                directory.mkdir()
                created_paths.append(directory)
            now = _utc_timestamp()
            manifest = {
                "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
                "dataset_type": "sam2_training_pseudo_labels",
                "session_id": session_directory.name,
                "created_at_utc": now,
                "updated_at_utc": now,
                "model_id": model_id,
                "category": {"id": 1, "name": category_name},
                "metadata": converted_session_metadata,
                "label_source": LABEL_SOURCE,
                "review_status": REVIEW_STATUS,
                "canonical_label_format": "binary_mask_png",
                "sam2_loader_compatibility": "adapter_required",
                "source_size": None,
                "samples": [],
            }
            instances = {
                "info": {
                    "version": COCO_SCHEMA_VERSION,
                    "description": "SAM2 pseudo-labels pending human review",
                    "created_at_utc": now,
                    "session_id": session_directory.name,
                    "model_id": model_id,
                    "label_source": LABEL_SOURCE,
                    "review_status": REVIEW_STATUS,
                    "canonical_label_format": "binary_mask_png",
                    "segmentation_format": "coco_uncompressed_rle_column_major",
                },
                "licenses": [],
                "categories": [
                    {"id": 1, "name": category_name, "supercategory": "object"}
                ],
                "images": [],
                "annotations": [],
            }
            instances_path = session_directory / "annotations" / "instances.json"
            manifest_path = session_directory / "manifest.json"
            _atomic_write(instances_path, _strict_json_bytes(instances))
            _atomic_write(manifest_path, _strict_json_bytes(manifest))
        except BaseException:
            (session_directory / "manifest.json").unlink(missing_ok=True)
            (session_directory / "annotations" / "instances.json").unlink(missing_ok=True)
            for path in reversed(created_paths):
                path.rmdir()
            session_directory.rmdir()
            raise
        return cls._from_documents(session_directory, manifest, instances)

    @classmethod
    def open(
        cls,
        session_directory: str | os.PathLike[str],
    ) -> "Sam2TrainingDatasetSession":
        directory = Path(session_directory).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"training dataset session is not a directory: {directory}")
        manifest_path = directory / "manifest.json"
        instances_path = directory / "annotations" / "instances.json"
        with _shared_session_lock(directory):
            manifest = cls._read_json(manifest_path, "training manifest")
            instances = cls._read_json(instances_path, "COCO instances")
            cls._validate_existing_session(directory, manifest, instances)
            return cls._from_documents(directory, manifest, instances)

    @classmethod
    def _from_documents(
        cls,
        directory: Path,
        manifest: dict[str, Any],
        instances: dict[str, Any],
    ) -> "Sam2TrainingDatasetSession":
        instance = cls.__new__(cls)
        instance._lock = _shared_session_lock(directory)
        with instance._lock:
            instance._state = _shared_session_state(directory, manifest, instances)
        instance.root_directory = directory.parent
        instance.session_directory = directory
        instance.session_dir = directory
        instance.session_id = directory.name
        instance.manifest_path = directory / "manifest.json"
        instance.instances_path = directory / "annotations" / "instances.json"
        return instance

    @property
    def _manifest(self) -> dict[str, Any]:
        return self._state.manifest

    @_manifest.setter
    def _manifest(self, value: dict[str, Any]) -> None:
        self._state.manifest = value

    @property
    def _instances(self) -> dict[str, Any]:
        return self._state.instances

    @_instances.setter
    def _instances(self, value: dict[str, Any]) -> None:
        self._state.instances = value

    @staticmethod
    def _read_json(path: Path, description: str) -> dict[str, Any]:
        try:
            value = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"{description} does not exist: {path}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{description} root must be a JSON object")
        return value

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
        raise FileExistsError("could not allocate a unique training dataset session")

    @classmethod
    def _validate_existing_session(
        cls,
        directory: Path,
        manifest: dict[str, Any],
        instances: dict[str, Any],
    ) -> None:
        required_manifest = {
            "schema_version",
            "dataset_type",
            "session_id",
            "created_at_utc",
            "updated_at_utc",
            "model_id",
            "category",
            "label_source",
            "review_status",
            "canonical_label_format",
            "sam2_loader_compatibility",
            "source_size",
            "samples",
        }
        if not required_manifest.issubset(manifest):
            missing = ", ".join(sorted(required_manifest.difference(manifest)))
            raise ValueError(f"training manifest is missing required fields: {missing}")
        if manifest["schema_version"] != TRAINING_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported training manifest schema")
        if manifest["dataset_type"] != "sam2_training_pseudo_labels":
            raise ValueError("manifest dataset_type is not a SAM2 training dataset")
        if manifest["session_id"] != directory.name:
            raise ValueError("manifest session_id does not match its directory")
        _validated_nonempty_string(manifest["model_id"], "manifest model_id")
        if manifest["label_source"] != LABEL_SOURCE:
            raise ValueError("manifest label_source is not sam2_pseudo_label")
        if manifest["review_status"] not in SESSION_REVIEW_STATUSES:
            raise ValueError("manifest review_status is invalid")
        if "metadata" in manifest and not isinstance(manifest["metadata"], dict):
            raise ValueError("manifest metadata must be an object")
        if (
            manifest["canonical_label_format"] != "binary_mask_png"
            or manifest["sam2_loader_compatibility"] != "adapter_required"
        ):
            raise ValueError("manifest mask format or SAM2 loader compatibility is invalid")
        category = manifest["category"]
        if (
            not isinstance(category, dict)
            or category.get("id") != 1
            or not isinstance(category.get("name"), str)
            or not category["name"].strip()
        ):
            raise ValueError("manifest category must contain id 1 and a name")
        source_size = manifest["source_size"]
        valid_size = (
            isinstance(source_size, list)
            and len(source_size) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in source_size
            )
        )
        if source_size is not None and not valid_size:
            raise ValueError("manifest source_size must be null or [width, height]")
        samples = manifest["samples"]
        if not isinstance(samples, list):
            raise ValueError("manifest samples must be a JSON array")
        if samples and source_size is None:
            raise ValueError("a manifest with samples must declare source_size")

        for name in _SUBDIRECTORIES:
            if not (directory / name).is_dir():
                raise ValueError(f"training session is missing the {name} directory")

        info = instances.get("info")
        images = instances.get("images")
        annotations = instances.get("annotations")
        categories = instances.get("categories")
        if (
            not isinstance(info, dict)
            or info.get("version") != COCO_SCHEMA_VERSION
            or info.get("session_id") != manifest["session_id"]
            or info.get("model_id") != manifest["model_id"]
            or info.get("label_source") != LABEL_SOURCE
            or info.get("review_status") != manifest["review_status"]
            or info.get("canonical_label_format") != "binary_mask_png"
            or info.get("segmentation_format")
            != "coco_uncompressed_rle_column_major"
        ):
            raise ValueError("COCO info does not match the training manifest")
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise ValueError("COCO images and annotations must be arrays")
        if categories != [
            {"id": 1, "name": category["name"], "supercategory": "object"}
        ]:
            raise ValueError("COCO category does not match the training manifest")
        if len(images) != len(samples) or len(annotations) != len(samples):
            raise ValueError("manifest and COCO sample counts do not match")
        if manifest["review_status"] != _aggregate_review_status(samples):
            raise ValueError("manifest review_status does not summarize its samples")

        def indexed(entries: list[Any], description: str) -> dict[int, dict[str, Any]]:
            records: dict[int, dict[str, Any]] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError(f"COCO {description} contains a non-object record")
                identifier = entry.get("id")
                if (
                    isinstance(identifier, bool)
                    or not isinstance(identifier, int)
                    or identifier <= 0
                    or identifier in records
                ):
                    raise ValueError(
                        f"COCO {description} contains an invalid or duplicate id"
                    )
                records[identifier] = entry
            return records

        coco_images = indexed(images, "images")
        coco_annotations = indexed(annotations, "annotations")

        seen_ids: set[str] = set()
        seen_image_ids: set[int] = set()
        seen_annotation_ids: set[int] = set()
        for sample in samples:
            paths = cls._validate_sample_entry(
                directory,
                sample,
                source_size,
                manifest["model_id"],
                seen_ids,
            )
            if sample["image_id"] in seen_image_ids:
                raise ValueError("manifest contains a duplicate image_id")
            if sample["annotation_id"] in seen_annotation_ids:
                raise ValueError("manifest contains a duplicate annotation_id")
            seen_image_ids.add(sample["image_id"])
            seen_annotation_ids.add(sample["annotation_id"])
            image_entry = coco_images.get(sample["image_id"])
            annotation_entry = coco_annotations.get(sample["annotation_id"])
            if image_entry is None or annotation_entry is None:
                raise ValueError("manifest references a missing COCO record")
            if (
                image_entry.get("file_name") != sample["image"]
                or image_entry.get("width") != sample["width"]
                or image_entry.get("height") != sample["height"]
                or image_entry.get("sam2_sample_id") != sample["id"]
                or image_entry.get("image_sha256") != sample["image_sha256"]
                or image_entry.get("metadata") != sample["metadata"]
                or annotation_entry.get("image_id") != sample["image_id"]
                or annotation_entry.get("category_id") != 1
                or annotation_entry.get("area") != sample["area"]
                or annotation_entry.get("bbox") != sample["bbox"]
                or annotation_entry.get("label_source") != LABEL_SOURCE
                or annotation_entry.get("review_status") != sample["review_status"]
                or annotation_entry.get("model_id") != manifest["model_id"]
                or annotation_entry.get("mask_file") != sample["mask"]
                or annotation_entry.get("contours_file") != sample["contours"]
                or annotation_entry.get("qa_overlay_file") != sample["qa_overlay"]
                or annotation_entry.get("mask_sha256") != sample["mask_sha256"]
                or annotation_entry.get("diagnostics") != sample["diagnostics"]
                or annotation_entry.get("prompts") != sample["prompts"]
                or annotation_entry.get("iscrowd") != 0
            ):
                raise ValueError("manifest and COCO records disagree")
            for field in (
                "review_revision",
                "reviewed_at_utc",
                "reviewer_metadata",
                "review_history",
                "original_pseudo_label",
            ):
                if annotation_entry.get(field) != sample.get(field):
                    raise ValueError("manifest and COCO review provenance disagree")

            image_payload = paths["image"].read_bytes()
            mask_payload = paths["mask"].read_bytes()
            overlay_payload = paths["qa_overlay"].read_bytes()
            if hashlib.sha256(image_payload).hexdigest() != sample["image_sha256"]:
                raise ValueError("training image SHA-256 does not match the manifest")
            if hashlib.sha256(mask_payload).hexdigest() != sample["mask_sha256"]:
                raise ValueError("training mask SHA-256 does not match the manifest")

            image = _decode_png(image_payload, cv2.IMREAD_COLOR, "training image")
            mask = _decode_png(mask_payload, cv2.IMREAD_GRAYSCALE, "training mask")
            overlay = _decode_png(overlay_payload, cv2.IMREAD_COLOR, "QA overlay")
            expected_shape = (sample["height"], sample["width"])
            if image.shape[:2] != expected_shape:
                raise ValueError("training image dimensions do not match the manifest")
            if mask.shape != expected_shape or np.any((mask != 0) & (mask != 255)):
                raise ValueError("training mask is not the declared binary image")
            if overlay.shape[:2] != expected_shape:
                raise ValueError("QA overlay dimensions do not match the manifest")
            area, bbox = _mask_area_and_bbox(mask)
            if area != sample["area"] or bbox != sample["bbox"]:
                raise ValueError("training mask geometry does not match the manifest")

            decoded = decode_coco_uncompressed_rle(annotation_entry.get("segmentation"))
            if not np.array_equal(decoded, mask):
                raise ValueError("COCO RLE does not decode to the saved mask exactly")

            contour_document = cls._read_json(paths["contours"], "contour annotation")
            expected_geometry, _contours, _outer_index = _contours_document(
                mask,
                sample["width"],
                sample["height"],
            )
            expected_contour_fields = {
                "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
                "session_id": manifest["session_id"],
                "sample_id": sample["id"],
                "image_id": sample["image_id"],
                "annotation_id": sample["annotation_id"],
                "created_at_utc": sample["created_at_utc"],
                "label_source": LABEL_SOURCE,
                "review_status": sample["review_status"],
                "model_id": manifest["model_id"],
                "category": manifest["category"],
                "image_size": {
                    "width": sample["width"],
                    "height": sample["height"],
                },
                "area": sample["area"],
                "bbox_xywh": sample["bbox"],
                "diagnostics": sample["diagnostics"],
                "prompts": sample["prompts"],
                "contours": expected_geometry,
            }
            for field in (
                "review_revision",
                "reviewed_at_utc",
                "reviewer_metadata",
                "review_history",
                "original_pseudo_label",
            ):
                if field in sample:
                    expected_contour_fields[field] = sample[field]
            if any(
                contour_document.get(key) != value
                for key, value in expected_contour_fields.items()
            ):
                raise ValueError("contour annotation does not match the saved mask")

            original = sample.get("original_pseudo_label")
            if original is not None:
                original_mask_payload = paths["original_mask"].read_bytes()
                if hashlib.sha256(original_mask_payload).hexdigest() != original["mask_sha256"]:
                    raise ValueError("original pseudo mask SHA-256 does not match")
                original_mask = _decode_png(
                    original_mask_payload,
                    cv2.IMREAD_GRAYSCALE,
                    "original pseudo mask",
                )
                original_overlay = _decode_png(
                    paths["original_qa_overlay"].read_bytes(),
                    cv2.IMREAD_COLOR,
                    "original pseudo QA overlay",
                )
                if (
                    original_mask.shape != expected_shape
                    or np.any((original_mask != 0) & (original_mask != 255))
                ):
                    raise ValueError("original pseudo mask is not the declared binary image")
                if original_overlay.shape[:2] != expected_shape:
                    raise ValueError("original pseudo QA overlay dimensions do not match")
                original_area, original_bbox = _mask_area_and_bbox(original_mask)
                if (
                    original_area != original["area"]
                    or original_bbox != original["bbox"]
                ):
                    raise ValueError("original pseudo mask geometry does not match")
                original_geometry, _original_contours, _original_outer = (
                    _contours_document(
                        original_mask,
                        sample["width"],
                        sample["height"],
                    )
                )
                original_contour_document = cls._read_json(
                    paths["original_contours"],
                    "original pseudo contour annotation",
                )
                expected_original_fields = {
                    "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
                    "session_id": manifest["session_id"],
                    "sample_id": sample["id"],
                    "image_id": sample["image_id"],
                    "annotation_id": sample["annotation_id"],
                    "created_at_utc": sample["created_at_utc"],
                    "label_source": LABEL_SOURCE,
                    "review_status": REVIEW_STATUS,
                    "model_id": manifest["model_id"],
                    "category": manifest["category"],
                    "image_size": {
                        "width": sample["width"],
                        "height": sample["height"],
                    },
                    "area": original["area"],
                    "bbox_xywh": original["bbox"],
                    "diagnostics": sample["diagnostics"],
                    "prompts": sample["prompts"],
                    "contours": original_geometry,
                }
                if any(
                    original_contour_document.get(key) != value
                    for key, value in expected_original_fields.items()
                ):
                    raise ValueError("original pseudo contour annotation is invalid")

    @classmethod
    def _validate_sample_entry(
        cls,
        directory: Path,
        sample: Any,
        source_size: list[int] | None,
        model_id: str,
        seen_ids: set[str],
    ) -> dict[str, Path]:
        required = {
            "id",
            "image_id",
            "annotation_id",
            "created_at_utc",
            "image",
            "mask",
            "contours",
            "qa_overlay",
            "width",
            "height",
            "category_id",
            "area",
            "bbox",
            "label_source",
            "review_status",
            "model_id",
            "diagnostics",
            "prompts",
            "metadata",
            "image_sha256",
            "mask_sha256",
        }
        if not isinstance(sample, dict) or not required.issubset(sample):
            raise ValueError("manifest contains an invalid training sample")
        sample_id = sample["id"]
        if (
            not isinstance(sample_id, str)
            or not sample_id.startswith("sample_")
            or not sample_id[len("sample_") :].isdigit()
            or sample_id in seen_ids
        ):
            raise ValueError("manifest contains an invalid or duplicate sample id")
        seen_ids.add(sample_id)
        for name in ("image_id", "annotation_id", "width", "height", "area"):
            value = sample[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"sample {name} must be a positive integer")
        if source_size != [sample["width"], sample["height"]]:
            raise ValueError("sample dimensions do not match manifest source_size")
        if sample["category_id"] != 1:
            raise ValueError("sample category_id must be 1")
        if (
            sample["label_source"] != LABEL_SOURCE
            or sample["review_status"] not in SAMPLE_REVIEW_STATUSES
        ):
            raise ValueError("sample review provenance is invalid")
        if sample["model_id"] != model_id:
            raise ValueError("sample model_id does not match the training manifest")
        if not isinstance(sample["diagnostics"], dict) or not isinstance(
            sample["prompts"], dict
        ) or not isinstance(sample["metadata"], dict):
            raise ValueError("sample diagnostics, prompts, and metadata must be objects")
        diagnostics = sample["diagnostics"]
        if (
            diagnostics.get("source_size") != [sample["width"], sample["height"]]
            or diagnostics.get("model_id") != model_id
        ):
            raise ValueError("sample diagnostics do not match the image or model")
        foreground_count, background_count = _validate_saved_prompts(
            sample["prompts"], sample["width"], sample["height"]
        )
        if (
            diagnostics.get("foreground_point_count") != foreground_count
            or diagnostics.get("background_point_count") != background_count
        ):
            raise ValueError("sample diagnostics point counts do not match prompts")
        bbox = sample["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
            or bbox[0] < 0
            or bbox[1] < 0
            or bbox[2] <= 0
            or bbox[3] <= 0
            or bbox[0] + bbox[2] > sample["width"]
            or bbox[1] + bbox[3] > sample["height"]
        ):
            raise ValueError("sample bbox is outside its image")
        for name in ("image_sha256", "mask_sha256"):
            digest = sample[name]
            if not _is_sha256_digest(digest):
                raise ValueError(f"sample {name} is not a SHA-256 digest")
        expected_paths = {
            "image": ("images", f"{sample_id}.png"),
            "mask": ("masks", f"{sample_id}.png"),
            "contours": ("contours", f"{sample_id}.json"),
            "qa_overlay": ("overlays", f"{sample_id}.png"),
        }
        paths: dict[str, Path] = {}
        for field, (folder, filename) in expected_paths.items():
            value = sample[field]
            if value != f"{folder}/{filename}":
                raise ValueError(f"sample {field} path does not match its sample id")
            paths[field] = cls._require_session_file(directory, value, folder)

        status = sample["review_status"]
        review_fields = (
            "review_revision",
            "reviewed_at_utc",
            "reviewer_metadata",
            "review_history",
            "original_pseudo_label",
        )
        if status == REVIEW_STATUS:
            if any(field in sample for field in review_fields):
                raise ValueError("pending sample cannot contain completed review provenance")
            return paths

        if not all(field in sample for field in review_fields):
            raise ValueError("reviewed sample is missing review provenance")
        revision = sample["review_revision"]
        reviewed_at = sample["reviewed_at_utc"]
        reviewer_metadata = sample["reviewer_metadata"]
        history = sample["review_history"]
        original = sample["original_pseudo_label"]
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision <= 0
            or not isinstance(reviewed_at, str)
            or not reviewed_at
            or not isinstance(reviewer_metadata, dict)
            or not isinstance(history, list)
            or len(history) != revision
            or not isinstance(original, dict)
        ):
            raise ValueError("reviewed sample contains invalid review provenance")
        original_required = {
            "label_source",
            "review_status",
            "archived_at_utc",
            "mask",
            "contours",
            "qa_overlay",
            "mask_sha256",
            "area",
            "bbox",
        }
        if not original_required.issubset(original):
            raise ValueError("reviewed sample is missing original pseudo provenance")
        if (
            original["label_source"] != LABEL_SOURCE
            or original["review_status"] != REVIEW_STATUS
            or not isinstance(original["archived_at_utc"], str)
            or not original["archived_at_utc"]
            or not _is_sha256_digest(original["mask_sha256"])
            or isinstance(original["area"], bool)
            or not isinstance(original["area"], int)
            or original["area"] <= 0
        ):
            raise ValueError("original pseudo provenance is invalid")
        original_bbox = original["bbox"]
        if (
            not isinstance(original_bbox, list)
            or len(original_bbox) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in original_bbox
            )
            or original_bbox[0] < 0
            or original_bbox[1] < 0
            or original_bbox[2] <= 0
            or original_bbox[3] <= 0
            or original_bbox[0] + original_bbox[2] > sample["width"]
            or original_bbox[1] + original_bbox[3] > sample["height"]
        ):
            raise ValueError("original pseudo bbox is outside its image")
        original_expected_paths = {
            "mask": ("masks", f"masks/pseudo/{sample_id}.png"),
            "contours": ("contours", f"contours/pseudo/{sample_id}.json"),
            "qa_overlay": ("overlays", f"overlays/pseudo/{sample_id}.png"),
        }
        for field, (folder, expected) in original_expected_paths.items():
            if original[field] != expected:
                raise ValueError(f"original pseudo {field} path is invalid")
            paths[f"original_{field}"] = cls._require_session_file(
                directory,
                original[field],
                folder,
            )
        previous_status = REVIEW_STATUS
        previous_result_sha = original["mask_sha256"]
        for expected_revision, event in enumerate(history, start=1):
            if not isinstance(event, dict):
                raise ValueError("review history contains a non-object event")
            required_event = {
                "revision",
                "decision",
                "review_status",
                "reviewed_at_utc",
                "reviewer_metadata",
                "mask_corrected",
                "previous_review_status",
                "previous_mask_sha256",
                "result_mask_sha256",
            }
            if not required_event.issubset(event):
                raise ValueError("review history event is incomplete")
            if (
                event["revision"] != expected_revision
                or event["decision"] not in {
                    HUMAN_VERIFIED_REVIEW_STATUS,
                    REJECTED_REVIEW_STATUS,
                }
                or event["review_status"] != event["decision"]
                or event["previous_review_status"] != previous_status
                or not isinstance(event["reviewed_at_utc"], str)
                or not event["reviewed_at_utc"]
                or not isinstance(event["reviewer_metadata"], dict)
                or not isinstance(event["mask_corrected"], bool)
                or not _is_sha256_digest(event["previous_mask_sha256"])
                or not _is_sha256_digest(event["result_mask_sha256"])
                or event["previous_mask_sha256"] != previous_result_sha
            ):
                raise ValueError("review history event is invalid")
            if event["decision"] == REJECTED_REVIEW_STATUS and (
                event["mask_corrected"]
                or event["result_mask_sha256"] != event["previous_mask_sha256"]
            ):
                raise ValueError("rejected review cannot modify the canonical mask")
            previous_status = event["review_status"]
            previous_result_sha = event["result_mask_sha256"]
        latest = history[-1]
        if (
            latest["review_status"] != status
            or latest["reviewed_at_utc"] != reviewed_at
            or latest["reviewer_metadata"] != reviewer_metadata
            or latest["result_mask_sha256"] != sample["mask_sha256"]
        ):
            raise ValueError("sample does not match its latest review event")
        return paths

    @staticmethod
    def _require_session_file(directory: Path, value: Any, folder: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("dataset file paths must be non-empty strings")
        relative = Path(value)
        if relative.is_absolute() or not relative.parts or relative.parts[0] != folder:
            raise ValueError(f"dataset path must remain inside {folder}/")
        expected_root = (directory / folder).resolve()
        resolved = (directory / relative).resolve()
        try:
            resolved.relative_to(expected_root)
        except ValueError as exc:
            raise ValueError(f"dataset path escapes the {folder} directory") from exc
        if not resolved.is_file():
            raise ValueError(f"dataset references a missing file: {value}")
        return resolved

    @property
    def source_size(self) -> Optional[tuple[int, int]]:
        with self._lock:
            value = self._manifest["source_size"]
            return None if value is None else (int(value[0]), int(value[1]))

    def read_manifest(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._manifest)

    def read_instances(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._instances)

    def _next_identifiers(self) -> tuple[str, int, int]:
        image_id = max(
            (int(entry["id"]) for entry in self._instances["images"]),
            default=0,
        ) + 1
        annotation_id = max(
            (int(entry["id"]) for entry in self._instances["annotations"]),
            default=0,
        ) + 1
        while True:
            sample_id = f"sample_{image_id:06d}"
            candidate_paths = (
                self.session_directory / "images" / f"{sample_id}.png",
                self.session_directory / "masks" / f"{sample_id}.png",
                self.session_directory / "contours" / f"{sample_id}.json",
                self.session_directory / "overlays" / f"{sample_id}.png",
            )
            if not any(path.exists() for path in candidate_paths):
                return sample_id, image_id, annotation_id
            image_id += 1
            annotation_id += 1

    def _lock_or_check_source_size(
        self,
        manifest: dict[str, Any],
        width: int,
        height: int,
    ) -> None:
        actual = [width, height]
        expected = manifest["source_size"]
        if expected is None:
            manifest["source_size"] = actual
        elif expected != actual:
            raise ValueError(
                f"image size {(width, height)} does not match session source size "
                f"{tuple(expected)}"
            )

    def save_sample(
        self,
        image_bgr: np.ndarray,
        result: Sam2SegmentationResult,
        prompts: SegmentationPrompts,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Sam2TrainingSampleRecord:
        """Atomically append one exact-frame SAM2 pseudo-label sample."""

        image = _validated_bgr(image_bgr)
        if not isinstance(result, Sam2SegmentationResult):
            raise TypeError("result must be a Sam2SegmentationResult")
        if not isinstance(prompts, SegmentationPrompts):
            raise TypeError("prompts must be a SegmentationPrompts")
        height, width = image.shape[:2]
        mask = _validated_mask(result.mask, (height, width))
        if result.diagnostics.source_size != (width, height):
            raise ValueError("result diagnostics size does not match image_bgr")
        if (
            result.diagnostics.foreground_point_count != len(prompts.foreground_points)
            or result.diagnostics.background_point_count != len(prompts.background_points)
        ):
            raise ValueError("result diagnostics point counts do not match prompts")
        converted_metadata = _json_compatible({} if metadata is None else metadata)
        if not isinstance(converted_metadata, dict):
            raise TypeError("metadata must be a mapping")
        _strict_json_bytes(converted_metadata)
        diagnostics = _json_compatible(result.diagnostics)
        assert isinstance(diagnostics, dict)
        prompt_document = _prompt_document(prompts, width, height)
        contour_geometry, _contours, _outer_index = _contours_document(
            mask,
            width,
            height,
        )
        area, bbox = _mask_area_and_bbox(mask)
        rle = encode_coco_uncompressed_rle(mask)
        if not np.array_equal(decode_coco_uncompressed_rle(rle), mask):
            raise RuntimeError("internal COCO RLE round-trip check failed")
        overlay = _qa_overlay(image, mask, prompts)
        image_payload = _encoded_png(image, "training image")
        mask_payload = _encoded_png(mask, "training mask")
        overlay_payload = _encoded_png(overlay, "QA overlay")
        image_sha256 = hashlib.sha256(image_payload).hexdigest()
        mask_sha256 = hashlib.sha256(mask_payload).hexdigest()

        with self._lock:
            if result.diagnostics.model_id != self._manifest["model_id"]:
                raise ValueError("result model_id does not match the training session")
            manifest = deepcopy(self._manifest)
            instances = deepcopy(self._instances)
            self._lock_or_check_source_size(manifest, width, height)
            sample_id, image_id, annotation_id = self._next_identifiers()
            created_at = _utc_timestamp()
            image_relative = Path("images") / f"{sample_id}.png"
            mask_relative = Path("masks") / f"{sample_id}.png"
            contours_relative = Path("contours") / f"{sample_id}.json"
            overlay_relative = Path("overlays") / f"{sample_id}.png"
            contour_document = {
                "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
                "session_id": self.session_id,
                "sample_id": sample_id,
                "image_id": image_id,
                "annotation_id": annotation_id,
                "created_at_utc": created_at,
                "label_source": LABEL_SOURCE,
                "review_status": REVIEW_STATUS,
                "model_id": self._manifest["model_id"],
                "category": deepcopy(self._manifest["category"]),
                "image_size": {"width": width, "height": height},
                "area": area,
                "bbox_xywh": bbox,
                "diagnostics": diagnostics,
                "prompts": prompt_document,
                "contours": contour_geometry,
            }
            contour_payload = _strict_json_bytes(contour_document)

            sample_entry = {
                "id": sample_id,
                "image_id": image_id,
                "annotation_id": annotation_id,
                "created_at_utc": created_at,
                "image": image_relative.as_posix(),
                "mask": mask_relative.as_posix(),
                "contours": contours_relative.as_posix(),
                "qa_overlay": overlay_relative.as_posix(),
                "image_sha256": image_sha256,
                "mask_sha256": mask_sha256,
                "width": width,
                "height": height,
                "category_id": 1,
                "area": area,
                "bbox": bbox,
                "label_source": LABEL_SOURCE,
                "review_status": REVIEW_STATUS,
                "model_id": self._manifest["model_id"],
                "diagnostics": diagnostics,
                "prompts": prompt_document,
                "metadata": converted_metadata,
            }
            manifest["samples"].append(sample_entry)
            manifest["review_status"] = _aggregate_review_status(manifest["samples"])
            manifest["updated_at_utc"] = created_at

            image_entry = {
                "id": image_id,
                "file_name": image_relative.as_posix(),
                "width": width,
                "height": height,
                "date_captured": created_at,
                "license": 0,
                "sam2_sample_id": sample_id,
                "image_sha256": image_sha256,
                "metadata": converted_metadata,
            }
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "label_source": LABEL_SOURCE,
                "review_status": REVIEW_STATUS,
                "model_id": self._manifest["model_id"],
                "mask_file": mask_relative.as_posix(),
                "contours_file": contours_relative.as_posix(),
                "qa_overlay_file": overlay_relative.as_posix(),
                "mask_sha256": mask_sha256,
                "diagnostics": diagnostics,
                "prompts": prompt_document,
            }
            instances["images"].append(image_entry)
            instances["annotations"].append(annotation_entry)
            instances["info"]["review_status"] = manifest["review_status"]

            files = (
                (self.session_directory / image_relative, image_payload),
                (self.session_directory / mask_relative, mask_payload),
                (self.session_directory / contours_relative, contour_payload),
                (self.session_directory / overlay_relative, overlay_payload),
            )
            self._commit_sample(manifest, instances, files)
            return Sam2TrainingSampleRecord(
                sample_id=sample_id,
                image_id=image_id,
                annotation_id=annotation_id,
                image_path=self.session_directory / image_relative,
                mask_path=self.session_directory / mask_relative,
                contours_path=self.session_directory / contours_relative,
                overlay_path=self.session_directory / overlay_relative,
            )

    def review_sample(
        self,
        sample_id: str,
        corrected_mask: Optional[np.ndarray],
        *,
        decision: str = HUMAN_VERIFIED_REVIEW_STATUS,
        reviewer_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Sam2TrainingSampleRecord:
        """Atomically review one sample while preserving its original pseudo label.

        ``human_verified`` requires a complete corrected binary mask, even when
        the reviewer intentionally accepts the SAM2 mask unchanged. ``rejected``
        requires ``corrected_mask=None`` and leaves the current canonical mask
        untouched. The first review archives the pending SAM2 mask, contour JSON,
        and overlay below each asset folder's ``pseudo/`` directory.
        """

        sample_id = _validated_nonempty_string(sample_id, "sample_id")
        if (
            not sample_id.startswith("sample_")
            or not sample_id[len("sample_") :].isdigit()
        ):
            raise ValueError("sample_id must use the sample_000001 format")
        if decision not in {
            HUMAN_VERIFIED_REVIEW_STATUS,
            REJECTED_REVIEW_STATUS,
        }:
            raise ValueError("decision must be human_verified or rejected")
        if decision == HUMAN_VERIFIED_REVIEW_STATUS and corrected_mask is None:
            raise ValueError("human_verified review requires corrected_mask")
        if decision == REJECTED_REVIEW_STATUS and corrected_mask is not None:
            raise ValueError("rejected review must not provide corrected_mask")
        converted_reviewer_metadata = _json_compatible(
            {} if reviewer_metadata is None else reviewer_metadata
        )
        if not isinstance(converted_reviewer_metadata, dict):
            raise TypeError("reviewer_metadata must be a mapping")
        _strict_json_bytes(converted_reviewer_metadata)

        with self._lock:
            manifest = deepcopy(self._manifest)
            instances = deepcopy(self._instances)
            sample_index = next(
                (
                    index
                    for index, entry in enumerate(manifest["samples"])
                    if entry.get("id") == sample_id
                ),
                None,
            )
            if sample_index is None:
                raise KeyError(f"training sample does not exist: {sample_id}")
            sample = deepcopy(manifest["samples"][sample_index])
            annotation_index = next(
                (
                    index
                    for index, entry in enumerate(instances["annotations"])
                    if entry.get("id") == sample["annotation_id"]
                ),
                None,
            )
            if annotation_index is None:
                raise ValueError("training sample has no matching COCO annotation")
            validated_paths = self._validate_sample_entry(
                self.session_directory,
                sample,
                manifest["source_size"],
                manifest["model_id"],
                set(),
            )
            image_path = validated_paths["image"]
            mask_path = validated_paths["mask"]
            contours_path = validated_paths["contours"]
            overlay_path = validated_paths["qa_overlay"]
            image_payload = image_path.read_bytes()
            current_mask_payload = mask_path.read_bytes()
            current_contour_payload = contours_path.read_bytes()
            current_overlay_payload = overlay_path.read_bytes()
            if hashlib.sha256(image_payload).hexdigest() != sample["image_sha256"]:
                raise ValueError("training image SHA-256 does not match the manifest")
            if (
                hashlib.sha256(current_mask_payload).hexdigest()
                != sample["mask_sha256"]
            ):
                raise ValueError("training mask SHA-256 does not match the manifest")
            image = _decode_png(image_payload, cv2.IMREAD_COLOR, "training image")
            current_mask = _decode_png(
                current_mask_payload,
                cv2.IMREAD_GRAYSCALE,
                "training mask",
            )
            expected_shape = (sample["height"], sample["width"])
            if image.shape[:2] != expected_shape:
                raise ValueError("training image dimensions do not match the manifest")
            current_mask = _validated_review_mask(current_mask, expected_shape)
            _validate_saved_prompts(
                sample["prompts"],
                sample["width"],
                sample["height"],
            )
            prompts = _prompts_from_document(sample["prompts"])
            if decision == HUMAN_VERIFIED_REVIEW_STATUS:
                assert corrected_mask is not None
                reviewed_mask = _validated_review_mask(corrected_mask, expected_shape)
            else:
                reviewed_mask = current_mask

            contour_geometry, _contours, _outer_index = _contours_document(
                reviewed_mask,
                sample["width"],
                sample["height"],
            )
            area, bbox = _mask_area_and_bbox(reviewed_mask)
            segmentation = encode_coco_uncompressed_rle(reviewed_mask)
            if not np.array_equal(
                decode_coco_uncompressed_rle(segmentation), reviewed_mask
            ):
                raise RuntimeError("internal COCO RLE round-trip check failed")
            reviewed_overlay = _qa_overlay(image, reviewed_mask, prompts)
            reviewed_mask_payload = _encoded_png(reviewed_mask, "reviewed mask")
            reviewed_overlay_payload = _encoded_png(
                reviewed_overlay,
                "reviewed QA overlay",
            )
            reviewed_mask_sha256 = hashlib.sha256(reviewed_mask_payload).hexdigest()
            reviewed_at = _utc_timestamp()

            history = deepcopy(sample.get("review_history", []))
            original = deepcopy(sample.get("original_pseudo_label"))
            archive_files: list[tuple[Path, bytes]] = []
            if original is None:
                if sample["review_status"] != REVIEW_STATUS or history:
                    raise ValueError("reviewed sample is missing its original pseudo label")
                pseudo_mask_relative = Path("masks") / "pseudo" / f"{sample_id}.png"
                pseudo_contours_relative = (
                    Path("contours") / "pseudo" / f"{sample_id}.json"
                )
                pseudo_overlay_relative = (
                    Path("overlays") / "pseudo" / f"{sample_id}.png"
                )
                for relative in (
                    pseudo_mask_relative,
                    pseudo_contours_relative,
                    pseudo_overlay_relative,
                ):
                    (self.session_directory / relative).parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                original = {
                    "label_source": LABEL_SOURCE,
                    "review_status": REVIEW_STATUS,
                    "archived_at_utc": reviewed_at,
                    "mask": pseudo_mask_relative.as_posix(),
                    "contours": pseudo_contours_relative.as_posix(),
                    "qa_overlay": pseudo_overlay_relative.as_posix(),
                    "mask_sha256": sample["mask_sha256"],
                    "area": sample["area"],
                    "bbox": deepcopy(sample["bbox"]),
                }
                archive_files.extend(
                    (
                        (
                            self.session_directory / pseudo_mask_relative,
                            current_mask_payload,
                        ),
                        (
                            self.session_directory / pseudo_contours_relative,
                            current_contour_payload,
                        ),
                        (
                            self.session_directory / pseudo_overlay_relative,
                            current_overlay_payload,
                        ),
                    )
                )

            previous_status = sample["review_status"]
            previous_mask_sha256 = sample["mask_sha256"]
            event = {
                "revision": len(history) + 1,
                "decision": decision,
                "review_status": decision,
                "reviewed_at_utc": reviewed_at,
                "reviewer_metadata": converted_reviewer_metadata,
                "mask_corrected": bool(
                    decision == HUMAN_VERIFIED_REVIEW_STATUS
                    and not np.array_equal(reviewed_mask, current_mask)
                ),
                "previous_review_status": previous_status,
                "previous_mask_sha256": previous_mask_sha256,
                "result_mask_sha256": reviewed_mask_sha256,
            }
            history.append(event)
            review_fields = {
                "review_revision": len(history),
                "reviewed_at_utc": reviewed_at,
                "reviewer_metadata": converted_reviewer_metadata,
                "review_history": history,
                "original_pseudo_label": original,
            }
            sample.update(
                {
                    "area": area,
                    "bbox": bbox,
                    "mask_sha256": reviewed_mask_sha256,
                    "review_status": decision,
                    **review_fields,
                }
            )
            manifest["samples"][sample_index] = sample
            manifest["review_status"] = _aggregate_review_status(manifest["samples"])
            manifest["updated_at_utc"] = reviewed_at

            annotation = instances["annotations"][annotation_index]
            annotation.update(
                {
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "mask_sha256": reviewed_mask_sha256,
                    "review_status": decision,
                    **deepcopy(review_fields),
                }
            )
            instances["info"]["review_status"] = manifest["review_status"]

            contour_document = {
                "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
                "session_id": self.session_id,
                "sample_id": sample_id,
                "image_id": sample["image_id"],
                "annotation_id": sample["annotation_id"],
                "created_at_utc": sample["created_at_utc"],
                "label_source": LABEL_SOURCE,
                "review_status": decision,
                "model_id": self._manifest["model_id"],
                "category": deepcopy(self._manifest["category"]),
                "image_size": {
                    "width": sample["width"],
                    "height": sample["height"],
                },
                "area": area,
                "bbox_xywh": bbox,
                "diagnostics": deepcopy(sample["diagnostics"]),
                "prompts": deepcopy(sample["prompts"]),
                "contours": contour_geometry,
                **deepcopy(review_fields),
            }
            reviewed_contour_payload = _strict_json_bytes(contour_document)
            replacement_files = (
                (mask_path, reviewed_mask_payload),
                (contours_path, reviewed_contour_payload),
                (overlay_path, reviewed_overlay_payload),
            )
            self._commit_review(
                manifest,
                instances,
                replacement_files,
                tuple(archive_files),
            )
            return Sam2TrainingSampleRecord(
                sample_id=sample_id,
                image_id=sample["image_id"],
                annotation_id=sample["annotation_id"],
                image_path=image_path,
                mask_path=mask_path,
                contours_path=contours_path,
                overlay_path=overlay_path,
            )

    def _commit_review(
        self,
        manifest: dict[str, Any],
        instances: dict[str, Any],
        replacement_files: tuple[tuple[Path, bytes], ...],
        archive_files: tuple[tuple[Path, bytes], ...],
    ) -> None:
        """Publish a review and restore all prior bytes on ordinary failures."""

        old_manifest_payload = _strict_json_bytes(self._manifest)
        old_instances_payload = _strict_json_bytes(self._instances)
        old_assets = {path: path.read_bytes() for path, _payload in replacement_files}
        committed_archives: list[Path] = []
        try:
            for path, payload in archive_files:
                _atomic_write(path, payload)
                committed_archives.append(path)
            for path, payload in replacement_files:
                _atomic_write(path, payload, replace=True)
            _atomic_write(
                self.instances_path,
                _strict_json_bytes(instances),
                replace=True,
            )
            _atomic_write(
                self.manifest_path,
                _strict_json_bytes(manifest),
                replace=True,
            )
        except BaseException:
            for path, payload in old_assets.items():
                try:
                    _atomic_write(path, payload, replace=True)
                except BaseException:
                    pass
            try:
                _atomic_write(
                    self.instances_path,
                    old_instances_payload,
                    replace=True,
                )
            except BaseException:
                pass
            try:
                _atomic_write(
                    self.manifest_path,
                    old_manifest_payload,
                    replace=True,
                )
            except BaseException:
                pass
            for path in reversed(committed_archives):
                path.unlink(missing_ok=True)
            raise
        self._manifest = manifest
        self._instances = instances

    def _commit_sample(
        self,
        manifest: dict[str, Any],
        instances: dict[str, Any],
        files: tuple[tuple[Path, bytes], ...],
    ) -> None:
        committed_assets: list[Path] = []
        old_instances_payload = _strict_json_bytes(self._instances)
        instances_replaced = False
        try:
            for path, payload in files:
                _atomic_write(path, payload)
                committed_assets.append(path)
            _atomic_write(
                self.instances_path,
                _strict_json_bytes(instances),
                replace=True,
            )
            instances_replaced = True
            _atomic_write(
                self.manifest_path,
                _strict_json_bytes(manifest),
                replace=True,
            )
        except BaseException:
            if instances_replaced:
                try:
                    _atomic_write(
                        self.instances_path,
                        old_instances_payload,
                        replace=True,
                    )
                except BaseException:
                    # Preserve the original failure.  The manifest remains the
                    # authoritative committed index and open() will reject a
                    # mismatched COCO file rather than silently trust it.
                    pass
            for path in reversed(committed_assets):
                path.unlink(missing_ok=True)
            raise
        self._manifest = manifest
        self._instances = instances


__all__ = [
    "COCO_SCHEMA_VERSION",
    "HUMAN_VERIFIED_REVIEW_STATUS",
    "LABEL_SOURCE",
    "MIXED_REVIEW_STATUS",
    "REJECTED_REVIEW_STATUS",
    "REVIEW_STATUS",
    "SAMPLE_REVIEW_STATUSES",
    "SESSION_REVIEW_STATUSES",
    "Sam2TrainingDatasetSession",
    "Sam2TrainingSampleRecord",
    "TRAINING_MANIFEST_SCHEMA_VERSION",
    "decode_coco_uncompressed_rle",
    "encode_coco_uncompressed_rle",
]
