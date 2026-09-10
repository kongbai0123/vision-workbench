"""Immutable, verified-only SAM2 training dataset exports.

The review sessions remain the mutable source of truth.  This module snapshots
only ``human_verified`` samples into a new, never-appended export directory.
Binary mask PNG files are canonical.  COCO, LabelMe, CVAT, contours, and outer
boundary images are derived from those masks and validated during the export.

Interchange references:

* COCO mask API (column-major RLE):
  https://github.com/cocodataset/cocoapi
* LabelMe's maintained JSON examples and standalone reader:
  https://github.com/wkentaro/labelme/tree/main/examples
* CVAT for images 1.1 (native mask RLE in an importable ZIP):
  https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .training_dataset import (
    decode_coco_uncompressed_rle,
    encode_coco_uncompressed_rle,
)


VERIFIED_EXPORT_SCHEMA_VERSION = 1
EXPORTER_VERSION = "1.1"
VERIFIED_STATUS = "human_verified"
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
DEFAULT_SEED = 20260903
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_PARAMETERS = (cv2.IMWRITE_PNG_COMPRESSION, 3)
_SAFE_SLUG = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class VerifiedExportResult:
    """Summary returned after a complete export has been atomically published."""

    export_directory: Path
    sample_count: int
    split_counts: dict[str, int]
    excluded_counts: dict[str, int]
    reproducibility_fingerprint: str


@dataclass(frozen=True)
class CvatMaskRle:
    """CVAT's cropped, row-major alternating RLE plus its inclusive bounds."""

    counts: tuple[int, ...]
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class _SourceSample:
    input_root: Path
    collection_id: str
    session_id: str
    session_directory: Path
    manifest_path: Path
    manifest_sha256: str
    model_id: str
    category_name: str
    sample: dict[str, Any]
    output_id: str

    @property
    def source_sample_id(self) -> str:
        return str(self.sample["id"])

    @property
    def group_key(self) -> str:
        return f"{self.collection_id}/{self.session_id}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _directory_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _strict_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{description} does not exist: {path}") from exc
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} root must be a JSON object: {path}")
    return value, payload


def _decode_png(payload: bytes, flags: int, description: str) -> np.ndarray:
    if not payload.startswith(_PNG_SIGNATURE):
        raise ValueError(f"{description} is not a PNG")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
    if decoded is None or decoded.size == 0:
        raise ValueError(f"{description} is not a readable PNG")
    return decoded


def _encode_png(image: np.ndarray, description: str) -> bytes:
    success, encoded = cv2.imencode(".png", image, _PNG_PARAMETERS)
    if not success or encoded is None or encoded.size == 0:
        raise OSError(f"could not encode {description} as PNG")
    return encoded.tobytes()


def _nonempty_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value.strip()


def _slug(value: str, fallback: str) -> str:
    slug = _SAFE_SLUG.sub("_", value).strip("_")[:40]
    return slug or fallback


def _validate_sha256(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _safe_asset_path(
    session_directory: Path,
    relative_value: Any,
    required_folder: str,
    description: str,
) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"{description} path must be a non-empty string")
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != required_folder
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ValueError(f"unsafe {description} path: {relative_value!r}")
    session_root = session_directory.resolve()
    expected_root = (session_root / required_folder).resolve()
    try:
        expected_root.relative_to(session_root)
    except ValueError as exc:
        raise ValueError(
            f"unsafe {description} folder: {required_folder!r} escapes the session"
        ) from exc
    resolved = (session_directory / relative).resolve()
    try:
        resolved.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(f"unsafe {description} path: {relative_value!r}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def _normalized_ratios(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("split ratios must contain train, val, and test")
    converted = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in converted):
        raise ValueError("split ratios must be finite and non-negative")
    total = sum(converted)
    if total <= 0.0:
        raise ValueError("at least one split ratio must be positive")
    return tuple(value / total for value in converted)  # type: ignore[return-value]


def _collection_id(
    manifest: Mapping[str, Any], input_root: Path, session_directory: Path
) -> str:
    explicit = manifest.get("collection_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    collection = manifest.get("collection")
    if isinstance(collection, Mapping):
        for key in ("id", "name"):
            value = collection.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    root_directory = input_root if input_root.is_dir() else input_root.parent
    if (root_directory / "manifest.json").resolve() == (
        session_directory / "manifest.json"
    ).resolve():
        return session_directory.parent.name
    try:
        relative = session_directory.relative_to(root_directory)
    except ValueError:
        return session_directory.parent.name
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return root_directory.name


def _candidate_manifest_paths(input_root: Path) -> list[Path]:
    if input_root.is_file():
        if input_root.name != "manifest.json":
            raise ValueError(f"input file must be named manifest.json: {input_root}")
        return [input_root.resolve()]
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    direct = input_root / "manifest.json"
    if direct.is_file():
        return [direct.resolve()]
    return sorted(path.resolve() for path in input_root.rglob("manifest.json"))


def _allocate_output_id(
    collection_id: str,
    session_id: str,
    sample_id: str,
    manifest_sha256: str,
) -> str:
    identity = "\0".join((collection_id, session_id, sample_id, manifest_sha256))
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return "__".join(
        (
            _slug(collection_id, "collection"),
            _slug(session_id, "session"),
            _slug(sample_id, "sample"),
            suffix,
        )
    )


def _validate_verified_provenance(
    sample: Mapping[str, Any], manifest_path: Path
) -> None:
    """Reject status-only edits that lack the review transaction's evidence."""

    revision = sample.get("review_revision")
    history = sample.get("review_history")
    reviewer = sample.get("reviewer_metadata")
    reviewed_at = sample.get("reviewed_at_utc")
    original = sample.get("original_pseudo_label")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision <= 0
        or not isinstance(history, list)
        or len(history) != revision
        or not isinstance(reviewer, Mapping)
        or not isinstance(reviewed_at, str)
        or not reviewed_at
        or not isinstance(original, Mapping)
        or not history
        or not isinstance(history[-1], Mapping)
    ):
        raise ValueError(
            f"human_verified sample lacks review provenance: {manifest_path}"
        )
    latest = history[-1]
    latest_status = latest.get("decision", latest.get("review_status"))
    if (
        latest.get("revision") != revision
        or latest_status != VERIFIED_STATUS
        or latest.get("result_mask_sha256") != sample.get("mask_sha256")
    ):
        raise ValueError(
            f"human_verified sample review history is inconsistent: {manifest_path}"
        )


def _discover_verified_samples(
    input_roots: Sequence[Path],
) -> tuple[list[_SourceSample], Counter[str], list[dict[str, Any]]]:
    manifests: dict[Path, Path] = {}
    for raw_root in input_roots:
        root = raw_root.expanduser().resolve()
        for manifest_path in _candidate_manifest_paths(root):
            manifests.setdefault(manifest_path, root)
    if not manifests:
        raise ValueError("no session manifest.json files were found")

    samples: list[_SourceSample] = []
    excluded: Counter[str] = Counter()
    source_records: list[dict[str, Any]] = []
    logical_identities: set[tuple[str, str, str]] = set()

    for manifest_path, input_root in sorted(manifests.items(), key=lambda item: str(item[0])):
        manifest, payload = _read_json(manifest_path, "training session manifest")
        if manifest.get("dataset_type") != "sam2_training_pseudo_labels":
            continue
        session_directory = manifest_path.parent.resolve()
        session_id = _nonempty_string(manifest.get("session_id"), "session_id")
        if session_id != session_directory.name:
            raise ValueError(
                f"session_id does not match directory name in {manifest_path}"
            )
        model_id = _nonempty_string(manifest.get("model_id"), "model_id")
        category = manifest.get("category")
        if not isinstance(category, Mapping):
            raise ValueError(f"manifest category is invalid: {manifest_path}")
        category_name = _nonempty_string(category.get("name"), "category name")
        collection_id = _collection_id(manifest, input_root, session_directory)
        manifest_digest = _sha256(payload)
        manifest_samples = manifest.get("samples")
        if not isinstance(manifest_samples, list):
            raise ValueError(f"manifest samples must be an array: {manifest_path}")

        status_counts: Counter[str] = Counter()
        for raw_sample in manifest_samples:
            if not isinstance(raw_sample, dict):
                raise ValueError(f"manifest contains a non-object sample: {manifest_path}")
            status = raw_sample.get("review_status")
            status_name = status if isinstance(status, str) and status else "invalid_status"
            status_counts[status_name] += 1
            if status != VERIFIED_STATUS:
                excluded[status_name] += 1
                continue
            _validate_verified_provenance(raw_sample, manifest_path)
            sample_id = _nonempty_string(raw_sample.get("id"), "sample id")
            logical_identity = (collection_id, session_id, sample_id)
            if logical_identity in logical_identities:
                raise ValueError(
                    "duplicate collection/session/sample identity: "
                    + "/".join(logical_identity)
                )
            logical_identities.add(logical_identity)
            output_id = _allocate_output_id(
                collection_id, session_id, sample_id, manifest_digest
            )
            samples.append(
                _SourceSample(
                    input_root=input_root,
                    collection_id=collection_id,
                    session_id=session_id,
                    session_directory=session_directory,
                    manifest_path=manifest_path,
                    manifest_sha256=manifest_digest,
                    model_id=model_id,
                    category_name=category_name,
                    sample=deepcopy(raw_sample),
                    output_id=output_id,
                )
            )
        source_records.append(
            {
                "collection_id": collection_id,
                "session_id": session_id,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_digest,
                "model_id": model_id,
                "category_name": category_name,
                "status_counts": dict(sorted(status_counts.items())),
            }
        )

    if not source_records:
        raise ValueError("no SAM2 training session manifests were found")
    if not samples:
        raise ValueError("no human_verified samples were found")
    samples.sort(
        key=lambda item: (
            item.collection_id,
            item.session_id,
            item.source_sample_id,
            item.output_id,
        )
    )
    return samples, excluded, source_records


def _group_quotas(group_count: int, ratios: Sequence[float]) -> list[int]:
    """Allocate groups while keeping every positive split represented when possible."""

    counts = [0, 0, 0]
    active = [index for index, ratio in enumerate(ratios) if ratio > 0.0]
    if group_count <= 0:
        return counts
    if group_count < len(active):
        ranked = sorted(active, key=lambda index: (-ratios[index], index))
        for index in ranked[:group_count]:
            counts[index] += 1
        return counts
    for index in active:
        counts[index] = 1
    for _ in range(group_count - len(active)):
        index = max(
            active,
            key=lambda candidate: (
                ratios[candidate] * group_count - counts[candidate],
                ratios[candidate],
                -candidate,
            ),
        )
        counts[index] += 1
    return counts


def _assign_splits(
    samples: Sequence[_SourceSample], seed: int, ratios: Sequence[float]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    groups = sorted({sample.group_key for sample in samples})
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    quotas = _group_quotas(len(shuffled), ratios)
    assignment: dict[str, str] = {}
    grouped: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    offset = 0
    for split, count in zip(SPLIT_NAMES, quotas):
        selected = sorted(shuffled[offset : offset + count])
        offset += count
        grouped[split] = selected
        for key in selected:
            assignment[key] = split
    if set(assignment) != set(groups):
        raise AssertionError("internal split assignment did not cover every group")
    return assignment, grouped


def _mask_geometry(mask: np.ndarray) -> tuple[int, list[int]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("verified mask is empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return int(xs.size), [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _training_prompt(mask: np.ndarray, bbox: Sequence[int]) -> dict[str, Any]:
    """Create deterministic SAM/SAM2 bbox and point prompts from ground truth."""

    x, y, width, height = (int(value) for value in bbox)
    binary = np.asarray(mask != 0, dtype=np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    point_y, point_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    image_height, image_width = binary.shape
    return {
        "bbox_xyxy": [x, y, x + width - 1, y + height - 1],
        "bbox_coordinate_convention": "inclusive_xyxy",
        "positive_point_xy": [int(point_x), int(point_y)],
        "positive_point_normalized": [
            float(point_x) / max(1, image_width - 1),
            float(point_y) / max(1, image_height - 1),
        ],
        "point_strategy": "maximum_inside_distance_from_verified_mask_boundary",
    }


def _contour_depth(index: int, hierarchy: np.ndarray) -> int:
    depth = 0
    visited = {index}
    parent = int(hierarchy[0, index, 3])
    while parent >= 0:
        if parent in visited:
            raise ValueError("cyclic contour hierarchy")
        visited.add(parent)
        depth += 1
        parent = int(hierarchy[0, parent, 3])
    return depth


def _contours_from_mask(
    mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    height, width = mask.shape
    component_count, _labels = cv2.connectedComponents(
        np.asarray(mask != 0, dtype=np.uint8), connectivity=8
    )
    if component_count != 2:
        raise ValueError(
            "verified mask must contain exactly one connected foreground object"
        )
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None or not contours:
        raise ValueError("verified mask has no contour")
    top_level = [
        index for index in range(len(contours)) if int(hierarchy[0, index, 3]) == -1
    ]
    if len(top_level) != 1:
        raise ValueError("verified mask must contain exactly one outer contour")
    outer_index = top_level[0]
    x_denominator = max(1, width - 1)
    y_denominator = max(1, height - 1)

    def record(index: int) -> dict[str, Any]:
        points = np.asarray(contours[index]).reshape(-1, 2)
        pixels = [[int(x), int(y)] for x, y in points]
        return {
            "hierarchy_index": int(index),
            "parent_index": int(hierarchy[0, index, 3]),
            "point_count": len(pixels),
            "closed": True,
            "pixels": pixels,
            "normalized": [
                [float(x) / x_denominator, float(y) / y_denominator]
                for x, y in pixels
            ],
        }

    holes = [
        record(index)
        for index in range(len(contours))
        if _contour_depth(index, hierarchy) % 2 == 1
    ]
    outer = record(outer_index)
    boundary = np.zeros_like(mask)
    cv2.drawContours(
        boundary,
        [contours[outer_index]],
        -1,
        255,
        1,
        lineType=cv2.LINE_8,
    )
    document = {
        "point_reduction": {"stage": "verified_export", "method": "CHAIN_APPROX_SIMPLE", "tolerance_px": 0, "canonical_mask_unchanged": True},
        "coordinate_convention": {
            "pixel": "inclusive_xy; origin=top_left; x=column; y=row",
            "normalized": "x/(width-1), y/(height-1)",
        },
        "outer": outer,
        "holes": holes,
        "hole_policy": (
            "holes are excluded from outer/boundary, but preserved exactly in "
            "canonical mask PNG, COCO RLE, LabelMe mask, and CVAT mask"
        ),
    }
    return document, boundary, contours[outer_index]


def encode_cvat_cropped_rle(mask: np.ndarray) -> CvatMaskRle:
    """Encode one binary object using CVAT's cropped row-major mask RLE."""

    array = np.asarray(mask)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("mask must be a non-empty two-dimensional array")
    if not np.all((array == 0) | (array == 1) | (array == 255)):
        raise ValueError("mask must contain only 0, 1, or 255")
    binary = np.asarray(array != 0, dtype=np.uint8)
    _area, (left, top, width, height) = _mask_geometry(binary)
    crop = binary[top : top + height, left : left + width]
    flat = crop.reshape(-1, order="C")
    counts: list[int] = []
    current = 0
    length = 0
    for raw_value in flat:
        value = int(raw_value)
        if value == current:
            length += 1
        else:
            counts.append(length)
            current = value
            length = 1
    counts.append(length)
    return CvatMaskRle(tuple(counts), left, top, width, height)


def decode_cvat_cropped_rle(
    encoded: CvatMaskRle, image_size: tuple[int, int]
) -> np.ndarray:
    """Decode :func:`encode_cvat_cropped_rle`; useful for export validation."""

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")
    if (
        encoded.left < 0
        or encoded.top < 0
        or encoded.width <= 0
        or encoded.height <= 0
        or encoded.left + encoded.width > width
        or encoded.top + encoded.height > height
        or not encoded.counts
        or any(count < 0 for count in encoded.counts)
        or sum(encoded.counts) != encoded.width * encoded.height
    ):
        raise ValueError("invalid CVAT cropped RLE")
    flat = np.zeros(encoded.width * encoded.height, dtype=np.uint8)
    offset = 0
    value = 0
    for count in encoded.counts:
        flat[offset : offset + count] = value
        offset += count
        value = 1 - value
    output = np.zeros((height, width), dtype=np.uint8)
    output[
        encoded.top : encoded.top + encoded.height,
        encoded.left : encoded.left + encoded.width,
    ] = flat.reshape((encoded.height, encoded.width), order="C") * 255
    return output


def _load_verified_assets(
    source: _SourceSample,
) -> tuple[bytes, bytes, np.ndarray, int, int, int, list[int]]:
    sample = source.sample
    for name in ("width", "height"):
        value = sample.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{source.group_key}/{source.source_sample_id}: invalid {name}"
            )
    width, height = int(sample["width"]), int(sample["height"])
    image_path = _safe_asset_path(
        source.session_directory, sample.get("image"), "images", "source image"
    )
    mask_path = _safe_asset_path(
        source.session_directory, sample.get("mask"), "masks", "canonical mask"
    )
    image_payload = image_path.read_bytes()
    mask_payload = mask_path.read_bytes()
    declared_image_digest = _validate_sha256(
        sample.get("image_sha256"), "source image_sha256"
    )
    declared_mask_digest = _validate_sha256(
        sample.get("mask_sha256"), "source mask_sha256"
    )
    if _sha256(image_payload) != declared_image_digest:
        raise ValueError(f"source image hash mismatch: {image_path}")
    if _sha256(mask_payload) != declared_mask_digest:
        raise ValueError(f"canonical mask hash mismatch: {mask_path}")
    image = _decode_png(image_payload, cv2.IMREAD_COLOR, "source image")
    mask = _decode_png(mask_payload, cv2.IMREAD_GRAYSCALE, "canonical mask")
    if image.shape[:2] != (height, width) or mask.shape != (height, width):
        raise ValueError(
            f"source dimensions disagree with manifest: {source.group_key}/"
            f"{source.source_sample_id}"
        )
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError("canonical mask PNG must contain only 0 and 255")
    area, bbox = _mask_geometry(mask)
    if sample.get("area") != area or sample.get("bbox") != bbox:
        raise ValueError(
            f"canonical mask geometry disagrees with manifest: {source.group_key}/"
            f"{source.source_sample_id}"
        )
    return image_payload, mask_payload, mask, width, height, area, bbox


def _labelme_document(
    *,
    source: _SourceSample,
    image_relative_from_labelme: str,
    mask: np.ndarray,
    width: int,
    height: int,
    bbox: Sequence[int],
) -> dict[str, Any]:
    x, y, box_width, box_height = (int(value) for value in bbox)
    crop = mask[y : y + box_height, x : x + box_width]
    crop_payload = _encode_png(crop, "LabelMe mask crop")
    return {
        "version": "5.7.0",
        "flags": {"human_verified": True},
        "shapes": [
            {
                "label": source.category_name,
                "points": [
                    [float(x), float(y)],
                    [float(x + box_width - 1), float(y + box_height - 1)],
                ],
                "group_id": 1,
                "description": (
                    "Exact verified binary mask. Internal zero pixels are holes; "
                    "see canonical contours JSON for explicit coordinates."
                ),
                "shape_type": "mask",
                "flags": {"human_verified": True},
                "mask": base64.b64encode(crop_payload).decode("ascii"),
            }
        ],
        "imagePath": image_relative_from_labelme,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def _coco_document(
    split: str,
    entries: Sequence[dict[str, Any]],
    category_ids: Mapping[str, int],
    created_at: str,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for identifier, entry in enumerate(entries, start=1):
        source: _SourceSample = entry["source"]
        images.append(
            {
                "id": identifier,
                "file_name": entry["image_relative"],
                "width": entry["width"],
                "height": entry["height"],
                "date_captured": source.sample.get("created_at_utc", ""),
                "license": 0,
                "verified_export_id": source.output_id,
                "collection_id": source.collection_id,
                "session_id": source.session_id,
                "source_sample_id": source.source_sample_id,
            }
        )
        annotations.append(
            {
                "id": identifier,
                "image_id": identifier,
                "category_id": category_ids[source.category_name],
                "segmentation": encode_coco_uncompressed_rle(entry["mask"]),
                "area": entry["area"],
                "bbox": entry["bbox"],
                "iscrowd": 0,
                "review_status": VERIFIED_STATUS,
                "verified_export_id": source.output_id,
                "mask_file": entry["mask_relative"],
                "outer_boundary_file": entry["boundary_relative"],
                "contours_file": entry["contour_relative"],
            }
        )
    return {
        "info": {
            "version": "1.0",
            "description": f"Human-verified SAM2 segmentation ({split})",
            "date_created": created_at,
            "canonical_label_format": "binary_mask_png",
            "segmentation_format": "coco_uncompressed_rle_column_major",
            "holes": "preserved exactly by RLE",
        },
        "licenses": [],
        "categories": [
            {"id": identifier, "name": name, "supercategory": "object"}
            for name, identifier in sorted(category_ids.items(), key=lambda item: item[1])
        ],
        "images": images,
        "annotations": annotations,
    }


def _write_flat_training_ready(
    directory: Path,
    entries: Sequence[dict[str, Any]],
    category_ids: Mapping[str, int],
    created_at: str,
) -> dict[str, Any]:
    """Write one drag-and-drop folder containing only images and JSON labels."""

    training_ready = directory / "training_ready"
    training_ready.mkdir()
    ordered = sorted(entries, key=lambda entry: entry["source"].output_id)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for identifier, entry in enumerate(ordered, start=1):
        source: _SourceSample = entry["source"]
        image_name = f"{source.output_id}.png"
        image_path = training_ready / image_name
        if image_path.exists():
            raise FileExistsError(f"duplicate flat training image: {image_name}")
        image_path.write_bytes(entry["image_payload"])
        segmentation = encode_coco_uncompressed_rle(entry["mask"])
        if not np.array_equal(
            decode_coco_uncompressed_rle(segmentation),
            entry["mask"],
        ):
            raise AssertionError("flat COCO RLE failed an internal exact round trip")
        category_id = category_ids[source.category_name]
        images.append(
            {
                "id": identifier,
                "file_name": image_name,
                "width": entry["width"],
                "height": entry["height"],
                "date_captured": source.sample.get("created_at_utc", ""),
                "license": 0,
                "verified_export_id": source.output_id,
                "collection_id": source.collection_id,
                "session_id": source.session_id,
                "source_sample_id": source.source_sample_id,
                "split": entry["split"],
            }
        )
        annotations.append(
            {
                "id": identifier,
                "image_id": identifier,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": entry["area"],
                "bbox": entry["bbox"],
                "iscrowd": 0,
                "review_status": VERIFIED_STATUS,
                "verified_export_id": source.output_id,
            }
        )
        samples.append(
            {
                "id": source.output_id,
                "image": image_name,
                "image_id": identifier,
                "annotation_id": identifier,
                "category_id": category_id,
                "category_name": source.category_name,
                "split": entry["split"],
                "review_status": VERIFIED_STATUS,
                "image_sha256": _sha256(entry["image_payload"]),
                "source": {
                    "collection_id": source.collection_id,
                    "session_id": source.session_id,
                    "sample_id": source.source_sample_id,
                    "manifest": str(source.manifest_path),
                },
            }
        )
    coco = {
        "info": {
            "version": "1.0",
            "description": "Flat human-verified SAM2 training handoff",
            "date_created": created_at,
            "canonical_label_format": "coco_uncompressed_rle_column_major",
            "verified_only": True,
            "holes": "preserved exactly by RLE",
        },
        "licenses": [],
        "categories": [
            {"id": identifier, "name": name, "supercategory": "object"}
            for name, identifier in sorted(category_ids.items(), key=lambda item: item[1])
        ],
        "images": images,
        "annotations": annotations,
    }
    annotation_payload = _strict_json_bytes(coco)
    (training_ready / "annotations.json").write_bytes(annotation_payload)
    flat_manifest = {
        "schema_version": VERIFIED_EXPORT_SCHEMA_VERSION,
        "dataset_type": "sam2_human_verified_flat_training",
        "created_at_utc": created_at,
        "verified_only": True,
        "sample_count": len(samples),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "annotation_format": "coco_uncompressed_rle_column_major",
        "annotations": "annotations.json",
        "annotations_sha256": _sha256(annotation_payload),
        "excluded_content": ["pending", "rejected", "pseudo", "overlay"],
        "categories": coco["categories"],
        "samples": samples,
    }
    (training_ready / "manifest.json").write_bytes(_strict_json_bytes(flat_manifest))
    return {
        "path": "training_ready",
        "sample_count": len(samples),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "annotations": "training_ready/annotations.json",
        "manifest": "training_ready/manifest.json",
        "contains_only_verified_images_and_json": True,
    }


def _append_xml_text(parent: ET.Element, name: str, value: Any) -> ET.Element:
    element = ET.SubElement(parent, name)
    element.text = str(value)
    return element


def _cvat_xml(
    split: str,
    entries: Sequence[dict[str, Any]],
    categories: Sequence[str],
    created_at: str,
) -> bytes:
    annotations = ET.Element("annotations")
    _append_xml_text(annotations, "version", "1.1")
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    _append_xml_text(task, "id", 0)
    _append_xml_text(task, "name", f"sam2_verified_{split}")
    _append_xml_text(task, "size", len(entries))
    _append_xml_text(task, "mode", "annotation")
    _append_xml_text(task, "overlap", 0)
    _append_xml_text(task, "bugtracker", "")
    _append_xml_text(task, "flipped", "False")
    _append_xml_text(task, "created", created_at)
    _append_xml_text(task, "updated", created_at)
    labels = ET.SubElement(task, "labels")
    for category in categories:
        label = ET.SubElement(labels, "label")
        _append_xml_text(label, "name", category)
        _append_xml_text(label, "type", "mask")
        ET.SubElement(label, "attributes")
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    _append_xml_text(segment, "id", 0)
    _append_xml_text(segment, "start", 0)
    _append_xml_text(segment, "stop", max(-1, len(entries) - 1))
    _append_xml_text(segment, "url", "")
    owner = ET.SubElement(task, "owner")
    _append_xml_text(owner, "username", "")
    _append_xml_text(owner, "email", "")
    _append_xml_text(meta, "dumped", created_at)

    for identifier, entry in enumerate(entries):
        source: _SourceSample = entry["source"]
        image = ET.SubElement(
            annotations,
            "image",
            {
                "id": str(identifier),
                # Media lives below images/ in the archive, while CVAT's XML
                # name is relative to that media root.
                "name": f"{source.output_id}.png",
                "width": str(entry["width"]),
                "height": str(entry["height"]),
            },
        )
        encoded = encode_cvat_cropped_rle(entry["mask"])
        if not np.array_equal(
            decode_cvat_cropped_rle(encoded, (entry["width"], entry["height"])),
            entry["mask"],
        ):
            raise AssertionError("CVAT RLE failed an internal exact round trip")
        ET.SubElement(
            image,
            "mask",
            {
                "label": source.category_name,
                "source": "manual",
                "occluded": "0",
                "rle": ", ".join(str(value) for value in encoded.counts),
                "left": str(encoded.left),
                "top": str(encoded.top),
                "width": str(encoded.width),
                "height": str(encoded.height),
                "z_order": "0",
            },
        )
    ET.indent(annotations, space="  ")
    return ET.tostring(annotations, encoding="utf-8", xml_declaration=True)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_cvat_archive(
    path: Path,
    split: str,
    entries: Sequence[dict[str, Any]],
    categories: Sequence[str],
    created_at: str,
) -> None:
    xml_payload = _cvat_xml(split, entries, categories, created_at)
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_zip_info("annotations.xml"), xml_payload)
        for entry in entries:
            source: _SourceSample = entry["source"]
            archive.writestr(
                _zip_info(f"images/{source.output_id}.png"), entry["image_payload"]
            )


def _write_readme(directory: Path) -> None:
    text = """# Human-verified segmentation export

This directory is an immutable snapshot: create a new export after any review
change; do not edit this directory in place.

## Canonical training data

- `images/{train,val,test}`: lossless source PNG images.
- `masks/{train,val,test}`: canonical 0/255 binary ground-truth masks.
- `boundaries/{train,val,test}`: one-pixel 0/255 **main outer boundary only**.
- `contours/{train,val,test}`: main outer contour and holes in pixel and normalized
  coordinates. Pixel origin is the top-left; `x` is column and `y` is row.
- `annotations/instances_*.json`: COCO instances using exact uncompressed,
  column-major RLE.
- `labelme/{train,val,test}/*.json`: LabelMe mask shapes that refer to the
  canonical images. The mask shape, rather than polygons, preserves holes.
- `cvat/cvat_for_images_*.zip`: self-contained CVAT for images 1.1 archives
  (only non-empty splits receive an archive).
- `indexes/{train,val,test,all}.jsonl`: loader-friendly image/mask records with
  category, `bbox_xyxy`, and a deterministic positive point prompt for SAM/SAM2.
- `training_ready`: one flat drag-and-drop folder containing only verified source
  PNG images plus `annotations.json` (exact COCO RLE masks) and `manifest.json`.
- `splits.json`: deterministic group assignment and sample lists.
- `export_manifest.json`: review provenance, source and exported hashes.
- `checksums.sha256`: SHA-256 for every other file in this export.

## Hole policy

Internal holes are not part of `boundaries/*.png` or the `outer` contour. They
remain explicit under `contours.*.holes` and are preserved pixel-for-pixel in
mask PNG, COCO RLE, LabelMe mask, and CVAT mask outputs. Do not replace these
masks with a single polygon when hole accuracy matters.

## Leakage policy

All samples sharing the same `collection_id/session_id` group are assigned to
exactly one split. Adjacent frames from one capture session therefore cannot
leak across train, validation, and test.

## Format references

- COCO API: https://github.com/cocodataset/cocoapi
- LabelMe examples: https://github.com/wkentaro/labelme/tree/main/examples
- CVAT for images 1.1: https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/
"""
    (directory / "README.md").write_text(text, encoding="utf-8", newline="\n")


def _verify_source_manifests(sources: Iterable[_SourceSample]) -> None:
    expected: dict[Path, str] = {}
    for source in sources:
        expected[source.manifest_path] = source.manifest_sha256
    for path, digest in expected.items():
        if _sha256(path.read_bytes()) != digest:
            raise RuntimeError(f"source manifest changed during export: {path}")


def _write_checksums(directory: Path) -> None:
    lines = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        relative = path.relative_to(directory).as_posix()
        lines.append(f"{_sha256(path.read_bytes())}  {relative}")
    (directory / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _allocate_export_directories(output_root: Path, name: str | None) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = "export"
    if name is not None:
        prefix += "_" + _slug(_nonempty_string(name, "export name"), "dataset")
    unique = f"{prefix}_{_directory_timestamp()}_{uuid.uuid4().hex[:8]}"
    final = output_root / unique
    staging = output_root / f".{unique}.staging"
    staging.mkdir()
    return staging, final


def export_verified_dataset(
    input_roots: Sequence[str | os.PathLike[str]],
    output_root: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    name: str | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> VerifiedExportResult:
    """Snapshot reviewed samples into a new immutable multi-format export.

    ``input_roots`` may contain individual sessions, collection directories, or
    a common directory containing multiple collections/sessions.  Only sample
    entries whose own ``review_status`` is exactly ``human_verified`` are read.
    Pending and rejected assets are never copied.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not input_roots:
        raise ValueError("at least one input root is required")
    roots = [Path(value).expanduser().resolve() for value in input_roots]
    ratios = _normalized_ratios(split_ratios)
    samples, excluded, source_records = _discover_verified_samples(roots)
    total_steps = len(samples) + 3
    if progress_callback is not None:
        progress_callback(
            f"已找到 {len(samples)} 筆已核准樣本",
            0,
            total_steps,
        )
    assignment, groups_by_split = _assign_splits(samples, seed, ratios)
    output_directory = Path(output_root).expanduser().resolve()
    staging, final = _allocate_export_directories(output_directory, name)
    created_at = _utc_now()

    try:
        for folder in ("images", "masks", "boundaries", "contours", "labelme"):
            for split in SPLIT_NAMES:
                (staging / folder / split).mkdir(parents=True, exist_ok=True)
        (staging / "annotations").mkdir()
        (staging / "cvat").mkdir()
        (staging / "indexes").mkdir()
        _write_readme(staging)

        category_names = sorted({source.category_name for source in samples})
        category_ids = {name: index for index, name in enumerate(category_names, 1)}
        entries_by_split: dict[str, list[dict[str, Any]]] = {
            split: [] for split in SPLIT_NAMES
        }
        exported_samples: list[dict[str, Any]] = []

        for sample_index, source in enumerate(samples, 1):
            split = assignment[source.group_key]
            (
                image_payload,
                mask_payload,
                mask,
                width,
                height,
                area,
                bbox,
            ) = _load_verified_assets(source)
            contours, boundary, _outer_array = _contours_from_mask(mask)
            training_prompt = _training_prompt(mask, bbox)
            boundary_payload = _encode_png(boundary, "outer boundary")
            image_relative = f"images/{split}/{source.output_id}.png"
            mask_relative = f"masks/{split}/{source.output_id}.png"
            boundary_relative = f"boundaries/{split}/{source.output_id}.png"
            contour_relative = f"contours/{split}/{source.output_id}.json"
            labelme_relative = f"labelme/{split}/{source.output_id}.json"
            contour_document = {
                "schema_version": VERIFIED_EXPORT_SCHEMA_VERSION,
                "export_sample_id": source.output_id,
                "collection_id": source.collection_id,
                "session_id": source.session_id,
                "source_sample_id": source.source_sample_id,
                "review_status": VERIFIED_STATUS,
                "image_size": {"width": width, "height": height},
                "area": area,
                "bbox_xywh": bbox,
                "contours": contours,
            }
            contour_payload = _strict_json_bytes(contour_document)
            labelme_document = _labelme_document(
                source=source,
                image_relative_from_labelme=f"../../{image_relative}",
                mask=mask,
                width=width,
                height=height,
                bbox=bbox,
            )
            labelme_payload = _strict_json_bytes(labelme_document)

            assets = {
                image_relative: image_payload,
                mask_relative: mask_payload,
                boundary_relative: boundary_payload,
                contour_relative: contour_payload,
                labelme_relative: labelme_payload,
            }
            for relative, payload in assets.items():
                path = staging / relative
                if path.exists():
                    raise FileExistsError(f"duplicate export asset: {relative}")
                path.write_bytes(payload)

            coco_rle = encode_coco_uncompressed_rle(mask)
            if not np.array_equal(decode_coco_uncompressed_rle(coco_rle), mask):
                raise AssertionError("COCO RLE failed an internal exact round trip")
            entry = {
                "source": source,
                "image_payload": image_payload,
                "mask": mask,
                "width": width,
                "height": height,
                "area": area,
                "bbox": bbox,
                "image_relative": image_relative,
                "mask_relative": mask_relative,
                "boundary_relative": boundary_relative,
                "contour_relative": contour_relative,
                "labelme_relative": labelme_relative,
                "training_prompt": training_prompt,
                "split": split,
            }
            entries_by_split[split].append(entry)
            exported_samples.append(
                {
                    "id": source.output_id,
                    "split": split,
                    "category": {
                        "id": category_ids[source.category_name],
                        "name": source.category_name,
                    },
                    "width": width,
                    "height": height,
                    "area": area,
                    "bbox_xywh": bbox,
                    "outer_contour_point_count": contours["outer"]["point_count"],
                    "hole_count": len(contours["holes"]),
                    "outer_boundary_pixel_count": int(np.count_nonzero(boundary)),
                    "training_prompt": deepcopy(training_prompt),
                    "assets": {
                        "image": {
                            "path": image_relative,
                            "sha256": _sha256(image_payload),
                        },
                        "mask": {
                            "path": mask_relative,
                            "sha256": _sha256(mask_payload),
                        },
                        "outer_boundary": {
                            "path": boundary_relative,
                            "sha256": _sha256(boundary_payload),
                        },
                        "contours": {
                            "path": contour_relative,
                            "sha256": _sha256(contour_payload),
                        },
                        "labelme": {
                            "path": labelme_relative,
                            "sha256": _sha256(labelme_payload),
                        },
                    },
                    "provenance": {
                        "collection_id": source.collection_id,
                        "session_id": source.session_id,
                        "source_session_directory": str(source.session_directory),
                        "source_manifest": str(source.manifest_path),
                        "source_manifest_sha256": source.manifest_sha256,
                        "source_sample_id": source.source_sample_id,
                        "source_image_path": source.sample.get("image"),
                        "source_mask_path": source.sample.get("mask"),
                        "source_image_sha256": source.sample.get("image_sha256"),
                        "source_mask_sha256": source.sample.get("mask_sha256"),
                        "label_source": source.sample.get("label_source"),
                        "review_status": source.sample.get("review_status"),
                        "review_history": deepcopy(source.sample.get("review_history", [])),
                        "original_pseudo_label": deepcopy(
                            source.sample.get("original_pseudo_label")
                        ),
                        "model_id": source.model_id,
                        "created_at_utc": source.sample.get("created_at_utc"),
                        "diagnostics": deepcopy(source.sample.get("diagnostics", {})),
                        "prompts": deepcopy(source.sample.get("prompts", {})),
                        "metadata": deepcopy(source.sample.get("metadata", {})),
                    },
                }
            )
            if progress_callback is not None:
                progress_callback(
                    f"整理已核准樣本 {sample_index}/{len(samples)}",
                    sample_index,
                    total_steps,
                )

        exported_samples.sort(key=lambda item: item["id"])
        training_index_records: dict[str, list[dict[str, Any]]] = {
            split: [] for split in SPLIT_NAMES
        }
        for split in SPLIT_NAMES:
            entries_by_split[split].sort(key=lambda entry: entry["source"].output_id)
            for entry in entries_by_split[split]:
                source = entry["source"]
                training_index_records[split].append(
                    {
                        "id": source.output_id,
                        "split": split,
                        "image": entry["image_relative"],
                        "mask": entry["mask_relative"],
                        "outer_boundary": entry["boundary_relative"],
                        "contours": entry["contour_relative"],
                        "category_id": category_ids[source.category_name],
                        "category_name": source.category_name,
                        "width": entry["width"],
                        "height": entry["height"],
                        "area": entry["area"],
                        "bbox_xywh": entry["bbox"],
                        "bbox": entry["training_prompt"]["bbox_xyxy"],
                        "point": entry["training_prompt"]["positive_point_xy"],
                        "prompt": deepcopy(entry["training_prompt"]),
                        "collection_id": source.collection_id,
                        "session_id": source.session_id,
                        "source_sample_id": source.source_sample_id,
                        "review_status": VERIFIED_STATUS,
                    }
                )
            coco = _coco_document(
                split, entries_by_split[split], category_ids, created_at
            )
            coco_path = staging / "annotations" / f"instances_{split}.json"
            coco_path.write_bytes(_strict_json_bytes(coco))
            if entries_by_split[split]:
                _write_cvat_archive(
                    staging / "cvat" / f"cvat_for_images_{split}.zip",
                    split,
                    entries_by_split[split],
                    category_names,
                    created_at,
                )

        index_files: dict[str, dict[str, Any]] = {}
        all_index_records: list[dict[str, Any]] = []
        for split in SPLIT_NAMES:
            all_index_records.extend(training_index_records[split])
            payload = b"".join(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
                for record in training_index_records[split]
            )
            relative = f"indexes/{split}.jsonl"
            (staging / relative).write_bytes(payload)
            index_files[split] = {
                "path": relative,
                "sha256": _sha256(payload),
                "sample_count": len(training_index_records[split]),
            }
        all_index_records.sort(key=lambda record: record["id"])
        all_payload = b"".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
            for record in all_index_records
        )
        (staging / "indexes" / "all.jsonl").write_bytes(all_payload)
        index_files["all"] = {
            "path": "indexes/all.jsonl",
            "sha256": _sha256(all_payload),
            "sample_count": len(all_index_records),
        }

        split_document = {
            "schema_version": VERIFIED_EXPORT_SCHEMA_VERSION,
            "seed": seed,
            "algorithm": "deterministic_session_group_shuffle_v1",
            "normalized_ratios": dict(zip(SPLIT_NAMES, ratios)),
            "group_key": "collection_id/session_id",
            "groups": groups_by_split,
            "samples": {
                split: [
                    entry["source"].output_id for entry in entries_by_split[split]
                ]
                for split in SPLIT_NAMES
            },
        }
        (staging / "splits.json").write_bytes(_strict_json_bytes(split_document))

        flat_entries = [
            entry
            for split in SPLIT_NAMES
            for entry in entries_by_split[split]
        ]
        training_ready = _write_flat_training_ready(
            staging,
            flat_entries,
            category_ids,
            created_at,
        )
        if progress_callback is not None:
            progress_callback("建立訓練索引與標註", len(samples) + 1, total_steps)

        fingerprint_input = {
            "exporter_version": EXPORTER_VERSION,
            "seed": seed,
            "ratios": list(ratios),
            "sources": [
                {
                    "collection_id": record["collection_id"],
                    "session_id": record["session_id"],
                    "manifest_sha256": record["manifest_sha256"],
                }
                for record in source_records
            ],
            "verified_assets": [
                {
                    "id": sample["id"],
                    "split": sample["split"],
                    "image_sha256": sample["assets"]["image"]["sha256"],
                    "mask_sha256": sample["assets"]["mask"]["sha256"],
                }
                for sample in exported_samples
            ],
        }
        fingerprint = _sha256(_strict_json_bytes(fingerprint_input))
        manifest = {
            "schema_version": VERIFIED_EXPORT_SCHEMA_VERSION,
            "dataset_type": "sam2_human_verified_immutable_export",
            "exporter_version": EXPORTER_VERSION,
            "created_at_utc": created_at,
            "immutable": True,
            "canonical_label_format": "binary_mask_png",
            "verified_only": True,
            "included_review_status": VERIFIED_STATUS,
            "excluded_status_counts": dict(sorted(excluded.items())),
            "sample_count": len(exported_samples),
            "split_counts": {
                split: len(entries_by_split[split]) for split in SPLIT_NAMES
            },
            "seed": seed,
            "normalized_split_ratios": dict(zip(SPLIT_NAMES, ratios)),
            "categories": [
                {"id": identifier, "name": name}
                for name, identifier in sorted(
                    category_ids.items(), key=lambda item: item[1]
                )
            ],
            "training_indexes": index_files,
            "training_ready": training_ready,
            "reproducibility_fingerprint": fingerprint,
            "hole_policy": {
                "outer_boundary_excludes_holes": True,
                "canonical_mask_preserves_holes": True,
                "coco_rle_preserves_holes": True,
                "labelme_mask_preserves_holes": True,
                "cvat_mask_preserves_holes": True,
            },
            "source_roots": [str(root) for root in roots],
            "source_sessions": source_records,
            "samples": exported_samples,
        }
        (staging / "export_manifest.json").write_bytes(_strict_json_bytes(manifest))
        _verify_source_manifests(samples)
        if progress_callback is not None:
            progress_callback("驗證來源與寫入檢查碼", len(samples) + 2, total_steps)
        _write_checksums(staging)
        if final.exists():
            raise FileExistsError(f"refusing to overwrite an export: {final}")
        staging.rename(final)
        if progress_callback is not None:
            progress_callback("封裝完成", total_steps, total_steps)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return VerifiedExportResult(
        export_directory=final,
        sample_count=len(samples),
        split_counts={split: len(entries_by_split[split]) for split in SPLIT_NAMES},
        excluded_counts=dict(sorted(excluded.items())),
        reproducibility_fingerprint=fingerprint,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export only human_verified SAM2 samples to an immutable, "
            "group-leakage-safe training package."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Session, collection, or common dataset directories",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Parent directory in which a new export_* directory is created",
    )
    parser.add_argument("--name", help="Optional human-readable export name")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        result = export_verified_dataset(
            arguments.inputs,
            arguments.output_root,
            seed=arguments.seed,
            split_ratios=(
                arguments.train_ratio,
                arguments.val_ratio,
                arguments.test_ratio,
            ),
            name=arguments.name,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "export_directory": str(result.export_directory),
                "sample_count": result.sample_count,
                "split_counts": result.split_counts,
                "excluded_counts": result.excluded_counts,
                "reproducibility_fingerprint": result.reproducibility_fingerprint,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI wrapper
    sys.exit(main())
