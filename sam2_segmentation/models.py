"""Dependency-free data models for prompted SAM2 image segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from classical_segmentation.models import ContourHierarchy


Size = Tuple[int, int]


class Sam2Error(RuntimeError):
    """Base class for recoverable SAM2 integration errors."""


class Sam2DependencyError(Sam2Error):
    """Raised when PyTorch or Transformers is not installed correctly."""


class Sam2ModelLoadError(Sam2Error):
    """Raised when the configured checkpoint cannot be loaded."""


class Sam2DeviceError(Sam2Error):
    """Raised when the requested inference device is unavailable."""


class Sam2InferenceError(Sam2Error):
    """Raised when a loaded model cannot complete an inference request."""


class Sam2NoMaskError(Sam2InferenceError):
    """Raised when no candidate produces a usable prompted object mask."""


class Sam2WorkerState(str, Enum):
    """Observable lifecycle states for the asynchronous inference worker."""

    IDLE = "idle"
    LOADING = "loading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


def _validate_size(value: Size, name: str = "size") -> Size:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a (width, height) tuple")
    width, height = value
    if isinstance(width, bool) or isinstance(height, bool):
        raise TypeError(f"{name} dimensions must be integers")
    if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)):
        raise TypeError(f"{name} dimensions must be integers")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} dimensions must be greater than zero")
    return width, height


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class Sam2Config:
    """Runtime and post-processing options for the SAM2.1 tiny adapter.

    The model dependency is optional.  Constructing this configuration or
    importing this package never imports PyTorch or Transformers.
    """

    model_id: str = "facebook/sam2.1-hiera-tiny"
    cache_dir: Optional[Path] = None
    device_preference: str = "auto"
    allow_cpu_fallback: bool = True
    cuda_dtype: str = "float16"
    multimask_output: bool = True
    mask_threshold: float = 0.0
    iou_score_window: float = 0.06
    prompt_search_radius_ratio: float = 0.005
    smart_boundary_smoothing: bool = True
    boundary_smoothing_radius: int = 4
    local_files_only: bool = False

    def __post_init__(self) -> None:
        model_id = str(self.model_id).strip()
        if not model_id:
            raise ValueError("model_id must not be empty")
        object.__setattr__(self, "model_id", model_id)

        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))

        device = str(self.device_preference).strip().casefold()
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device_preference must be 'auto', 'cuda', or 'cpu'")
        object.__setattr__(self, "device_preference", device)

        dtype = str(self.cuda_dtype).strip().casefold()
        if dtype not in {"float16", "bfloat16"}:
            raise ValueError("cuda_dtype must be 'float16' or 'bfloat16'")
        object.__setattr__(self, "cuda_dtype", dtype)

        for name in (
            "allow_cpu_fallback",
            "multimask_output",
            "smart_boundary_smoothing",
            "local_files_only",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

        mask_threshold = _finite_float(self.mask_threshold, "mask_threshold")
        object.__setattr__(self, "mask_threshold", mask_threshold)
        iou_score_window = _finite_float(self.iou_score_window, "iou_score_window")
        if not 0.0 <= iou_score_window <= 1.0:
            raise ValueError("iou_score_window must be within [0, 1]")
        object.__setattr__(self, "iou_score_window", iou_score_window)
        radius = _finite_float(
            self.prompt_search_radius_ratio,
            "prompt_search_radius_ratio",
        )
        if not 0.0 <= radius <= 0.05:
            raise ValueError("prompt_search_radius_ratio must be within [0, 0.05]")
        object.__setattr__(self, "prompt_search_radius_ratio", radius)
        smoothing_radius = _validate_nonnegative_integer(
            self.boundary_smoothing_radius,
            "boundary_smoothing_radius",
        )
        if smoothing_radius > 32:
            raise ValueError("boundary_smoothing_radius must not exceed 32")
        object.__setattr__(self, "boundary_smoothing_radius", smoothing_radius)


@dataclass(frozen=True, eq=False)
class Sam2RawOutput:
    """Full-resolution candidate masks produced by a SAM2 runtime."""

    candidate_masks: np.ndarray
    iou_scores: np.ndarray
    object_score_logit: Optional[float] = None

    def __post_init__(self) -> None:
        masks = np.asarray(self.candidate_masks)
        if masks.ndim != 3 or masks.shape[0] <= 0 or masks.shape[1] <= 0 or masks.shape[2] <= 0:
            raise ValueError("candidate_masks must have shape (candidate, height, width)")
        if not (
            masks.dtype == np.bool_
            or np.issubdtype(masks.dtype, np.integer)
            or np.issubdtype(masks.dtype, np.floating)
        ):
            raise TypeError("candidate_masks must contain boolean or numeric values")
        if np.issubdtype(masks.dtype, np.floating) and not np.all(np.isfinite(masks)):
            raise ValueError("candidate_masks must contain finite values")
        if masks.dtype != np.bool_:
            valid_binary_values = (masks == 0) | (masks == 1) | (masks == 255)
            if not np.all(valid_binary_values):
                raise ValueError("candidate_masks must contain only binary values")
        masks = np.ascontiguousarray(masks != 0, dtype=np.bool_)

        scores = np.asarray(self.iou_scores, dtype=np.float32)
        if scores.ndim != 1 or scores.shape[0] != masks.shape[0]:
            raise ValueError("iou_scores must contain one score per candidate mask")
        if not np.all(np.isfinite(scores)):
            raise ValueError("iou_scores must contain finite values")
        scores = np.ascontiguousarray(scores, dtype=np.float32)

        object_score = self.object_score_logit
        if object_score is not None:
            object_score = _finite_float(object_score, "object_score_logit")

        masks.setflags(write=False)
        scores.setflags(write=False)
        object.__setattr__(self, "candidate_masks", masks)
        object.__setattr__(self, "iou_scores", scores)
        object.__setattr__(self, "object_score_logit", object_score)


@dataclass(frozen=True)
class Sam2Diagnostics:
    """Measurements useful to the UI, dataset writer, and error analysis."""

    source_size: Size
    model_id: str
    device: str
    dtype: str
    candidate_index: int
    candidate_count: int
    predicted_iou: float
    object_score_logit: Optional[float]
    prompt_violation_count: int
    foreground_ratio: float
    foreground_point_count: int
    background_point_count: int
    processing_time_ms: float
    model_load_time_ms: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_size", _validate_size(self.source_size, "source_size"))
        for name in ("model_id", "device", "dtype"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)

        candidate_index = _validate_nonnegative_integer(self.candidate_index, "candidate_index")
        candidate_count = _validate_nonnegative_integer(self.candidate_count, "candidate_count")
        if candidate_count == 0 or candidate_index >= candidate_count:
            raise ValueError("candidate_index must refer to an available candidate")
        object.__setattr__(self, "candidate_index", candidate_index)
        object.__setattr__(self, "candidate_count", candidate_count)

        object.__setattr__(self, "predicted_iou", _finite_float(self.predicted_iou, "predicted_iou"))
        if self.object_score_logit is not None:
            object.__setattr__(
                self,
                "object_score_logit",
                _finite_float(self.object_score_logit, "object_score_logit"),
            )
        for name in (
            "prompt_violation_count",
            "foreground_point_count",
            "background_point_count",
        ):
            object.__setattr__(
                self,
                name,
                _validate_nonnegative_integer(getattr(self, name), name),
            )
        foreground_ratio = _finite_float(self.foreground_ratio, "foreground_ratio")
        if not 0.0 <= foreground_ratio <= 1.0:
            raise ValueError("foreground_ratio must be within [0, 1]")
        object.__setattr__(self, "foreground_ratio", foreground_ratio)
        for name in ("processing_time_ms", "model_load_time_ms"):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class Sam2SegmentationResult:
    """A canonical binary SAM2 mask plus its reusable display contour tree.

    The hierarchy may be curvature-smoothed for rendering; ``mask`` retains
    the selected model pixels used by training and exact mask metrics.
    """

    mask: np.ndarray
    hierarchy: ContourHierarchy
    diagnostics: Sam2Diagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.hierarchy, ContourHierarchy):
            raise TypeError("hierarchy must be a ContourHierarchy")
        if not isinstance(self.diagnostics, Sam2Diagnostics):
            raise TypeError("diagnostics must be a Sam2Diagnostics")
        mask = np.asarray(self.mask)
        if mask.dtype != np.uint8 or mask.ndim != 2:
            raise ValueError("mask must be a two-dimensional uint8 array")
        width, height = self.diagnostics.source_size
        if mask.shape != (height, width):
            raise ValueError("mask must have the diagnostics source resolution")
        if mask.size and np.any((mask != 0) & (mask != 255)):
            raise ValueError("mask must contain only 0 and 255")
        mask = np.ascontiguousarray(mask, dtype=np.uint8).copy()
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)


__all__ = [
    "Sam2Config",
    "Sam2DependencyError",
    "Sam2DeviceError",
    "Sam2Diagnostics",
    "Sam2Error",
    "Sam2InferenceError",
    "Sam2ModelLoadError",
    "Sam2NoMaskError",
    "Sam2RawOutput",
    "Sam2SegmentationResult",
    "Sam2WorkerState",
]
