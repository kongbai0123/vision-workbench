"""Data models shared by the classical segmentation pipeline.

The public models deliberately keep user prompts in normalized coordinates.
That makes a prompt independent of preview scaling while the segmentation
implementation remains free to work at a smaller resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import operator
from typing import Optional, Sequence, Tuple

import numpy as np


Size = Tuple[int, int]
PixelPoint = Tuple[int, int]


def _validated_integer(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(parsed)


def _validated_size(width: int, height: int) -> Size:
    width = _validated_integer(width, "width")
    height = _validated_integer(height, "height")
    return width, height


@dataclass(frozen=True)
class NormalizedPoint:
    """A point expressed in inclusive normalized image coordinates."""

    u: float
    v: float

    def __post_init__(self) -> None:
        u = float(self.u)
        v = float(self.v)
        if not math.isfinite(u) or not math.isfinite(v):
            raise ValueError("normalized point coordinates must be finite")
        if not 0.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
            raise ValueError("normalized point coordinates must be within [0, 1]")
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)

    def to_pixel(
        self,
        width: int | Sequence[int],
        height: Optional[int] = None,
    ) -> PixelPoint:
        """Map the point to a valid integer pixel in a ``width x height`` image."""

        if height is None:
            if not isinstance(width, Sequence) or len(width) != 2:
                raise TypeError("provide width and height, or a (width, height) sequence")
            width, height = width
        assert height is not None
        width, height = _validated_size(width, height)
        x = min(width - 1, max(0, int(round(self.u * (width - 1)))))
        y = min(height - 1, max(0, int(round(self.v * (height - 1)))))
        return x, y


@dataclass(frozen=True)
class NormalizedRect:
    """An axis-aligned normalized rectangle with an exclusive pixel end."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in (self.left, self.top, self.right, self.bottom))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("normalized rectangle coordinates must be finite")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("normalized rectangle coordinates must be within [0, 1]")
        left, top, right, bottom = values
        if left >= right or top >= bottom:
            raise ValueError("normalized rectangle must have positive width and height")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "bottom", bottom)

    def contains(self, point: NormalizedPoint) -> bool:
        within_horizontal = self.left <= point.u < self.right or (
            self.right == 1.0 and point.u == 1.0
        )
        within_vertical = self.top <= point.v < self.bottom or (
            self.bottom == 1.0 and point.v == 1.0
        )
        return within_horizontal and within_vertical

    def to_pixels(
        self,
        width: int | Sequence[int],
        height: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` with ``x1``/``y1`` exclusive."""

        if height is None:
            if not isinstance(width, Sequence) or len(width) != 2:
                raise TypeError("provide width and height, or a (width, height) sequence")
            width, height = width
        assert height is not None
        width, height = _validated_size(width, height)
        x0 = min(width - 1, max(0, int(math.floor(self.left * width))))
        y0 = min(height - 1, max(0, int(math.floor(self.top * height))))
        x1 = min(width, max(x0 + 1, int(math.ceil(self.right * width))))
        y1 = min(height, max(y0 + 1, int(math.ceil(self.bottom * height))))
        return x0, y0, x1, y1


def _coerce_point(point: NormalizedPoint | Sequence[float]) -> NormalizedPoint:
    if isinstance(point, NormalizedPoint):
        return point
    if isinstance(point, Sequence) and len(point) == 2:
        return NormalizedPoint(point[0], point[1])
    raise TypeError("point must be a NormalizedPoint or a two-value sequence")


@dataclass(frozen=True)
class SegmentationPrompts:
    """Immutable foreground/background prompts for one segmentation request."""

    foreground_points: Tuple[NormalizedPoint, ...] = ()
    background_points: Tuple[NormalizedPoint, ...] = ()
    roi: Optional[NormalizedRect] = None

    def __post_init__(self) -> None:
        foreground = tuple(_coerce_point(point) for point in self.foreground_points)
        background = tuple(_coerce_point(point) for point in self.background_points)
        if self.roi is not None and not isinstance(self.roi, NormalizedRect):
            raise TypeError("roi must be a NormalizedRect or None")
        object.__setattr__(self, "foreground_points", foreground)
        object.__setattr__(self, "background_points", background)

    def add_foreground(
        self,
        point: NormalizedPoint | Sequence[float] | float,
        v: Optional[float] = None,
    ) -> "SegmentationPrompts":
        normalized = NormalizedPoint(point, v) if v is not None else _coerce_point(point)  # type: ignore[arg-type]
        return replace(self, foreground_points=self.foreground_points + (normalized,))

    def add_background(
        self,
        point: NormalizedPoint | Sequence[float] | float,
        v: Optional[float] = None,
    ) -> "SegmentationPrompts":
        normalized = NormalizedPoint(point, v) if v is not None else _coerce_point(point)  # type: ignore[arg-type]
        return replace(self, background_points=self.background_points + (normalized,))

    def clear_foreground(self) -> "SegmentationPrompts":
        return replace(self, foreground_points=())

    def clear_background(self) -> "SegmentationPrompts":
        return replace(self, background_points=())

    def clear_points(self) -> "SegmentationPrompts":
        return SegmentationPrompts(roi=self.roi)

    def clear_roi(self) -> "SegmentationPrompts":
        return replace(self, roi=None)

    def clear(self) -> "SegmentationPrompts":
        return SegmentationPrompts()

    def with_roi(self, roi: Optional[NormalizedRect]) -> "SegmentationPrompts":
        return replace(self, roi=roi)


@dataclass(frozen=True)
class SegmentationConfig:
    """Tunable, resolution-independent parameters for classical segmentation."""

    background_sample_count: int = 15
    max_working_dimension: int = 960
    mad_scale: float = 1.4826
    mad_floor: float = 2.0
    lab_diff_low_threshold: float = 2.5
    lab_diff_high_threshold: float = 6.0
    seed_radius_ratio: float = 0.0125
    border_background_ratio: float = 0.01
    morphology_radius_ratio: float = 0.0025
    grabcut_iterations: int = 5
    min_component_area_ratio: float = 0.0005
    max_hole_area_ratio: float = 0.0003

    def __post_init__(self) -> None:
        integer_fields = {
            "background_sample_count": self.background_sample_count,
            "max_working_dimension": self.max_working_dimension,
            "grabcut_iterations": self.grabcut_iterations,
        }
        for name, value in integer_fields.items():
            object.__setattr__(self, name, _validated_integer(value, name))

        positive_fields = {
            "mad_scale": self.mad_scale,
            "mad_floor": self.mad_floor,
            "lab_diff_high_threshold": self.lab_diff_high_threshold,
            "seed_radius_ratio": self.seed_radius_ratio,
        }
        for name, value in positive_fields.items():
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
            object.__setattr__(self, name, value)
        if self.mad_scale < 1e-6 or self.mad_floor < 1e-6:
            raise ValueError("mad_scale and mad_floor must each be at least 1e-6")

        low = float(self.lab_diff_low_threshold)
        if not math.isfinite(low) or low < 0.0:
            raise ValueError("lab_diff_low_threshold must be finite and non-negative")
        object.__setattr__(self, "lab_diff_low_threshold", low)
        if self.lab_diff_high_threshold <= low:
            raise ValueError("lab_diff_high_threshold must exceed lab_diff_low_threshold")

        ratio_limits = {
            "border_background_ratio": (self.border_background_ratio, 0.25),
            "morphology_radius_ratio": (self.morphology_radius_ratio, 0.05),
            "min_component_area_ratio": (self.min_component_area_ratio, 1.0),
            "max_hole_area_ratio": (self.max_hole_area_ratio, 1.0),
        }
        for name, (raw_value, upper_bound) in ratio_limits.items():
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= upper_bound:
                raise ValueError(f"{name} must be within [0, {upper_bound}]")
            object.__setattr__(self, name, value)
        if self.seed_radius_ratio > 0.25:
            raise ValueError("seed_radius_ratio must not exceed 0.25")


@dataclass(frozen=True, eq=False)
class BackgroundModel:
    """Robust per-pixel Lab background statistics at the working resolution."""

    median_lab: np.ndarray
    mad_lab: np.ndarray
    source_size: Size
    working_size: Size
    sample_count: int

    def __post_init__(self) -> None:
        source_size = _validated_size(*self.source_size)
        working_size = _validated_size(*self.working_size)
        sample_count = _validated_integer(self.sample_count, "sample_count")
        median_lab = np.asarray(self.median_lab)
        mad_lab = np.asarray(self.mad_lab)
        expected_shape = (working_size[1], working_size[0], 3)
        if median_lab.dtype != np.uint8 or median_lab.shape != expected_shape:
            raise ValueError(f"median_lab must be uint8 with shape {expected_shape}")
        if mad_lab.dtype != np.float32 or mad_lab.shape != expected_shape:
            raise ValueError(f"mad_lab must be float32 with shape {expected_shape}")
        if not np.all(np.isfinite(mad_lab)) or np.any(mad_lab < 0.0):
            raise ValueError("mad_lab must contain finite, non-negative values")
        median_lab = np.array(median_lab, dtype=np.uint8, order="C", copy=True)
        mad_lab = np.array(mad_lab, dtype=np.float32, order="C", copy=True)
        median_lab.setflags(write=False)
        mad_lab.setflags(write=False)
        object.__setattr__(self, "median_lab", median_lab)
        object.__setattr__(self, "mad_lab", mad_lab)
        object.__setattr__(self, "source_size", source_size)
        object.__setattr__(self, "working_size", working_size)
        object.__setattr__(self, "sample_count", sample_count)


@dataclass(frozen=True, eq=False)
class ContourHierarchy:
    """OpenCV contour tree plus the main object and its true hole indices."""

    contours: Tuple[np.ndarray, ...]
    hierarchy: np.ndarray
    outer_index: Optional[int]
    hole_indices: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        normalized_contours = []
        for raw_contour in self.contours:
            contour = np.asarray(raw_contour)
            valid_shape = (
                (contour.ndim == 2 and contour.shape[1:] == (2,))
                or (contour.ndim == 3 and contour.shape[1:] == (1, 2))
            )
            if not valid_shape or contour.shape[0] == 0:
                raise ValueError("each contour must be an Nx2 or Nx1x2 coordinate array")
            if not np.issubdtype(contour.dtype, np.number) or not np.all(np.isfinite(contour)):
                raise ValueError("contour coordinates must be finite numeric values")
            if np.issubdtype(contour.dtype, np.integer):
                int32_info = np.iinfo(np.int32)
                if np.any(contour < int32_info.min) or np.any(contour > int32_info.max):
                    raise ValueError("integer contour coordinates must fit in int32")
                contour = np.array(contour, dtype=np.int32, order="C", copy=True)
            else:
                float32_limit = np.finfo(np.float32).max
                if np.any(np.abs(contour) > float32_limit):
                    raise ValueError("floating contour coordinates must fit in float32")
                contour = np.array(contour, dtype=np.float32, order="C", copy=True)
            contour.setflags(write=False)
            normalized_contours.append(contour)
        contours = tuple(normalized_contours)

        hierarchy = np.asarray(self.hierarchy)
        expected_shape = (1, len(contours), 4)
        if hierarchy.shape != expected_shape:
            raise ValueError(f"hierarchy must have shape {expected_shape}")
        hierarchy = np.array(hierarchy, dtype=np.int32, order="C", copy=True)
        if hierarchy.size and (np.any(hierarchy < -1) or np.any(hierarchy >= len(contours))):
            raise ValueError("hierarchy contains an out-of-range contour index")

        if self.outer_index is None:
            outer_index = None
        else:
            outer_index = _validated_integer(
                self.outer_index,
                "outer_index",
                allow_zero=True,
            )
            if outer_index >= len(contours):
                raise ValueError("outer_index is outside the contour collection")
            if int(hierarchy[0, outer_index, 3]) != -1:
                raise ValueError("outer_index must refer to a top-level contour")
        if contours and outer_index is None:
            raise ValueError("outer_index is required when contours are present")

        holes = []
        for raw_index in self.hole_indices:
            index = _validated_integer(raw_index, "hole index", allow_zero=True)
            if index >= len(contours):
                raise ValueError("hole index is outside the contour collection")
            if index in holes:
                raise ValueError("hole_indices must not contain duplicates")
            holes.append(index)

        depths = []
        for index in range(len(contours)):
            depth = 0
            parent = int(hierarchy[0, index, 3])
            visited = {index}
            while parent >= 0:
                if parent in visited:
                    raise ValueError("hierarchy parent links contain a cycle")
                visited.add(parent)
                depth += 1
                parent = int(hierarchy[0, parent, 3])
            depths.append(depth)
        if any(depths[index] % 2 != 1 for index in holes):
            raise ValueError("hole_indices must contain only odd-depth contours")
        expected_holes = tuple(index for index, depth in enumerate(depths) if depth % 2 == 1)
        if tuple(holes) != expected_holes:
            raise ValueError("hole_indices must list every odd-depth contour in index order")

        hierarchy.setflags(write=False)
        object.__setattr__(self, "contours", contours)
        object.__setattr__(self, "hierarchy", hierarchy)
        object.__setattr__(self, "outer_index", outer_index)
        object.__setattr__(self, "hole_indices", tuple(holes))


@dataclass(frozen=True)
class SegmentationDiagnostics:
    """Measurements useful for UI status, tuning, and later error analysis."""

    source_size: Size
    working_size: Size
    lab_diff_low_threshold: float
    lab_diff_high_threshold: float
    difference_min: float
    difference_mean: float
    difference_max: float
    probable_foreground_ratio: float
    foreground_ratio: float
    foreground_point_count: int
    background_point_count: int
    grabcut_iterations: int
    processing_time_ms: float

    def __post_init__(self) -> None:
        source_size = _validated_size(*self.source_size)
        working_size = _validated_size(*self.working_size)
        finite_values = {
            "lab_diff_low_threshold": self.lab_diff_low_threshold,
            "lab_diff_high_threshold": self.lab_diff_high_threshold,
            "difference_min": self.difference_min,
            "difference_mean": self.difference_mean,
            "difference_max": self.difference_max,
            "processing_time_ms": self.processing_time_ms,
        }
        parsed_values = {}
        for name, raw_value in finite_values.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            parsed_values[name] = value
            object.__setattr__(self, name, value)
        if parsed_values["lab_diff_low_threshold"] < 0.0:
            raise ValueError("lab_diff_low_threshold must be non-negative")
        if parsed_values["lab_diff_high_threshold"] <= parsed_values["lab_diff_low_threshold"]:
            raise ValueError("lab_diff_high_threshold must exceed lab_diff_low_threshold")
        difference_min = parsed_values["difference_min"]
        difference_mean = parsed_values["difference_mean"]
        difference_max = parsed_values["difference_max"]
        tolerance = max(1e-6, abs(difference_min) * 1e-6, abs(difference_max) * 1e-6)
        if difference_min > difference_max or not (
            difference_min - tolerance <= difference_mean <= difference_max + tolerance
        ):
            raise ValueError("difference statistics must satisfy min <= mean <= max")
        if parsed_values["processing_time_ms"] < 0.0:
            raise ValueError("processing_time_ms must be non-negative")

        for name in ("probable_foreground_ratio", "foreground_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "foreground_point_count",
            _validated_integer(
                self.foreground_point_count,
                "foreground_point_count",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "background_point_count",
            _validated_integer(
                self.background_point_count,
                "background_point_count",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "grabcut_iterations",
            _validated_integer(self.grabcut_iterations, "grabcut_iterations"),
        )
        object.__setattr__(self, "source_size", source_size)
        object.__setattr__(self, "working_size", working_size)


@dataclass(frozen=True, eq=False)
class SegmentationResult:
    """A full-resolution binary mask, its contour tree, and diagnostics."""

    mask: np.ndarray
    hierarchy: ContourHierarchy
    diagnostics: SegmentationDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.hierarchy, ContourHierarchy):
            raise TypeError("hierarchy must be a ContourHierarchy")
        if not isinstance(self.diagnostics, SegmentationDiagnostics):
            raise TypeError("diagnostics must be a SegmentationDiagnostics")
        mask = np.asarray(self.mask)
        if mask.dtype != np.uint8 or mask.ndim != 2:
            raise ValueError("mask must be a two-dimensional uint8 array")
        expected_shape = (
            self.diagnostics.source_size[1],
            self.diagnostics.source_size[0],
        )
        if mask.shape != expected_shape:
            raise ValueError(f"mask must have source shape {expected_shape}")
        if mask.size and np.any((mask != 0) & (mask != 255)):
            raise ValueError("mask must contain only 0 and 255")
        mask = np.array(mask, dtype=np.uint8, order="C", copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
