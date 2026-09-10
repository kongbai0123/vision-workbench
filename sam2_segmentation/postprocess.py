"""Pure prompt conversion and candidate-mask selection for SAM2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import cv2
import numpy as np

from classical_segmentation import (
    ContourHierarchy,
    SegmentationPrompts,
    extract_contour_hierarchy,
    keep_prompt_components,
)

from .models import Sam2NoMaskError, Sam2RawOutput


PixelPoint = Tuple[int, int]


@dataclass(frozen=True)
class Sam2PromptEncoding:
    """One-image, one-object point prompt in original-image coordinates."""

    foreground_pixels: Tuple[PixelPoint, ...]
    background_pixels: Tuple[PixelPoint, ...]

    def __post_init__(self) -> None:
        if not self.foreground_pixels:
            raise ValueError("at least one foreground point is required")

    @property
    def all_pixels(self) -> Tuple[PixelPoint, ...]:
        return self.foreground_pixels + self.background_pixels

    def processor_kwargs(self) -> dict[str, list]:
        """Return Hugging Face's [image][object][point][xy] list layout."""

        points = [[[list(point) for point in self.all_pixels]]]
        labels = [[
            [1] * len(self.foreground_pixels)
            + [0] * len(self.background_pixels)
        ]]
        return {"input_points": points, "input_labels": labels}


@dataclass(frozen=True, eq=False)
class Sam2CandidateSelection:
    """The best raw candidate before main-component cleanup."""

    mask: np.ndarray
    candidate_index: int
    predicted_iou: float
    prompt_violation_count: int
    main_component_area: int
    touches_image_border: bool

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.size:
            raise ValueError("mask must be a non-empty two-dimensional bool array")
        if isinstance(self.candidate_index, bool) or not isinstance(
            self.candidate_index, (int, np.integer)
        ):
            raise TypeError("candidate_index must be an integer")
        if int(self.candidate_index) < 0:
            raise ValueError("candidate_index must be non-negative")
        predicted_iou = float(self.predicted_iou)
        if not math.isfinite(predicted_iou):
            raise ValueError("predicted_iou must be finite")
        for name in ("prompt_violation_count", "main_component_area"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.touches_image_border, bool):
            raise TypeError("touches_image_border must be a bool")
        mask = np.ascontiguousarray(mask, dtype=np.bool_).copy()
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        object.__setattr__(self, "predicted_iou", predicted_iou)
        object.__setattr__(self, "prompt_violation_count", int(self.prompt_violation_count))
        object.__setattr__(self, "main_component_area", int(self.main_component_area))


def encode_sam2_prompts(
    prompts: SegmentationPrompts,
    source_size: tuple[int, int],
) -> Sam2PromptEncoding:
    """Map normalized application prompts to original image pixel coordinates."""

    if not isinstance(prompts, SegmentationPrompts):
        raise TypeError("prompts must be a SegmentationPrompts")
    if not prompts.foreground_points:
        raise ValueError("at least one foreground point is required")
    if not isinstance(source_size, tuple) or len(source_size) != 2:
        raise TypeError("source_size must be a (width, height) tuple")
    width, height = source_size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, np.integer))
        or not isinstance(height, (int, np.integer))
    ):
        raise TypeError("source_size dimensions must be integers")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("source_size dimensions must be greater than zero")
    return Sam2PromptEncoding(
        foreground_pixels=tuple(
            point.to_pixel(width, height) for point in prompts.foreground_points
        ),
        background_pixels=tuple(
            point.to_pixel(width, height) for point in prompts.background_points
        ),
    )


def _point_neighborhood_has_foreground(
    mask: np.ndarray,
    point: PixelPoint,
    radius: int,
) -> bool:
    x, y = point
    height, width = mask.shape
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = mask[y0:y1, x0:x1]
    if radius <= 0:
        return bool(patch[0, 0])
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
    return bool(np.any(patch[disk]))


def _validated_pixel_points(
    points: Sequence[PixelPoint],
    width: int,
    height: int,
    name: str,
) -> Tuple[PixelPoint, ...]:
    validated = []
    for raw_point in points:
        if not isinstance(raw_point, Sequence) or len(raw_point) != 2:
            raise ValueError(f"{name} entries must have two coordinates")
        raw_x, raw_y = raw_point
        if (
            isinstance(raw_x, bool)
            or isinstance(raw_y, bool)
            or not isinstance(raw_x, (int, np.integer))
            or not isinstance(raw_y, (int, np.integer))
        ):
            raise TypeError(f"{name} coordinates must be integers")
        x, y = int(raw_x), int(raw_y)
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(f"{name} point {(x, y)} lies outside the mask")
        validated.append((x, y))
    return tuple(validated)


def _prompt_violation_count(
    mask: np.ndarray,
    foreground_pixels: Sequence[PixelPoint],
    background_pixels: Sequence[PixelPoint],
    search_radius: int,
) -> int:
    violations = sum(
        not _point_neighborhood_has_foreground(mask, point, search_radius)
        for point in foreground_pixels
    )
    # A negative prompt is a precise exclusion request.  Using a disk here
    # would penalize a correct boundary merely for passing near the click.
    violations += sum(bool(mask[y, x]) for x, y in background_pixels)
    if not np.any(mask):
        violations += 1
    return int(violations)


def _nearest_foreground_pixel(
    mask: np.ndarray,
    point: PixelPoint,
    radius: int,
) -> PixelPoint | None:
    x, y = point
    if mask[y, x]:
        return x, y
    if radius <= 0:
        return None
    height, width = mask.shape
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if xs.size == 0:
        return None
    xs = xs.astype(np.int64) + x0
    ys = ys.astype(np.int64) + y0
    squared_distance = (xs - x) ** 2 + (ys - y) ** 2
    inside = squared_distance <= radius * radius
    if not np.any(inside):
        return None
    valid_indices = np.flatnonzero(inside)
    nearest_index = int(valid_indices[np.argmin(squared_distance[valid_indices])])
    return int(xs[nearest_index]), int(ys[nearest_index])


def _prompted_component_stats(
    mask: np.ndarray,
    foreground_pixels: Sequence[PixelPoint],
    search_radius: int,
) -> tuple[int, bool]:
    anchor = None
    for point in foreground_pixels:
        anchor = _nearest_foreground_pixel(mask, point, search_radius)
        if anchor is not None:
            break
    if anchor is None:
        return 0, False
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return 0, False
    label = int(labels[anchor[1], anchor[0]])
    if label <= 0:
        return 0, False
    touches_border = bool(
        np.any(labels[0, :] == label)
        or np.any(labels[-1, :] == label)
        or np.any(labels[:, 0] == label)
        or np.any(labels[:, -1] == label)
    )
    return int(stats[label, cv2.CC_STAT_AREA]), touches_border


def select_best_sam2_candidate(
    output: Sam2RawOutput,
    foreground_pixels: Sequence[PixelPoint],
    background_pixels: Sequence[PixelPoint] = (),
    *,
    search_radius: int = 0,
    iou_score_window: float = 0.06,
) -> Sam2CandidateSelection:
    """Prefer complete, non-border masks within a near-best IoU window.

    SAM2's highest predicted-IoU candidate can be the tight interpretation of
    a click and omit distant parts of one connected object.  We first minimize
    prompt violations, then prefer candidates that do not leak to an image
    border.  Among candidates whose score is within ``iou_score_window`` of
    the best remaining score, the largest prompted connected component wins.
    """

    if not isinstance(output, Sam2RawOutput):
        raise TypeError("output must be a Sam2RawOutput")
    if not foreground_pixels:
        raise ValueError("at least one foreground point is required")
    if isinstance(search_radius, bool) or not isinstance(search_radius, (int, np.integer)):
        raise TypeError("search_radius must be an integer")
    search_radius = int(search_radius)
    if search_radius < 0:
        raise ValueError("search_radius must be non-negative")
    iou_score_window = float(iou_score_window)
    if not math.isfinite(iou_score_window) or not 0.0 <= iou_score_window <= 1.0:
        raise ValueError("iou_score_window must be within [0, 1]")

    height, width = output.candidate_masks.shape[1:]
    foreground_pixels = _validated_pixel_points(
        foreground_pixels,
        width,
        height,
        "foreground_pixels",
    )
    background_pixels = _validated_pixel_points(
        background_pixels,
        width,
        height,
        "background_pixels",
    )

    records: list[tuple[int, float, bool, int]] = []
    for index, mask in enumerate(output.candidate_masks):
        violations = _prompt_violation_count(
            mask,
            foreground_pixels,
            background_pixels,
            search_radius,
        )
        area, touches_border = _prompted_component_stats(
            mask,
            foreground_pixels,
            search_radius,
        )
        records.append(
            (violations, float(output.iou_scores[index]), touches_border, area)
        )

    minimum_violations = min(record[0] for record in records)
    eligible = [
        index for index, record in enumerate(records) if record[0] == minimum_violations
    ]
    non_border = [index for index in eligible if not records[index][2]]
    if non_border:
        eligible = non_border
    highest_iou = max(records[index][1] for index in eligible)
    near_best = [
        index
        for index in eligible
        if records[index][1] >= highest_iou - iou_score_window
    ]
    candidate_index = max(
        near_best,
        key=lambda index: (records[index][3], records[index][1], -index),
    )
    violations, predicted_iou, touches_border, area = records[candidate_index]
    return Sam2CandidateSelection(
        mask=output.candidate_masks[candidate_index],
        candidate_index=candidate_index,
        predicted_iou=predicted_iou,
        prompt_violation_count=violations,
        main_component_area=area,
        touches_image_border=touches_border,
    )


def keep_main_prompted_component(
    mask: np.ndarray,
    foreground_pixels: Sequence[PixelPoint],
    *,
    search_radius: int = 0,
) -> np.ndarray:
    """Keep only the component selected by the first usable positive point."""

    binary = np.ascontiguousarray(np.asarray(mask) != 0, dtype=np.uint8) * 255
    if binary.ndim != 2 or not binary.size:
        raise ValueError("mask must be a non-empty two-dimensional array")
    if not foreground_pixels:
        raise ValueError("at least one foreground point is required")
    if isinstance(search_radius, bool) or not isinstance(search_radius, (int, np.integer)):
        raise TypeError("search_radius must be an integer")
    search_radius = int(search_radius)
    if search_radius < 0:
        raise ValueError("search_radius must be non-negative")
    height, width = binary.shape
    foreground_pixels = _validated_pixel_points(
        foreground_pixels,
        width,
        height,
        "foreground_pixels",
    )
    boolean_mask = binary != 0
    anchor = None
    for point in foreground_pixels:
        anchor = _nearest_foreground_pixel(boolean_mask, point, search_radius)
        if anchor is not None:
            break
    if anchor is None:
        raise Sam2NoMaskError("no SAM2 candidate contains the foreground prompt")
    main = keep_prompt_components(binary, (anchor,))
    if not np.any(main):
        raise Sam2NoMaskError("the selected SAM2 component is empty")
    return main


def smooth_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Round pixel-scale boundary serrations while keeping a binary mask."""

    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)):
        raise TypeError("radius must be an integer")
    radius = int(radius)
    if not 0 <= radius <= 32:
        raise ValueError("radius must be within [0, 32]")
    binary = np.asarray(mask)
    if binary.ndim != 2 or not binary.size:
        raise ValueError("mask must be a non-empty two-dimensional array")
    if not (
        binary.dtype == np.bool_
        or np.issubdtype(binary.dtype, np.integer)
        or np.issubdtype(binary.dtype, np.floating)
    ):
        raise TypeError("mask must contain boolean or numeric values")
    if np.issubdtype(binary.dtype, np.floating) and not np.all(np.isfinite(binary)):
        raise ValueError("mask must contain finite values")
    binary = np.ascontiguousarray(binary != 0, dtype=np.uint8) * 255
    if radius == 0:
        return binary
    kernel_size = radius * 2 + 1
    blurred = cv2.GaussianBlur(
        binary,
        (kernel_size, kernel_size),
        sigmaX=max(0.8, radius * 0.55),
        sigmaY=max(0.8, radius * 0.55),
        borderType=cv2.BORDER_REPLICATE,
    )
    return np.where(blurred >= 128, 255, 0).astype(np.uint8)


def extract_curvature_adaptive_contour_hierarchy(
    mask: np.ndarray,
) -> ContourHierarchy:
    """Keep sparse straight runs and denser samples where curvature changes.

    The canonical binary mask is not modified.  Teh-Chin's curvature-aware
    chain approximation only creates a lighter contour representation for
    drawing and coordinate processing while retaining the contour tree.
    """

    binary = np.asarray(mask)
    if binary.ndim != 2 or not binary.size:
        raise ValueError("mask must be a non-empty two-dimensional array")
    if not (
        binary.dtype == np.bool_
        or np.issubdtype(binary.dtype, np.integer)
        or np.issubdtype(binary.dtype, np.floating)
    ):
        raise TypeError("mask must contain boolean or numeric values")
    if np.issubdtype(binary.dtype, np.floating) and not np.all(np.isfinite(binary)):
        raise ValueError("mask must contain finite values")
    binary = np.ascontiguousarray(binary != 0, dtype=np.uint8) * 255
    found_contours, found_hierarchy = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_TC89_KCOS,
    )
    contours = tuple(found_contours)
    if found_hierarchy is None or not contours:
        return ContourHierarchy(
            contours=(),
            hierarchy=np.empty((1, 0, 4), dtype=np.int32),
            outer_index=None,
            hole_indices=(),
        )

    hierarchy = np.ascontiguousarray(found_hierarchy, dtype=np.int32)

    def contour_depth(index: int) -> int:
        depth = 0
        visited = {index}
        parent = int(hierarchy[0, index, 3])
        while parent >= 0:
            if parent in visited:
                raise ValueError("invalid cyclic contour hierarchy")
            visited.add(parent)
            depth += 1
            parent = int(hierarchy[0, parent, 3])
        return depth

    top_level = [
        index
        for index in range(len(contours))
        if int(hierarchy[0, index, 3]) == -1
    ]
    outer_index = (
        max(top_level, key=lambda index: abs(cv2.contourArea(contours[index])))
        if top_level
        else None
    )
    hole_indices = tuple(
        index
        for index in range(len(contours))
        if contour_depth(index) % 2 == 1
    )
    return ContourHierarchy(
        contours=contours,
        hierarchy=hierarchy,
        outer_index=outer_index,
        hole_indices=hole_indices,
    )


def finalize_sam2_mask(
    output: Sam2RawOutput,
    encoding: Sam2PromptEncoding,
    *,
    prompt_search_radius_ratio: float,
    iou_score_window: float = 0.06,
    smart_boundary_smoothing: bool = True,
    boundary_smoothing_radius: int = 4,
) -> tuple[np.ndarray, ContourHierarchy, Sam2CandidateSelection]:
    """Select a candidate and optionally smooth/sample its outer boundary."""

    ratio = float(prompt_search_radius_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 0.05:
        raise ValueError("prompt_search_radius_ratio must be within [0, 0.05]")
    height, width = output.candidate_masks.shape[1:]
    search_radius = int(round(min(width, height) * ratio))
    selection = select_best_sam2_candidate(
        output,
        encoding.foreground_pixels,
        encoding.background_pixels,
        search_radius=search_radius,
        iou_score_window=iou_score_window,
    )
    main_mask = keep_main_prompted_component(
        selection.mask,
        encoding.foreground_pixels,
        search_radius=search_radius,
    )
    if not isinstance(smart_boundary_smoothing, bool):
        raise TypeError("smart_boundary_smoothing must be a bool")
    if smart_boundary_smoothing:
        contour_mask = main_mask
        smoothed = smooth_binary_mask(main_mask, boundary_smoothing_radius)
        try:
            smoothed_main = keep_main_prompted_component(
                smoothed,
                encoding.foreground_pixels,
                search_radius=max(search_radius, int(boundary_smoothing_radius)),
            )
        except Sam2NoMaskError:
            smoothed_main = main_mask
        violates_background = any(
            smoothed_main[y, x] != 0 for x, y in encoding.background_pixels
        )
        if not violates_background:
            contour_mask = smoothed_main
        hierarchy = extract_curvature_adaptive_contour_hierarchy(contour_mask)
    else:
        hierarchy = extract_contour_hierarchy(main_mask)
    if hierarchy.outer_index is None:
        raise Sam2NoMaskError("the selected SAM2 mask has no outer contour")
    return main_mask, hierarchy, selection


__all__ = [
    "Sam2CandidateSelection",
    "Sam2PromptEncoding",
    "encode_sam2_prompts",
    "extract_curvature_adaptive_contour_hierarchy",
    "finalize_sam2_mask",
    "keep_main_prompted_component",
    "select_best_sam2_candidate",
    "smooth_binary_mask",
]
