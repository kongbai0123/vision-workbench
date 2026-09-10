"""Binary-mask cleanup and topology-preserving contour extraction."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

from .models import ContourHierarchy, PixelPoint


def as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a contiguous uint8 mask whose only values are 0 and 255."""

    array = np.asarray(mask)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("mask must be a non-empty two-dimensional array")
    return np.ascontiguousarray(np.where(array != 0, 255, 0).astype(np.uint8))


def _validated_points(
    points: Iterable[PixelPoint],
    width: int,
    height: int,
) -> Tuple[PixelPoint, ...]:
    validated = []
    for point in points:
        if len(point) != 2:
            raise ValueError("pixel points must contain exactly two coordinates")
        x, y = int(point[0]), int(point[1])
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(f"pixel point {(x, y)} lies outside a {width}x{height} mask")
        validated.append((x, y))
    return tuple(validated)


def remove_small_components(
    mask: np.ndarray,
    min_area_ratio: float,
    protected_points: Sequence[PixelPoint] = (),
) -> np.ndarray:
    """Remove small foreground islands, except components touched by prompts."""

    binary = as_binary_mask(mask)
    ratio = float(min_area_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("min_area_ratio must be within [0, 1]")
    height, width = binary.shape
    points = _validated_points(protected_points, width, height)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (binary != 0).astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return np.zeros_like(binary)

    protected_labels = {int(labels[y, x]) for x, y in points if labels[y, x] > 0}
    minimum_area = int(math.ceil(binary.size * ratio)) if ratio > 0.0 else 0
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        keep[label] = area >= minimum_area or label in protected_labels
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def keep_prompt_components(
    mask: np.ndarray,
    foreground_points: Sequence[PixelPoint],
) -> np.ndarray:
    """Keep only connected foreground components containing a positive point."""

    binary = as_binary_mask(mask)
    height, width = binary.shape
    points = _validated_points(foreground_points, width, height)
    if not points:
        raise ValueError("at least one foreground point is required")
    count, labels = cv2.connectedComponents((binary != 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(binary)
    selected_labels = {int(labels[y, x]) for x, y in points if labels[y, x] > 0}
    if not selected_labels:
        return np.zeros_like(binary)
    lookup = np.zeros(count, dtype=bool)
    lookup[list(selected_labels)] = True
    return np.where(lookup[labels], 255, 0).astype(np.uint8)


def fill_small_holes(mask: np.ndarray, max_hole_area_ratio: float) -> np.ndarray:
    """Fill only enclosed background regions below a proportional area limit."""

    binary = as_binary_mask(mask)
    ratio = float(max_hole_area_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("max_hole_area_ratio must be within [0, 1]")
    if ratio == 0.0:
        return binary

    background = (binary == 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        background,
        # Pair 8-connected foreground with 4-connected background. Using 8
        # for both makes a diagonally enclosed pixel simultaneously look like
        # a RETR_TREE hole and like exterior-connected background.
        connectivity=4,
    )
    if count <= 1:
        return binary

    border_labels = set(int(label) for label in labels[0, :])
    border_labels.update(int(label) for label in labels[-1, :])
    border_labels.update(int(label) for label in labels[:, 0])
    border_labels.update(int(label) for label in labels[:, -1])
    maximum_area = int(math.floor(binary.size * ratio))
    output = binary.copy()
    for label in range(1, count):
        if label in border_labels:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= maximum_area:
            output[labels == label] = 255
    return output


def clean_prompted_mask(
    mask: np.ndarray,
    foreground_points: Sequence[PixelPoint],
    *,
    min_component_area_ratio: float,
    max_hole_area_ratio: float,
) -> np.ndarray:
    """Apply component and small-hole policy without destroying real holes."""

    cleaned = remove_small_components(
        mask,
        min_component_area_ratio,
        protected_points=foreground_points,
    )
    cleaned = keep_prompt_components(cleaned, foreground_points)
    return fill_small_holes(cleaned, max_hole_area_ratio)


def _contour_depth(index: int, hierarchy: np.ndarray) -> int:
    depth = 0
    parent = int(hierarchy[0, index, 3])
    visited = {index}
    while parent >= 0:
        if parent in visited:
            raise ValueError("invalid cyclic contour hierarchy")
        visited.add(parent)
        depth += 1
        parent = int(hierarchy[0, parent, 3])
    return depth


def extract_contour_hierarchy(mask: np.ndarray) -> ContourHierarchy:
    """Extract an unfiltered RETR_TREE contour hierarchy from a binary mask.

    No post-extraction contour filtering is performed because removing contour
    nodes without remapping OpenCV's indices corrupts parent/child topology.
    Small components and holes must instead be removed from the binary mask
    before this function is called.
    """

    binary = as_binary_mask(mask)
    found_contours, found_hierarchy = cv2.findContours(
        binary.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    contours = tuple(found_contours)
    if found_hierarchy is None or not contours:
        hierarchy = np.empty((1, 0, 4), dtype=np.int32)
        return ContourHierarchy(
            contours=(),
            hierarchy=hierarchy,
            outer_index=None,
            hole_indices=(),
        )

    hierarchy = np.ascontiguousarray(found_hierarchy, dtype=np.int32)
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
        if _contour_depth(index, hierarchy) % 2 == 1
    )
    return ContourHierarchy(
        contours=contours,
        hierarchy=hierarchy,
        outer_index=outer_index,
        hole_indices=hole_indices,
    )
