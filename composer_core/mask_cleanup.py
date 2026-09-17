"""Small, auditable repairs for canonical binary masks."""
from __future__ import annotations

import cv2
import numpy as np


DEFAULT_TINY_HOLE_POLICY = {
    "max_hole_pixels": 16,
    "max_total_pixels": 16,
    "max_ratio": .0001,
    "max_dimension": 16,
}


def enclosed_background_components(mask: np.ndarray, *, include_points: bool = False) -> list[dict]:
    """Return background regions that polygon rasterization treats as holes.

    Foreground uses 8-connectivity elsewhere in the application. Background
    deliberately uses 4-connectivity: a one-pixel diagonal opening is still
    filled by OpenCV's outer polygon and therefore must be reported here.
    """
    binary = np.asarray(mask) != 0
    if binary.ndim != 2 or not binary.size:
        raise ValueError("mask must be a non-empty two-dimensional array")
    background = np.ascontiguousarray(~binary, dtype=np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        background, connectivity=4,
    )
    height, width = binary.shape
    holes = []
    for component in range(1, count):
        x, y, w, h, pixels = map(int, stats[component])
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        item = {"pixels": pixels, "bbox": [x, y, w, h]}
        if include_points:
            ys, xs = np.nonzero(labels[y:y + h, x:x + w] == component)
            item["points"] = [
                [int(px + x), int(py + y)] for px, py in zip(xs, ys)
            ]
        holes.append(item)
    return holes


def repair_tiny_holes(
    mask: np.ndarray,
    *,
    protected_background_points=(),
    max_hole_pixels: int = 16,
    max_total_pixels: int = 16,
    max_ratio: float = .0001,
    max_dimension: int = 16,
) -> tuple[np.ndarray, dict]:
    """Fill only bounded holes and return a reversible repair audit.

    The input is never modified. Large holes and holes containing an explicit
    background prompt are retained. If the combined eligible repair exceeds a
    total or relative limit, all eligible holes are retained for review.
    """
    binary = np.asarray(mask) != 0
    if binary.ndim != 2 or not binary.size:
        raise ValueError("mask must be a non-empty two-dimensional array")
    limits = (max_hole_pixels, max_total_pixels, max_dimension)
    if any(isinstance(value, bool) or int(value) < 0 for value in limits):
        raise ValueError("hole pixel limits must be non-negative integers")
    max_hole_pixels, max_total_pixels, max_dimension = map(int, limits)
    max_ratio = float(max_ratio)
    if not np.isfinite(max_ratio) or max_ratio < 0:
        raise ValueError("max_ratio must be a non-negative finite number")

    height, width = binary.shape
    protected = set()
    for point in protected_background_points or ():
        if (isinstance(point, (list, tuple)) and len(point) == 2
                and all(isinstance(value, (int, np.integer)) for value in point)):
            x, y = map(int, point)
            if 0 <= x < width and 0 <= y < height:
                protected.add((x, y))

    holes = enclosed_background_components(binary, include_points=True)
    eligible, retained = [], []
    for hole in holes:
        points = {tuple(point) for point in hole["points"]}
        reason = None
        if points & protected:
            reason = "protected_background_prompt"
        elif (hole["pixels"] > max_hole_pixels
                or hole["bbox"][2] > max_dimension
                or hole["bbox"][3] > max_dimension):
            reason = "above_individual_limit"
        if reason:
            retained.append({**hole, "reason": reason})
        else:
            eligible.append(hole)

    foreground = int(np.count_nonzero(binary))
    eligible_pixels = sum(item["pixels"] for item in eligible)
    ratio = eligible_pixels / max(1, foreground)
    if eligible_pixels > max_total_pixels or ratio > max_ratio:
        retained.extend({**hole, "reason": "above_combined_limit"} for hole in eligible)
        eligible = []

    repaired = binary.copy()
    for hole in eligible:
        for x, y in hole["points"]:
            repaired[y, x] = True
    public = lambda item: {key: value for key, value in item.items() if key != "points"}
    report = {
        "holes_detected": len(holes),
        "holes_filled": len(eligible),
        "pixels_filled": sum(item["pixels"] for item in eligible),
        "fill_ratio": sum(item["pixels"] for item in eligible) / max(1, foreground),
        "filled": [public(item) | {"points": item["points"]} for item in eligible],
        "retained": [public(item) for item in retained],
        "source_foreground_pixels": foreground,
        "source_mask_unchanged": not eligible,
        "reversible": True,
    }
    return np.ascontiguousarray(repaired, dtype=np.uint8) * 255, report
