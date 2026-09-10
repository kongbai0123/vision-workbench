"""Median/MAD + Lab evidence + seeded GrabCut segmentation MVP."""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .background import _resize_to, _validate_bgr_frame
from .contours import clean_prompted_mask, extract_contour_hierarchy
from .models import (
    BackgroundModel,
    NormalizedPoint,
    SegmentationConfig,
    SegmentationDiagnostics,
    SegmentationPrompts,
    SegmentationResult,
)


class ClassicalSegmenter:
    """Segment a prompted object against a calibrated empty background."""

    def __init__(
        self,
        background_model: Optional[BackgroundModel],
        config: Optional[SegmentationConfig] = None,
    ) -> None:
        if background_model is not None and not isinstance(background_model, BackgroundModel):
            raise TypeError("background_model must be a BackgroundModel or None")
        if config is not None and not isinstance(config, SegmentationConfig):
            raise TypeError("config must be a SegmentationConfig or None")
        self.background_model = background_model
        self.config = config or SegmentationConfig()

    @staticmethod
    def _point_pixels(
        points: Sequence[NormalizedPoint],
        width: int,
        height: int,
    ) -> Tuple[Tuple[int, int], ...]:
        return tuple(point.to_pixel(width, height) for point in points)

    def _robust_lab_difference(self, working_bgr: np.ndarray) -> np.ndarray:
        assert self.background_model is not None
        current_lab = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        median_lab = self.background_model.median_lab.astype(np.float32)
        denominator = np.maximum(self.background_model.mad_lab, self.config.mad_floor)
        standardized = np.abs(current_lab - median_lab) / denominator
        # RMS prevents a one-channel change from being diluted as heavily as a
        # simple mean while retaining a stable scale across all Lab channels.
        difference = np.sqrt(np.mean(np.square(standardized), axis=2))
        return np.ascontiguousarray(difference, dtype=np.float32)

    def _initial_trimap(
        self,
        difference: np.ndarray,
        prompts: SegmentationPrompts,
    ) -> Tuple[np.ndarray, Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]:
        height, width = difference.shape
        config = self.config

        # Use the two thresholds as hysteresis: a low-threshold changed region
        # becomes probable foreground only when it is connected to high-
        # confidence evidence. Everything else remains *probable* background,
        # rather than hard background, so GrabCut may recover a weak boundary.
        low_evidence = (difference >= config.lab_diff_low_threshold).astype(np.uint8)
        high_evidence = difference >= config.lab_diff_high_threshold
        component_count, component_labels = cv2.connectedComponents(
            low_evidence,
            connectivity=8,
        )
        strong_labels = np.unique(component_labels[high_evidence])
        strong_labels = strong_labels[strong_labels != 0]
        evidence_foreground = np.zeros_like(low_evidence, dtype=bool)
        if component_count > 1 and strong_labels.size:
            evidence_foreground = np.isin(component_labels, strong_labels)
        trimap = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
        trimap[evidence_foreground] = cv2.GC_PR_FGD

        foreground_pixels = self._point_pixels(prompts.foreground_points, width, height)
        background_pixels = self._point_pixels(prompts.background_points, width, height)

        if prompts.roi is not None:
            x0, y0, x1, y1 = prompts.roi.to_pixels(width, height)
            if any(not (x0 <= x < x1 and y0 <= y < y1) for x, y in foreground_pixels):
                raise ValueError("every foreground point must lie inside the ROI")

        seed_radius = max(1, int(round(min(width, height) * config.seed_radius_ratio)))
        minimum_separation_squared = float((seed_radius * 2) ** 2)
        for foreground_x, foreground_y in foreground_pixels:
            for background_x, background_y in background_pixels:
                distance_squared = float(
                    (foreground_x - background_x) ** 2
                    + (foreground_y - background_y) ** 2
                )
                if distance_squared <= minimum_separation_squared:
                    raise ValueError(
                        "foreground and background seed regions must not overlap"
                    )

        for point in background_pixels:
            cv2.circle(trimap, point, seed_radius, int(cv2.GC_BGD), thickness=-1)
        for point in foreground_pixels:
            cv2.circle(trimap, point, seed_radius, int(cv2.GC_FGD), thickness=-1)

        # Spatial constraints are applied after seed disks so ROI and border
        # background remain genuinely hard constraints rather than suggestions.
        border = 0
        if config.border_background_ratio > 0.0:
            border = max(
                1,
                int(round(min(width, height) * config.border_background_ratio)),
            )
            trimap[:border, :] = cv2.GC_BGD
            trimap[-border:, :] = cv2.GC_BGD
            trimap[:, :border] = cv2.GC_BGD
            trimap[:, -border:] = cv2.GC_BGD
        if prompts.roi is not None:
            x0, y0, x1, y1 = prompts.roi.to_pixels(width, height)
            outside_roi = np.ones((height, width), dtype=bool)
            outside_roi[y0:y1, x0:x1] = False
            trimap[outside_roi] = cv2.GC_BGD

        if any(trimap[y, x] != cv2.GC_FGD for x, y in foreground_pixels):
            raise ValueError(
                "a foreground point lies in the hard background border or outside the ROI"
            )
        if any(trimap[y, x] != cv2.GC_BGD for x, y in background_pixels):
            raise ValueError("a background point was overwritten by a foreground seed")

        if not np.any(trimap == cv2.GC_BGD):
            raise ValueError("GrabCut requires at least one definite background pixel")
        if not np.any(trimap == cv2.GC_FGD):
            raise ValueError("GrabCut requires at least one definite foreground pixel")
        return trimap, foreground_pixels, background_pixels

    def _enforce_binary_prompts(
        self,
        mask: np.ndarray,
        prompts: SegmentationPrompts,
        foreground_pixels: Sequence[Tuple[int, int]],
        background_pixels: Sequence[Tuple[int, int]],
    ) -> np.ndarray:
        """Reapply hard prompts and spatial limits after unconstrained morphology."""

        output = np.ascontiguousarray(mask, dtype=np.uint8).copy()
        height, width = output.shape
        seed_radius = max(
            1,
            int(round(min(width, height) * self.config.seed_radius_ratio)),
        )
        for point in foreground_pixels:
            cv2.circle(output, point, seed_radius, 255, thickness=-1)
        for point in background_pixels:
            cv2.circle(output, point, seed_radius, 0, thickness=-1)

        if self.config.border_background_ratio > 0.0:
            border = max(
                1,
                int(round(min(width, height) * self.config.border_background_ratio)),
            )
            output[:border, :] = 0
            output[-border:, :] = 0
            output[:, :border] = 0
            output[:, -border:] = 0
        if prompts.roi is not None:
            x0, y0, x1, y1 = prompts.roi.to_pixels(width, height)
            constrained = np.zeros_like(output)
            constrained[y0:y1, x0:x1] = output[y0:y1, x0:x1]
            output = constrained
        return output

    def _morphology(self, mask: np.ndarray) -> np.ndarray:
        radius = int(round(min(mask.shape) * self.config.morphology_radius_ratio))
        if radius <= 0:
            return mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    def segment(
        self,
        frame_bgr: np.ndarray,
        prompts: SegmentationPrompts,
    ) -> SegmentationResult:
        """Return a full-resolution 0/255 mask for the prompted object."""

        started_at = time.perf_counter()
        if self.background_model is None:
            raise ValueError("a calibrated background model is required")
        if not isinstance(prompts, SegmentationPrompts):
            raise TypeError("prompts must be a SegmentationPrompts")
        if not prompts.foreground_points:
            raise ValueError("at least one foreground point is required")

        frame = _validate_bgr_frame(frame_bgr)
        source_size = (int(frame.shape[1]), int(frame.shape[0]))
        if source_size != self.background_model.source_size:
            raise ValueError(
                "frame size does not match the calibrated background; "
                f"expected {self.background_model.source_size}, received {source_size}"
            )

        working_size = self.background_model.working_size
        working_bgr = _resize_to(frame, working_size)
        difference = self._robust_lab_difference(working_bgr)
        trimap, working_foreground_points, working_background_points = self._initial_trimap(
            difference,
            prompts,
        )
        probable_foreground_ratio = float(np.count_nonzero(trimap == cv2.GC_PR_FGD)) / float(
            difference.size
        )

        background_distribution = np.zeros((1, 65), dtype=np.float64)
        foreground_distribution = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(
                working_bgr,
                trimap,
                None,
                background_distribution,
                foreground_distribution,
                self.config.grabcut_iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error as exc:
            raise RuntimeError(f"GrabCut failed: {exc}") from exc

        working_mask = np.where(
            (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        working_mask = self._morphology(working_mask)
        working_mask = self._enforce_binary_prompts(
            working_mask,
            prompts,
            working_foreground_points,
            working_background_points,
        )
        working_mask = clean_prompted_mask(
            working_mask,
            working_foreground_points,
            min_component_area_ratio=self.config.min_component_area_ratio,
            max_hole_area_ratio=self.config.max_hole_area_ratio,
        )
        working_mask = self._enforce_binary_prompts(
            working_mask,
            prompts,
            working_foreground_points,
            working_background_points,
        )

        if working_size == source_size:
            source_mask = working_mask.copy()
        else:
            source_mask = cv2.resize(
                working_mask,
                source_size,
                interpolation=cv2.INTER_NEAREST,
            )
        source_foreground_points = self._point_pixels(
            prompts.foreground_points,
            source_size[0],
            source_size[1],
        )
        source_background_points = self._point_pixels(
            prompts.background_points,
            source_size[0],
            source_size[1],
        )
        source_mask = self._enforce_binary_prompts(
            source_mask,
            prompts,
            source_foreground_points,
            source_background_points,
        )
        source_mask = clean_prompted_mask(
            source_mask,
            source_foreground_points,
            min_component_area_ratio=self.config.min_component_area_ratio,
            max_hole_area_ratio=self.config.max_hole_area_ratio,
        )
        source_mask = self._enforce_binary_prompts(
            source_mask,
            prompts,
            source_foreground_points,
            source_background_points,
        )
        contour_hierarchy = extract_contour_hierarchy(source_mask)

        foreground_ratio = float(np.count_nonzero(source_mask)) / float(source_mask.size)
        diagnostics = SegmentationDiagnostics(
            source_size=source_size,
            working_size=working_size,
            lab_diff_low_threshold=self.config.lab_diff_low_threshold,
            lab_diff_high_threshold=self.config.lab_diff_high_threshold,
            difference_min=float(np.min(difference)),
            difference_mean=float(np.mean(difference, dtype=np.float64)),
            difference_max=float(np.max(difference)),
            probable_foreground_ratio=probable_foreground_ratio,
            foreground_ratio=foreground_ratio,
            foreground_point_count=len(prompts.foreground_points),
            background_point_count=len(prompts.background_points),
            grabcut_iterations=self.config.grabcut_iterations,
            processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
        )
        return SegmentationResult(
            mask=source_mask,
            hierarchy=contour_hierarchy,
            diagnostics=diagnostics,
        )
