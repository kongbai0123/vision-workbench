"""Robust empty-workspace background calibration."""

from __future__ import annotations

import threading
from typing import List, Optional

import cv2
import numpy as np

from .models import BackgroundModel, SegmentationConfig, Size


def _validate_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_bgr)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame_bgr must be a non-empty HxWx3 uint8 BGR image")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("frame_bgr must be a non-empty HxWx3 uint8 BGR image")
    return frame


def _fit_working_size(source_size: Size, max_dimension: int) -> Size:
    """Fit a source size inside a square bound without ever upscaling it."""

    source_width, source_height = source_size
    largest = max(source_width, source_height)
    if largest <= max_dimension:
        return source_width, source_height
    scale = float(max_dimension) / float(largest)
    return (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )


def _resize_to(frame_bgr: np.ndarray, size: Size) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    if (width, height) == size:
        return frame_bgr
    return cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)


class BackgroundCalibrator:
    """Collect empty-scene frames and build a per-pixel median/MAD model.

    All samples must have the same source resolution. Samples are immediately
    resized and converted to Lab, so retained calibration memory is bounded by
    ``background_sample_count`` and ``max_working_dimension``.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None) -> None:
        self.config = config or SegmentationConfig()
        if not isinstance(self.config, SegmentationConfig):
            raise TypeError("config must be a SegmentationConfig")
        self._lock = threading.RLock()
        self._samples_lab: List[np.ndarray] = []
        self._source_size: Optional[Size] = None
        self._working_size: Optional[Size] = None

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples_lab)

    @property
    def target_sample_count(self) -> int:
        return self.config.background_sample_count

    @property
    def progress(self) -> float:
        with self._lock:
            return min(1.0, len(self._samples_lab) / self.config.background_sample_count)

    @property
    def ready(self) -> bool:
        with self._lock:
            return len(self._samples_lab) >= self.config.background_sample_count

    @property
    def source_size(self) -> Optional[Size]:
        with self._lock:
            return self._source_size

    @property
    def working_size(self) -> Optional[Size]:
        with self._lock:
            return self._working_size

    def add(self, frame_bgr: np.ndarray) -> float:
        """Add one empty-scene frame and return calibration progress in [0, 1]."""

        frame = _validate_bgr_frame(frame_bgr)
        source_size = (int(frame.shape[1]), int(frame.shape[0]))
        with self._lock:
            if self._source_size is None:
                self._source_size = source_size
                self._working_size = _fit_working_size(
                    source_size,
                    self.config.max_working_dimension,
                )
            elif source_size != self._source_size:
                raise ValueError(
                    "all background samples must have the same source size; "
                    f"expected {self._source_size}, received {source_size}"
                )

            if len(self._samples_lab) >= self.config.background_sample_count:
                return 1.0

            assert self._working_size is not None
            working = _resize_to(frame, self._working_size)
            lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
            self._samples_lab.append(np.ascontiguousarray(lab))
            return min(1.0, len(self._samples_lab) / self.config.background_sample_count)

    def add_frame(self, frame_bgr: np.ndarray) -> float:
        """Named alias for callers that prefer an explicit frame operation."""

        return self.add(frame_bgr)

    def build(self) -> BackgroundModel:
        """Build the robust background model after the requested sample count."""

        with self._lock:
            if len(self._samples_lab) < self.config.background_sample_count:
                raise RuntimeError(
                    "background calibration is not ready: "
                    f"{len(self._samples_lab)}/{self.config.background_sample_count} samples"
                )
            samples = np.stack(self._samples_lab, axis=0)
            source_size = self._source_size
            working_size = self._working_size
            sample_count = len(self._samples_lab)

        assert source_size is not None and working_size is not None
        median_float = np.median(samples, axis=0).astype(np.float32)
        absolute_deviation = samples.astype(np.float32)
        absolute_deviation -= median_float
        np.abs(absolute_deviation, out=absolute_deviation)
        mad_lab = np.median(absolute_deviation, axis=0).astype(np.float32)
        mad_lab *= np.float32(self.config.mad_scale)
        median_lab = np.rint(median_float).clip(0, 255).astype(np.uint8)
        return BackgroundModel(
            median_lab=median_lab,
            mad_lab=mad_lab,
            source_size=source_size,
            working_size=working_size,
            sample_count=sample_count,
        )

    def reset(self) -> None:
        with self._lock:
            self._samples_lab.clear()
            self._source_size = None
            self._working_size = None
