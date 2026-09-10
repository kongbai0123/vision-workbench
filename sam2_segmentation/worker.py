"""Latest-frame-only background worker for lazy SAM2 inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Callable, Optional, Tuple

import numpy as np

from classical_segmentation import SegmentationPrompts

from .backend import Sam2Segmenter
from .models import Sam2Config, Sam2SegmentationResult, Sam2WorkerState


SegmenterFactory = Callable[[], Sam2Segmenter]


def _validate_generation(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("generation must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError("generation must be non-negative")
    return value


@dataclass(frozen=True, eq=False)
class Sam2WorkerResult:
    """Outcome of one request; exactly one payload field is populated."""

    generation: int
    result: Optional[Sam2SegmentationResult] = None
    exception: Optional[Exception] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", _validate_generation(self.generation))
        if (self.result is None) == (self.exception is None):
            raise ValueError("exactly one of result or exception must be populated")
        if self.result is not None and not isinstance(self.result, Sam2SegmentationResult):
            raise TypeError("result must be a Sam2SegmentationResult or None")
        if self.exception is not None and not isinstance(self.exception, Exception):
            raise TypeError("exception must be an Exception or None")

    @property
    def succeeded(self) -> bool:
        return self.exception is None


class Sam2Worker:
    """Run SAM2 off the Tk thread without accumulating stale camera frames.

    Neither the segmenter factory nor its lazy model loader is invoked during
    construction.  The first accepted request creates the segmenter on the
    worker thread.  While inference is active, each new submission replaces
    the one pending frame.
    """

    def __init__(
        self,
        config: Optional[Sam2Config] = None,
        *,
        segmenter_factory: Optional[SegmenterFactory] = None,
    ) -> None:
        parsed_config = config or Sam2Config()
        if not isinstance(parsed_config, Sam2Config):
            raise TypeError("config must be a Sam2Config or None")
        if segmenter_factory is None:
            segmenter_factory = lambda: Sam2Segmenter(parsed_config)
        if not callable(segmenter_factory):
            raise TypeError("segmenter_factory must be callable")

        self.config = parsed_config
        self._segmenter_factory = segmenter_factory
        self._segmenter: Optional[Sam2Segmenter] = None
        self._condition = threading.Condition()
        self._pending: Optional[
            Tuple[int, int, np.ndarray, SegmentationPrompts]
        ] = None
        self._latest: Optional[Sam2WorkerResult] = None
        self._submission_serial = 0
        self._stop_requested = False
        self._state = Sam2WorkerState.IDLE
        self._thread = threading.Thread(
            target=self._run,
            name="sam2-segmentation",
            daemon=True,
        )
        self._thread.start()

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._thread.is_alive() and not self._stop_requested

    @property
    def state(self) -> Sam2WorkerState:
        with self._condition:
            return self._state

    def submit(
        self,
        frame_bgr: np.ndarray,
        prompts: SegmentationPrompts,
        generation: int,
    ) -> bool:
        """Submit a copied frame, replacing an older request not yet started."""

        if not isinstance(prompts, SegmentationPrompts):
            raise TypeError("prompts must be a SegmentationPrompts")
        generation = _validate_generation(generation)
        frame = np.asarray(frame_bgr)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("frame_bgr must be a non-empty HxWx3 uint8 BGR image")
        snapshot = np.ascontiguousarray(frame).copy()
        with self._condition:
            if self._stop_requested:
                return False
            self._submission_serial += 1
            self._pending = (
                self._submission_serial,
                generation,
                snapshot,
                prompts,
            )
            self._condition.notify()
            return True

    def poll_latest(self) -> Optional[Sam2WorkerResult]:
        """Return and consume the newest completed outcome, if available."""

        with self._condition:
            latest = self._latest
            self._latest = None
            return latest

    def close(self, timeout: float = 1.0) -> bool:
        """Request shutdown and return whether the daemon stopped in time."""

        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("timeout must be finite and non-negative")
        with self._condition:
            self._stop_requested = True
            self._pending = None
            self._latest = None
            self._state = Sam2WorkerState.CLOSED
            self._condition.notify_all()
            thread = self._thread
        if thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._stop_requested:
                        self._condition.wait()
                    if self._stop_requested:
                        return
                    work = self._pending
                    self._pending = None

                assert work is not None
                _serial, generation, frame_bgr, prompts = work
                try:
                    if self._segmenter is None:
                        with self._condition:
                            self._state = Sam2WorkerState.LOADING
                        self._segmenter = self._segmenter_factory()
                    if getattr(self._segmenter, "is_loaded", False):
                        with self._condition:
                            self._state = Sam2WorkerState.PROCESSING
                    result = self._segmenter.segment(frame_bgr, prompts)
                    outcome = Sam2WorkerResult(generation=generation, result=result)
                    completed_state = Sam2WorkerState.READY
                except Exception as exc:
                    outcome = Sam2WorkerResult(generation=generation, exception=exc)
                    completed_state = Sam2WorkerState.FAILED

                with self._condition:
                    if self._stop_requested:
                        return
                    self._latest = outcome
                    self._state = completed_state
        finally:
            segmenter = self._segmenter
            if segmenter is not None:
                try:
                    segmenter.close()
                except Exception:
                    pass
            with self._condition:
                self._state = Sam2WorkerState.CLOSED


__all__ = ["Sam2Worker", "Sam2WorkerResult", "SegmenterFactory"]
