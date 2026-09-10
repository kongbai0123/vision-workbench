"""A bounded-latency, latest-frame-only segmentation worker."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Optional, Tuple

import numpy as np

from .models import SegmentationPrompts, SegmentationResult, _validated_integer
from .segmenter import ClassicalSegmenter


@dataclass(frozen=True, eq=False)
class SegmentationWorkerResult:
    """Outcome of one worker request; exactly one payload field is populated."""

    generation: int
    result: Optional[SegmentationResult] = None
    exception: Optional[Exception] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generation",
            _validated_integer(self.generation, "generation", allow_zero=True),
        )
        if (self.result is None) == (self.exception is None):
            raise ValueError("exactly one of result or exception must be populated")
        if self.result is not None and not isinstance(self.result, SegmentationResult):
            raise TypeError("result must be a SegmentationResult or None")
        if self.exception is not None and not isinstance(self.exception, Exception):
            raise TypeError("exception must be an Exception or None")

    @property
    def succeeded(self) -> bool:
        return self.exception is None


class SegmentationWorker:
    """Run GrabCut away from the UI thread without accumulating a FIFO queue.

    There is at most one pending input. A new submission overwrites an input
    that has not started. Completed in-flight work is still published so a
    producer that submits faster than GrabCut can run cannot starve consumers;
    consumers use ``generation`` to reject results from obsolete prompts.
    """

    def __init__(self, segmenter: ClassicalSegmenter) -> None:
        if not isinstance(segmenter, ClassicalSegmenter):
            raise TypeError("segmenter must be a ClassicalSegmenter")
        self._segmenter = segmenter
        self._condition = threading.Condition()
        self._pending: Optional[
            Tuple[int, int, np.ndarray, SegmentationPrompts]
        ] = None
        self._latest: Optional[SegmentationWorkerResult] = None
        self._submission_serial = 0
        self._stop_requested = False
        self._thread = threading.Thread(
            target=self._run,
            name="classical-segmentation",
            daemon=True,
        )
        self._thread.start()

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._thread.is_alive() and not self._stop_requested

    def submit(
        self,
        frame_bgr: np.ndarray,
        prompts: SegmentationPrompts,
        generation: int,
    ) -> bool:
        """Submit a snapshot, replacing any older frame that is still pending."""

        if not isinstance(prompts, SegmentationPrompts):
            raise TypeError("prompts must be a SegmentationPrompts")
        generation = _validated_integer(generation, "generation", allow_zero=True)
        frame = np.asarray(frame_bgr)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("frame_bgr must be a non-empty HxWx3 uint8 BGR image")
        # Camera backends often reuse frame buffers. Copy at the ownership
        # boundary so the worker always processes the submitted image.
        frame_snapshot = np.ascontiguousarray(frame).copy()
        with self._condition:
            if self._stop_requested:
                return False
            self._submission_serial += 1
            serial = self._submission_serial
            self._pending = (serial, generation, frame_snapshot, prompts)
            self._condition.notify()
            return True

    def poll_latest(self) -> Optional[SegmentationWorkerResult]:
        """Return and consume the newest completed result, if one is available."""

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
            self._condition.notify_all()
            thread = self._thread
        if thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return not thread.is_alive()

    def _run(self) -> None:
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
                result = self._segmenter.segment(frame_bgr, prompts)
                outcome = SegmentationWorkerResult(
                    generation=generation,
                    result=result,
                )
            except Exception as exc:
                outcome = SegmentationWorkerResult(
                    generation=generation,
                    exception=exc,
                )

            with self._condition:
                if self._stop_requested:
                    return
                self._latest = outcome
