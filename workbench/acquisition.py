"""Local acquisition and prompted segmentation for Vision Workbench.

The camera and encoder belong to one worker thread. Importing this module never
opens hardware, imports Torch, starts a process, or downloads a model. Generated
annotations are candidates; review state is always controlled by the store.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.util
import io
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable
import uuid

import cv2
import numpy as np


class AcquisitionError(RuntimeError):
    """An actionable acquisition or inference failure, suitable for the UI."""


class AcquisitionCancelled(AcquisitionError):
    pass


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _identifier(prefix: str) -> str:
    return f"{prefix}_{_stamp()}_{uuid.uuid4().hex[:8]}"


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} 必須是整數")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} 必須介於 {low} 與 {high}")
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必須是數字")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必須是數字") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} 必須介於 {low} 與 {high}")
    return value


def _opencv_path(path: str | Path) -> str:
    """Use Windows short ancestors when available, retaining Unicode otherwise."""
    path = Path(path).resolve()
    if os.name != "nt":
        return str(path)
    import ctypes
    existing = path if path.exists() else path.parent
    function = ctypes.windll.kernel32.GetShortPathNameW
    required = function(str(existing), None, 0)
    if required:
        buffer = ctypes.create_unicode_buffer(required)
        if function(str(existing), buffer, required):
            return str(Path(buffer.value) / path.name) if existing != path else buffer.value
    return str(path)


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到影像：{path}")
    try:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        # Native annotations use the stored pixel matrix, not EXIF rotations.
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    except (OSError, cv2.error) as exc:
        raise AcquisitionError(f"無法讀取影像：{path.name}：{exc}") from exc
    if frame is None or frame.size == 0:
        raise AcquisitionError(f"影像格式無法解碼：{path.name}")
    return frame


def preview_image(image_path: str | Path) -> tuple[bytes, str]:
    """Serve the original pixel coordinate system without rewriting source bytes.

    Browsers automatically orient JPEGs using EXIF. An annotation image must use
    the same raw pixel matrix as the project dimensions and OpenCV inference.
    For oriented JPEGs only, decode with Pillow without exif_transpose and create
    a fresh, metadata-free PNG response. The immutable source remains untouched.
    """
    from PIL import Image, UnidentifiedImageError
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到影像：{path}")
    raw = path.read_bytes()
    try:
        with Image.open(io.BytesIO(raw)) as source:
            mime = Image.MIME.get(source.format, "application/octet-stream")
            if source.format != "JPEG" or source.getexif().get(274, 1) == 1:
                return raw, mime
            decoded = source.convert("RGB")
            # Construct a new image instead of copy(): copy/convert can preserve
            # EXIF in info and PNG's eXIf chunk would reintroduce auto-rotation.
            preview = Image.frombytes("RGB", decoded.size, decoded.tobytes())
            buffer = io.BytesIO()
            preview.save(buffer, format="PNG")
            return buffer.getvalue(), "image/png"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise AcquisitionError(f"無法產生影像預覽：{path.name}：{exc}") from exc


def _save_png(path: Path, frame: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise AcquisitionError("影像 PNG 編碼失敗")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def transform_frame(frame: np.ndarray, mode: str = "original", *, threshold: int = 127,
                    invert: bool = False, state: dict | None = None) -> np.ndarray:
    """Explicit capture preview transforms; never applied to stored source images."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("影格必須是 H×W×3 uint8 BGR")
    mode = str(mode).lower()
    if mode == "optical_flow":
        mode = "flow"
    if mode == "original":
        result = frame.copy()
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if mode == "binary":
            result = cv2.threshold(gray, _integer(threshold, "threshold", 0, 255), 255, cv2.THRESH_BINARY)[1]
        elif mode == "adaptive":
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        elif mode == "canny":
            result = cv2.Canny(gray, 80, 160)
        elif mode == "mog2":
            if state is None:
                raise ValueError("MOG2 需要持續的預覽狀態")
            if "mog2" not in state:
                state["mog2"] = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            result = state["mog2"].apply(frame)
        elif mode == "flow":
            if state is None:
                raise ValueError("Optical Flow 需要持續的預覽狀態")
            previous = state.get("gray")
            state["gray"] = gray.copy()
            result = np.zeros(gray.shape, np.uint8)
            if previous is not None and previous.shape == gray.shape:
                flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
                magnitude = np.sqrt(np.square(flow).sum(axis=2))
                result[magnitude > 1.0] = 255
        else:
            raise ValueError(f"未知影像處理模式：{mode}")
    return cv2.bitwise_not(result) if invert else result


PROCESSING_MODES = ("original", "binary", "adaptive", "canny", "mog2", "optical_flow", "classical", "sam2")


def _roi_mask(frame: np.ndarray, shape) -> np.ndarray | None:
    """Validate a workbench rectangle, polygon or uncompressed mask for preview ROI use."""
    if shape is None:
        return None
    if not isinstance(shape, dict):
        raise ValueError("目標區域必須為物件")
    height, width = frame.shape[:2]
    result = np.zeros((height, width), np.uint8)
    kind = shape.get("type", "rectangle")
    if kind == "rectangle":
        values = [_number(shape.get(key), "目標框座標", 0, width if key in {"x", "width"} else height)
                  for key in ("x", "y", "width", "height")]
        x, y, box_width, box_height = values
        if box_width <= 0 or box_height <= 0 or x + box_width > width or y + box_height > height:
            raise ValueError("目標框超出影像範圍")
        x1, y1 = int(math.floor(x)), int(math.floor(y))
        x2, y2 = int(math.ceil(x + box_width)), int(math.ceil(y + box_height))
        result[y1:y2, x1:x2] = 255
    elif kind == "polygon":
        points = shape.get("points")
        if not isinstance(points, list) or not 3 <= len(points) <= 10000:
            raise ValueError("目標 Polygon 至少需要三個頂點")
        clean = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("目標 Polygon 座標無效")
            x = _number(point[0], "Polygon X", 0, width); y = _number(point[1], "Polygon Y", 0, height)
            clean.append([round(x), round(y)])
        cv2.fillPoly(result, [np.asarray(clean, dtype=np.int32)], 255)
    elif kind == "mask":
        counts = shape.get("counts")
        if (width * height > 16777216 or not isinstance(counts, list)
                or len(counts) > width * height * 2 + 1
                or any(type(value) is not int or value < 0 for value in counts)
                or sum(counts) != width * height):
            raise ValueError("目標 Mask 與目前影像尺寸不一致")
        flat = np.zeros(width * height, np.uint8); offset = 0; active = False
        for count in counts:
            if active:
                flat[offset:offset + count] = 255
            offset += count; active = not active
        result = flat.reshape((width, height)).T.copy()
    else:
        raise ValueError("目標區域只支援 Bounding Box、Polygon 或 Mask")
    if not np.any(result):
        raise ValueError("目標區域不可為空")
    return result


def _roi_bounds(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _frame_prompts(frame: np.ndarray, settings: dict):
    height, width = frame.shape[:2]
    positive = _validated_points(settings.get("points"), width, height, "前景點")
    negative = _validated_points(settings.get("negative_points"), width, height, "背景點")
    if set(positive) & set(negative):
        raise ValueError("同一像素不能同時為前景點及背景點")
    box = settings.get("box")
    if box is not None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("選取框需要 [x1,y1,x2,y2]")
        raw = [_number(value, "選取框座標", 0, width if index % 2 == 0 else height)
               for index, value in enumerate(box)]
        box = (int(math.floor(raw[0])), int(math.floor(raw[1])), int(math.ceil(raw[2])), int(math.ceil(raw[3])))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("選取框需要正寬度與高度")
    if not positive and box is None:
        positive = [(width // 2, height // 2)]
    if set(positive) & set(negative):
        raise ValueError("中心已設定為背景，請另外指定前景點或選取框")
    return positive, negative, box


def process_frame(frame: np.ndarray, mode: str = "original", *, settings: dict | None = None,
                  state: dict | None = None) -> np.ndarray:
    """Seven graph algorithms, with caller-owned temporal/calibration state.

    For classical segmentation, place a calibrated ``background_model`` in state.
    ``CameraService`` builds it from 15 deliberately requested empty-scene frames.
    All results remain previews; this function never mutates or saves annotations.
    """
    settings = settings or {}
    state = state if state is not None else {}
    if mode == "flow":
        mode = "optical_flow"
    roi = _roi_mask(frame, settings.get("roi_shape"))
    prompt_settings = settings
    if roi is not None and mode in ("classical", "sam2") and settings.get("box") is None:
        prompt_settings = dict(settings, box=_roi_bounds(roi))
    if mode in ("classical", "sam2"):
        positive, negative, box = _frame_prompts(frame, prompt_settings)
        if mode == "sam2":
            output, diagnostics = _sam2(frame, positive, negative, box, settings.get("model_dir"))
            state["diagnostics"] = diagnostics
        else:
            from classical_segmentation import ClassicalSegmenter, NormalizedPoint, SegmentationPrompts
            from classical_segmentation.models import NormalizedRect
            background = state.get("background_model")
            if background is None:
                raise AcquisitionError("請先移開工件並擷取 15 張空背景；校正完成後再放回工件。")
            height, width = frame.shape[:2]
            if background.source_size != (width, height):
                state.pop("background_model", None)
                raise AcquisitionError("影像尺寸已改變，請重新擷取空背景。")
            if not positive:
                positive = [((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)]
            to_point = lambda point: NormalizedPoint(point[0] / max(1, width - 1), point[1] / max(1, height - 1))
            roi = None if box is None else NormalizedRect(box[0] / width, box[1] / height, box[2] / width, box[3] / height)
            prompts = SegmentationPrompts(tuple(map(to_point, positive)), tuple(map(to_point, negative)), roi=roi)
            result = ClassicalSegmenter(background, state.get("segmentation_config")).segment(frame, prompts)
            output = result.mask
        output = cv2.bitwise_not(output) if bool(settings.get("invert", False)) else output
    else:
        output = transform_frame(frame, mode, threshold=settings.get("threshold", 127),
                                 invert=bool(settings.get("invert", False)), state=state)
    return cv2.bitwise_and(output, output, mask=roi) if roi is not None else output


def _preview_jpeg(frame: np.ndarray) -> bytes:
    if max(frame.shape[:2]) > 1280:
        scale = 1280 / max(frame.shape[:2])
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        raise AcquisitionError("預覽影像編碼失敗")
    return encoded.tobytes()


class _PreviewProcessor:
    """Latest-frame processing worker; slow inference never blocks acquisition."""

    def __init__(self, owner):
        self.owner = owner
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="WorkbenchProcessing", daemon=True)
        self._thread.start()

    def submit(self, frame, generation, mode, settings):
        with self._condition:
            if not self._closed:
                self._pending = (frame.copy(), generation, mode, settings)
                self._condition.notify()

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=.25)

    def _publish(self, generation, *, image=None, state="ready", error=None, elapsed=0,
                 sample_count=None, diagnostics=None):
        with self.owner._lock:
            if self._closed or self.owner._processor is not self or generation != self.owner._processing_generation:
                return
            if image is not None:
                self.owner._processed_jpeg = image
            elif error:
                self.owner._processed_jpeg = b""
            self.owner._state.update(processing_state=state, processing_error=error,
                                      processing_ms=round(elapsed * 1000, 1))
            if sample_count is not None:
                self.owner._state["background_samples"] = sample_count
            if diagnostics is not None:
                self.owner._state["processing_diagnostics"] = diagnostics

    def _run(self):
        state, previous_mode, calibration_token = {}, None, None
        calibrator, last_sample = None, 0.
        retry_after, failed_generation = 0., None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                frame, generation, mode, settings = self._pending
                self._pending = None
            started = time.monotonic()
            with self.owner._lock:
                if generation != self.owner._processing_generation or self.owner._processor is not self:
                    continue
            if mode == "sam2" and failed_generation == generation and started < retry_after:
                continue
            try:
                if mode != previous_mode:
                    state.pop("gray", None)
                    state.pop("mog2", None)
                    previous_mode = mode
                token = settings.get("calibration_token")
                if mode == "classical" and token and token != calibration_token:
                    from classical_segmentation import BackgroundCalibrator, SegmentationConfig
                    calibration_token = token
                    state.pop("background_model", None)
                    state["segmentation_config"] = SegmentationConfig()
                    calibrator = BackgroundCalibrator(state["segmentation_config"])
                    last_sample = 0.
                if mode == "classical" and calibrator is not None:
                    if started - last_sample >= .075:
                        calibrator.add_frame(frame)
                        last_sample = started
                    self._publish(generation, state="calibrating", sample_count=calibrator.sample_count)
                    if calibrator.ready:
                        state["background_model"] = calibrator.build()
                        calibrator = None
                        self._publish(generation, state="background_ready", sample_count=15,
                                      elapsed=time.monotonic() - started)
                    continue
                self._publish(generation, state="loading" if mode == "sam2" else "processing")
                processed = process_frame(frame, mode, settings=settings, state=state)
                self._publish(generation, image=_preview_jpeg(processed), elapsed=time.monotonic() - started,
                              diagnostics=state.get("diagnostics"))
            except Exception as exc:
                if mode == "sam2":
                    retry_after, failed_generation = time.monotonic() + 2., generation
                self._publish(generation, state="needs_background" if mode == "classical" and "background_model" not in state else "error",
                              error=str(exc), elapsed=time.monotonic() - started)


@dataclass
class _Command:
    kind: str
    payload: dict = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class CameraService:
    """Camera lifecycle and recording, with a single native-handle owner."""

    def __init__(self, storage_root: Path):
        self.storage_root = Path(storage_root).resolve()
        self._lock = threading.RLock()
        self._control = threading.RLock()
        self._stop_event = threading.Event()
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._jpeg = b""
        self._processed_jpeg = b""
        self._processor = None
        self._processing_generation = 0
        self._processing_mode = "original"
        self._processing_settings: dict = {}
        self._timestamps: deque[float] = deque(maxlen=90)
        self._closed = False
        self._batch_id = ""
        self._state = {"state": "stopped", "running": False, "recording": False,
                       "error": None, "index": None, "width": 0, "height": 0,
                       "fps": 0.0, "measured_fps": 0.0, "frame_count": 0,
                       "record_frames": 0, "record_seconds": 0.0,
                       "last_recording": None, "processing_mode": "original",
                       "processing_state": "idle", "processing_error": None,
                       "processing_ms": 0, "background_samples": 0}

    @staticmethod
    def devices() -> list[dict]:
        if os.name != "nt":
            return []
        try:
            import comtypes
            comtypes.CoInitialize()
            try:
                from pygrabber.dshow_graph import FilterGraph
                graph = FilterGraph()
                names = list(graph.get_input_devices())
                del graph
                return [{"index": index, "name": str(name)} for index, name in enumerate(names)]
            finally:
                comtypes.CoUninitialize()
        except ImportError as exc:
            raise AcquisitionError("相機列舉需要安裝 comtypes 與 pygrabber，請執行本軟體安裝程序。") from exc
        except Exception as exc:
            raise AcquisitionError(f"無法取得 DirectShow 相機清單：{exc}") from exc

    def status(self) -> dict:
        with self._lock:
            result = dict(self._state)
            if result["last_recording"]:
                result["last_recording"] = dict(result["last_recording"])
            result["has_frame"] = self._frame is not None and result["running"]
            result["has_processed_frame"] = bool(self._processed_jpeg) and result["running"]
            result["processing_settings"] = {key: deepcopy(value) for key, value in self._processing_settings.items()
                                             if key not in {"calibration_token", "roi_shape"}}
            result["has_processing_roi"] = self._processing_settings.get("roi_shape") is not None
            result["batch_id"] = self._batch_id
            if len(self._timestamps) >= 2:
                interval = self._timestamps[-1] - self._timestamps[0]
                result["measured_fps"] = round((len(self._timestamps) - 1) / interval, 2) if interval else 0
            result["frame_age_seconds"] = round(time.monotonic() - self._frame_time, 3) if self._frame_time else None
            return result

    @staticmethod
    def capabilities(index=0):
        from .camera_modes import device_modes
        return device_modes(index)

    def start(self, index=0, width=1280, height=720, fps=30, pixel_format="MJPG") -> dict:
        if not isinstance(pixel_format,str) or len(pixel_format)!=4 or not pixel_format.isascii():
            raise ValueError("相機像素格式必須是 4 字元 FOURCC")
        config = {"index": _integer(index, "相機編號", 0, 128),
                  "width": _integer(width, "寬度", 32, 8192),
                  "height": _integer(height, "高度", 32, 8192),
                  "fps": _number(fps, "FPS", 1, 240), "pixel_format":pixel_format}
        with self._control:
            if self._closed:
                raise AcquisitionError("相機服務已關閉")
            if self._thread is not None and self._thread.is_alive():
                raise AcquisitionError("相機正在啟動或運作中；請先停止再變更來源。")
            self._stop_event = threading.Event()
            self._commands = queue.Queue()
            with self._lock:
                self._frame = None
                self._jpeg = b""
                self._processed_jpeg = b""
                self._processing_generation += 1
                self._processing_settings.pop("calibration_token", None)
                self._frame_time = 0
                self._timestamps.clear()
                self._batch_id = _identifier("camera")
                self._state.update(config, state="starting", running=False, recording=False,
                                   error=None, frame_count=0, record_frames=0, record_seconds=0,
                                   processing_state="idle", processing_error=None, background_samples=0,
                                   requested=dict(config))
            self._thread = threading.Thread(target=self._run, args=(config,), name="WorkbenchCamera", daemon=True)
            self._thread.start()
        return self.status()

    def stop(self) -> dict:
        with self._control:
            self._stop_event.set()
            thread = self._thread
            if thread and thread.is_alive():
                with self._lock:
                    self._state["state"] = "stopping"
                thread.join(timeout=3)
            if thread and thread.is_alive():
                # Never release a native camera from a second thread.
                with self._lock:
                    self._state["error"] = "正在等待相機驅動程式結束；完成前無法重新啟動。"
            else:
                with self._lock:
                    self._state.update(state="stopped", running=False, recording=False)
                    self._frame = None
                    self._jpeg = b""
                    self._processed_jpeg = b""
        return self.status()

    def close(self) -> None:
        self._closed = True
        self.stop()

    def set_processing(self, mode: str, settings: dict | None = None) -> dict:
        mode = "optical_flow" if mode == "flow" else mode
        if mode not in PROCESSING_MODES:
            raise ValueError(f"處理模式必須為：{', '.join(PROCESSING_MODES)}")
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("處理設定必須為物件")
        options = deepcopy(settings or {})
        if "threshold" in options:
            _integer(options["threshold"], "二值門檻", 0, 255)
        if "invert" in options and not isinstance(options["invert"], bool):
            raise ValueError("invert 必須為布林值")
        calibrate = options.pop("calibrate", False)
        if not isinstance(calibrate, bool):
            raise ValueError("calibrate 必須為布林值")
        with self._lock:
            if self._closed:
                raise AcquisitionError("相機服務已關閉")
            if calibrate:
                if mode != "classical" or not self._state["running"]:
                    raise AcquisitionError("請先啟動相機，再在古典分割模式擷取空背景。")
                options["calibration_token"] = uuid.uuid4().hex
                self._state["background_samples"] = 0
            elif "calibration_token" in self._processing_settings:
                options["calibration_token"] = self._processing_settings["calibration_token"]
            if self._frame is not None and mode in ("classical", "sam2"):
                _frame_prompts(self._frame, options)
            if self._frame is not None and options.get("roi_shape") is not None:
                _roi_mask(self._frame, options["roi_shape"])
            self._processing_mode = mode
            self._processing_settings = options
            self._processing_generation += 1
            self._processed_jpeg = self._jpeg if mode == "original" else b""
            self._state.update(processing_mode=mode, processing_state="idle", processing_error=None,
                               processing_ms=0)
        return self.status()

    def frame_jpeg(self, processed: bool = False) -> bytes:
        with self._lock:
            if not self._state["running"] or not self._jpeg:
                raise AcquisitionError(self._state["error"] or "相機尚未取得影像")
            if processed:
                # A processing algorithm may legitimately need calibration or
                # model warm-up. Keep the live view usable while its status
                # explains why the processed result is not ready yet.
                return self._processed_jpeg or self._jpeg
            return self._jpeg

    def snapshot(self) -> dict:
        with self._lock:
            if not self._state["running"] or self._frame is None:
                raise AcquisitionError(self._state["error"] or "請先啟動相機並等待影像")
            if time.monotonic() - self._frame_time > 3:
                raise AcquisitionError("相機影像已停止更新，請重新連線後擷取")
            frame = self._frame.copy()
            batch_id = self._batch_id
            source = {"kind": "camera", "index": self._state["index"],
                      "width": frame.shape[1], "height": frame.shape[0],
                      "fps": self._state["fps"], "captured_at": datetime.now(timezone.utc).isoformat()}
        path = self.storage_root / "incoming" / f"{_identifier('camera')}.png"
        _save_png(path, frame)
        return {"path": str(path), "name": path.name, "batch_id": batch_id, "source": source}

    def _request(self, kind: str, **payload) -> dict:
        with self._control:
            if not self.status()["running"] or not self._thread or not self._thread.is_alive():
                raise AcquisitionError("請先啟動相機並等待影像")
            command = _Command(kind, payload)
            self._commands.put(command)
            if not command.done.wait(5):
                command.cancelled.set()
                raise AcquisitionError("相機驅動程式未回應；請查看相機狀態或停止相機。")
            if command.error is not None:
                raise AcquisitionError(str(command.error)) from command.error
            return self.status()

    def start_recording(self, project_id: str) -> dict:
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("錄影需要專案識別碼")
        return self._request("record_start", project_id=project_id)

    def stop_recording(self) -> dict:
        if not self.status()["recording"]:
            return self.status()
        return self._request("record_stop")

    def _open_capture(self, config: dict):
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if os.name == "nt" else (cv2.CAP_ANY,)
        for backend in backends:
            if self._stop_event.is_set():
                raise AcquisitionCancelled("相機啟動已取消")
            capture = cv2.VideoCapture(config["index"], backend)
            if not capture.isOpened():
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
            # Set the selected encoding after dimensions, which may reset it.
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.get("pixel_format","MJPG")))
            capture.set(cv2.CAP_PROP_FPS, config["fps"])
            return capture
        raise AcquisitionError("無法開啟相機。請檢查連接、Windows 相機存取權及其他程式是否占用裝置。")

    def _begin_recording(self, project_id: str, frame: np.ndarray) -> dict:
        directory = self.storage_root / "recordings"
        directory.mkdir(parents=True, exist_ok=True)
        final_path = directory / f"{_identifier('recording')}.mp4"
        working_path = final_path.with_name(final_path.stem + ".part.mp4")
        height, width = frame.shape[:2]
        # MPEG-4 encoders truncate odd rows/columns; pad explicitly and disclose it.
        encoded_width, encoded_height = width + width % 2, height + height % 2
        fps = self._state["fps"] or 30.0
        writer = cv2.VideoWriter(_opencv_path(working_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (encoded_width, encoded_height))
        if not writer.isOpened():
            writer.release()
            working_path.unlink(missing_ok=True)
            raise AcquisitionError("MP4 編碼器無法啟動，請確認儲存路徑與磁碟空間。")
        record = {"writer": writer, "working_path": working_path, "path": final_path,
                  "fps": fps, "width": width, "height": height, "encoded_width": encoded_width,
                  "encoded_height": encoded_height, "frames": 0, "started": time.monotonic(),
                  "project_id": project_id, "batch_id": self._batch_id}
        with self._lock:
            self._state.update(recording=True, record_frames=0, record_seconds=0,
                               recording_project_id=project_id, error=None)
        return record

    def _write_recording(self, record: dict, frame: np.ndarray, now: float) -> None:
        if frame.shape[:2] != (record["height"], record["width"]):
            raise AcquisitionError("相機解析度在錄影期間改變，已停止錄影以保護影像資料。")
        expected = max(1, int((now - record["started"]) * record["fps"]) + 1)
        if expected - record["frames"] > max(120, record["fps"] * 5):
            raise AcquisitionError("相機影像長時間中斷，錄影已停止；已保留完成的部分。")
        if frame.shape[1] != record["encoded_width"] or frame.shape[0] != record["encoded_height"]:
            frame = cv2.copyMakeBorder(frame, 0, frame.shape[0] % 2, 0, frame.shape[1] % 2, cv2.BORDER_REPLICATE)
        while record["frames"] < expected:
            record["writer"].write(frame)
            record["frames"] += 1
        with self._lock:
            self._state.update(record_frames=record["frames"], record_seconds=round(now - record["started"], 2))

    def _finish_recording(self, record: dict, *, interrupted: bool = False) -> None:
        record["writer"].release()
        temporary, final_path = record["working_path"], record["path"]
        try:
            if not record["frames"] or not temporary.is_file() or temporary.stat().st_size <= 32:
                temporary.unlink(missing_ok=True)
                raise AcquisitionError("錄影沒有產生可用影格")
            # Verify a real decodable container before exposing the published file.
            probe = cv2.VideoCapture(_opencv_path(temporary))
            try:
                ok, first = probe.read()
            finally:
                probe.release()
            if not ok or first is None:
                raise AcquisitionError(f"錄影檔案無法解碼，暫存資料保留於：{temporary}")
            temporary.replace(final_path)
            summary = {"path": str(final_path), "name": final_path.name, "project_id": record["project_id"],
                       "batch_id": record["batch_id"], "fps": record["fps"], "frames": record["frames"],
                       "duration_seconds": record["frames"] / record["fps"], "width": record["width"],
                       "height": record["height"], "encoded_width": record["encoded_width"],
                       "encoded_height": record["encoded_height"], "interrupted": interrupted,
                       "audio": False}
            with self._lock:
                self._state["last_recording"] = summary
        finally:
            with self._lock:
                self._state["recording"] = False

    def _run(self, config: dict) -> None:
        capture, record, current_frame = None, None, None
        error = None
        try:
            capture = self._open_capture(config)
            self._processor = _PreviewProcessor(self)
            negotiated_fps = float(capture.get(cv2.CAP_PROP_FPS))
            reported_fourcc=capture.get(cv2.CAP_PROP_FOURCC)
            fourcc=int(reported_fourcc) if math.isfinite(reported_fourcc) else 0
            actual_format=''.join(chr((fourcc>>(8*i))&255) for i in range(4))
            if not actual_format.isprintable():actual_format='unknown'
            with self._lock:self._state['pixel_format']=actual_format
            if not math.isfinite(negotiated_fps) or not 1 <= negotiated_fps <= 240:
                negotiated_fps = config["fps"]
            failures = 0
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    if failures >= 5:
                        raise AcquisitionError("相機已中斷或連續無法讀取影格；請重新連線。")
                    self._stop_event.wait(.05)
                    continue
                failures = 0
                current_frame = np.ascontiguousarray(frame)
                now = time.monotonic()
                jpeg = _preview_jpeg(frame)
                with self._lock:
                    self._frame = current_frame.copy()
                    self._frame_time = now
                    self._jpeg = jpeg
                    self._timestamps.append(now)
                    self._state.update(state="running", running=True, width=frame.shape[1],
                                       height=frame.shape[0], fps=negotiated_fps)
                    self._state["frame_count"] += 1
                    processing_generation = self._processing_generation
                    processing_mode = self._processing_mode
                    processing_settings = deepcopy(self._processing_settings)
                    if processing_mode == "original":
                        self._processed_jpeg = jpeg
                        self._state.update(processing_state="ready", processing_ms=0)
                if processing_mode != "original":
                    self._processor.submit(current_frame, processing_generation, processing_mode, processing_settings)
                while True:
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if command.cancelled.is_set():
                            continue
                        if command.kind == "record_start":
                            if record is not None:
                                raise AcquisitionError("錄影已在進行中")
                            record = self._begin_recording(command.payload["project_id"], current_frame)
                        elif command.kind == "record_stop" and record is not None:
                            closing_record, record = record, None
                            try:
                                self._write_recording(closing_record, current_frame, time.monotonic())
                            except Exception:
                                self._finish_recording(closing_record, interrupted=True)
                                raise
                            else:
                                self._finish_recording(closing_record)
                    except BaseException as exc:
                        command.error = exc
                    finally:
                        command.done.set()
                if record is not None:
                    self._write_recording(record, current_frame, now)
                # Devices may return cached images immediately; yield cooperatively.
                self._stop_event.wait(.001)
        except AcquisitionCancelled:
            pass
        except Exception as exc:
            error = str(exc)
        finally:
            if self._processor is not None:
                self._processor.close()
                self._processor = None
            if record is not None:
                try:
                    self._finish_recording(record, interrupted=bool(error))
                except Exception as exc:
                    error = error or str(exc)
            if capture is not None:
                capture.release()
            with self._lock:
                self._state.update(state="error" if error else "stopped", running=False, recording=False, error=error)
                self._frame = None
                self._jpeg = b""
                self._processed_jpeg = b""
            while True:
                try:
                    command = self._commands.get_nowait()
                except queue.Empty:
                    break
                command.error = AcquisitionError(error or "相機已停止")
                command.done.set()


def capture_screen(storage_root: Path) -> dict:
    """Capture the complete Windows virtual desktop only on explicit request."""
    try:
        import mss
        with mss.mss() as capture:
            monitor = dict(capture.monitors[0])
            frame = np.asarray(capture.grab(monitor), dtype=np.uint8)[:, :, :3].copy()
    except ImportError as exc:
        raise AcquisitionError("螢幕擷取需要 mss，請執行本軟體安裝程序。") from exc
    except Exception as exc:
        raise AcquisitionError(f"無法擷取螢幕：{exc}") from exc
    batch_id = _identifier("screen")
    path = Path(storage_root).resolve() / "incoming" / f"{batch_id}.png"
    _save_png(path, frame)
    return {"path": str(path), "name": path.name, "batch_id": batch_id,
            "source": {"kind": "screen", "monitor": monitor, "captured_at": datetime.now(timezone.utc).isoformat()}}


def extract_video(path, storage_root, interval_seconds=1, *, cancel_event: threading.Event | None = None,
                  progress: Callable[[float], None] | None = None) -> list[dict]:
    interval = _number(interval_seconds, "取樣間隔（秒）", .01, 86400)
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到影片：{path}")
    capture = cv2.VideoCapture(_opencv_path(path))
    records: list[dict] = []
    batch_id = _identifier("video")
    output_dir = Path(storage_root).resolve() / "incoming" / batch_id
    try:
        if not capture.isOpened():
            raise AcquisitionError(f"無法開啟影片：{path.name}")
        from hashlib import sha256
        video_hash = sha256()
        with path.open("rb") as source_file:
            while chunk := source_file.read(1024 * 1024):
                if cancel_event is not None and cancel_event.is_set():
                    raise AcquisitionCancelled("影片取樣已取消")
                video_hash.update(chunk)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise AcquisitionError("影片沒有有效的影格率，無法可靠地依時間取樣。")
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_index, next_time = 0, 0.0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise AcquisitionCancelled("影片取樣已取消")
            ok, frame = capture.read()
            if not ok:
                if frame_count > 0 and frame_index < frame_count - 2:
                    raise AcquisitionError(f"影片在影格 {frame_index} 解碼中斷，未匯入不完整批次。")
                break
            timestamp = frame_index / fps
            if timestamp + 1e-8 >= next_time:
                target = output_dir / f"frame_{frame_index:09d}.png"
                _save_png(target, frame)
                records.append({"path": str(target), "name": f"{path.stem}_{frame_index:09d}.png",
                                "batch_id": batch_id, "source": {"kind": "video", "video": str(path),
                                "video_sha256": video_hash.hexdigest(),
                                "frame_index": frame_index, "timestamp_seconds": timestamp, "fps": fps}})
                next_time = (math.floor(timestamp / interval + 1e-8) + 1) * interval
            frame_index += 1
            if progress is not None and frame_count and frame_index % 30 == 0:
                progress(min(99., frame_index / frame_count * 100))
        if not records:
            raise AcquisitionError("影片沒有可解碼的影格")
        if progress is not None:
            progress(100.)
        return records
    except BaseException:
        # Only files created in this fresh batch are removed; input stays read-only.
        for record in records:
            Path(record["path"]).unlink(missing_ok=True)
        if output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass
        raise
    finally:
        capture.release()


_AI_LOCK = threading.RLock()
_SAM2_RUNTIME = None
_SAM2_MODEL_KEY: str | None = None


def _model_location(model_dir=None) -> str:
    default = Path(__file__).resolve().parents[1] / "models" / "sam2.1-hiera-tiny"
    if model_dir is not None:
        root = Path(model_dir).resolve()
        for candidate in (root, root / "sam2.1-hiera-tiny"):
            if (candidate / "config.json").is_file():
                return str(candidate)
        raise AcquisitionError(f"SAM2 模型目錄缺少 config.json：{root}。請先安裝或選擇完整本機模型。")
    if (default / "config.json").is_file():
        return str(default)
    return "facebook/sam2.1-hiera-tiny"  # Existing HF cache only, never downloaded here.


def ai_capabilities(model_dir=None) -> dict:
    available = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))
    try:
        location = _model_location(model_dir)
        model_local = Path(location).is_dir()
    except AcquisitionError:
        location, model_local = None, False
    return {"grabcut": True, "sam2_dependencies": available, "sam2_model": location,
            "sam2_model_local": model_local, "sam2_downloads_during_inference": False}


def close_ai() -> None:
    global _SAM2_RUNTIME, _SAM2_MODEL_KEY
    with _AI_LOCK:
        if _SAM2_RUNTIME is not None:
            _SAM2_RUNTIME.close()
        _SAM2_RUNTIME = None
        _SAM2_MODEL_KEY = None


def _validated_points(points, width, height, name) -> list[tuple[int, int]]:
    if points is None:
        return []
    if not isinstance(points, (list, tuple)):
        raise ValueError(f"{name} 必須為座標清單")
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{name} 每一點必須包含 x、y")
        x = _number(point[0], f"{name} x", 0, width - 1)
        y = _number(point[1], f"{name} y", 0, height - 1)
        result.append((int(round(x)), int(round(y))))
    return result


def _grabcut(frame, positives, negatives, box) -> np.ndarray:
    height, width = frame.shape[:2]
    if width < 8 or height < 8:
        raise ValueError("GrabCut 影像至少需要 8×8 像素")
    mask = np.full((height, width), cv2.GC_PR_FGD, dtype=np.uint8)
    if box is not None:
        left, top, right, bottom = box
        mask[:] = cv2.GC_BGD
        mask[top:bottom, left:right] = cv2.GC_PR_FGD
    else:
        border = max(1, min(width, height) // 100)
        mask[:border, :] = mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = mask[:, -border:] = cv2.GC_BGD
    radius = max(1, min(width, height) // 250)
    for point in positives:
        cv2.circle(mask, point, radius, cv2.GC_FGD, -1)
    for point in negatives:
        cv2.circle(mask, point, radius, cv2.GC_BGD, -1)
    if not np.any((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD)):
        raise ValueError("GrabCut 需要框外背景或背景提示點，請縮小選取框。")
    try:
        cv2.grabCut(frame, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        raise AcquisitionError(f"GrabCut 無法完成分割，請調整前景／背景點或選取框：{exc}") from exc
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _sam2(frame, positives, negatives, box, model_dir):
    global _SAM2_RUNTIME, _SAM2_MODEL_KEY
    from sam2_segmentation import HuggingFaceSam2Runtime, Sam2Config
    from sam2_segmentation.postprocess import Sam2PromptEncoding, finalize_sam2_mask
    location = _model_location(model_dir)
    with _AI_LOCK:
        # Transformers/HF must not initiate telemetry or network resolution here.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        if _SAM2_RUNTIME is None or _SAM2_MODEL_KEY != location:
            close_ai()
            _SAM2_RUNTIME = HuggingFaceSam2Runtime(Sam2Config(
                model_id=location, local_files_only=True, device_preference="auto",
                allow_cpu_fallback=True, smart_boundary_smoothing=False))
            _SAM2_MODEL_KEY = location
        runtime = _SAM2_RUNTIME
        try:
            points_arguments = {}
            if positives:
                encoding = Sam2PromptEncoding(tuple(positives), tuple(negatives))
                points_arguments = encoding.processor_kwargs()
            elif negatives:
                points_arguments = {"input_points": [[[list(point) for point in negatives]]],
                                    "input_labels": [[[0] * len(negatives)]]}
            output = runtime.infer(np.ascontiguousarray(frame[:, :, ::-1]),
                input_points=points_arguments.get("input_points", []),
                input_labels=points_arguments.get("input_labels", []),
                input_boxes=[[list(box)]] if box is not None else None,
                multimask_output=True, mask_threshold=0.0)
            if output.candidate_masks.shape[1:] != frame.shape[:2]:
                raise AcquisitionError("SAM2 回傳 mask 尺寸與原始影像不一致")
            if box is None:
                mask, _hierarchy, chosen = finalize_sam2_mask(
                    output, encoding, prompt_search_radius_ratio=0.005, smart_boundary_smoothing=False)
                candidate, score, violations = chosen.candidate_index, chosen.predicted_iou, chosen.prompt_violation_count
            else:
                # A box may surround an object with a hole at its center; never
                # fabricate a center foreground click for a box-only request.
                candidates = []
                for index, raw in enumerate(output.candidate_masks):
                    violations = sum(not raw[y, x] for x, y in positives) + sum(bool(raw[y, x]) for x, y in negatives)
                    if np.any(raw):
                        candidates.append((violations, -float(output.iou_scores[index]), index))
                if not candidates:
                    raise AcquisitionError("SAM2 沒有產生可用的前景遮罩，請调整提示。")
                violations, negative_score, candidate = min(candidates)
                score = -negative_score
                mask = output.candidate_masks[candidate].astype(np.uint8) * 255
            return mask, {"model_id": location, "device": runtime.device_name, "dtype": runtime.dtype_name,
                          "predicted_iou": score, "candidate_index": candidate,
                          "prompt_violation_count": int(violations), "local_files_only": True}
        except (ImportError, ModuleNotFoundError) as exc:
            close_ai()
            raise AcquisitionError("SAM2 尚未安裝 PyTorch／Transformers。請先完成本軟體的 AI 安裝；亦可明確選用 GrabCut。") from exc
        except Exception as exc:
            close_ai()
            if isinstance(exc, AcquisitionError):
                raise
            raise AcquisitionError(f"SAM2 本機推論失敗：{exc}。請確認完整模型與依賴已安装；推論不會自動下載模型。") from exc


def segment_image(image_path, engine="sam2", points=None, negative_points=None, box=None,
                  label="object", model_dir=None, *, cancel_event: threading.Event | None = None) -> dict:
    """Produce one lossless full-image native mask candidate, never approval."""
    started = time.monotonic()
    if engine not in ("sam2", "grabcut"):
        raise ValueError("分割引擎必須是 sam2 或 grabcut")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("請設定非空白的標註類別")
    if cancel_event is not None and cancel_event.is_set():
        raise AcquisitionCancelled("分割已取消")
    frame = read_image(image_path)
    height, width = frame.shape[:2]
    positive = _validated_points(points, width, height, "前景點")
    negative = _validated_points(negative_points, width, height, "背景點")
    if set(positive) & set(negative):
        raise ValueError("同一像素不能同時為前景點及背景點")
    if box is not None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("選取框需要 [x1,y1,x2,y2]")
        raw = [_number(value, "選取框座標", 0, width if index % 2 == 0 else height)
               for index, value in enumerate(box)]
        box = (int(math.floor(raw[0])), int(math.floor(raw[1])), int(math.ceil(raw[2])), int(math.ceil(raw[3])))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("選取框需要正寬度與高度")
        if any(not (box[0] <= x < box[2] and box[1] <= y < box[3]) for x, y in positive):
            raise ValueError("前景點必須位於選取框內")
    center_prompt = not positive and box is None
    if center_prompt:
        center = (width // 2, height // 2)
        if center in negative:
            raise ValueError("中心已設定為背景，請另外指定前景點或選取框")
        positive = [center]
    if engine == "grabcut":
        mask = _grabcut(frame, positive, negative, box)
        diagnostics = {"device": "cpu", "method": "opencv_grabcut", "iterations": 5}
    else:
        mask, diagnostics = _sam2(frame, positive, negative, box, model_dir)
    if cancel_event is not None and cancel_event.is_set():
        raise AcquisitionCancelled("分割已取消；候選結果未儲存")
    if mask.shape != (height, width) or not np.any(mask):
        raise AcquisitionError("分割沒有產生可用前景，請調整選取框與提示點。")
    from sam2_segmentation.training_dataset import encode_coco_uncompressed_rle
    rle = encode_coco_uncompressed_rle(mask)
    diagnostics.update(engine=engine, elapsed_ms=round((time.monotonic() - started) * 1000, 1),
                       foreground_pixels=int(np.count_nonzero(mask)), generated_center_prompt=center_prompt)
    return {"shape": {"id": uuid.uuid4().hex, "type": "mask", "label": label.strip(),
                      "hidden": False, "x": 0, "y": 0, "width": width, "height": height,
                      "counts": rle["counts"], "metadata": {"source": "ai_candidate", "engine": engine,
                          "review_state": "pending", "generated_at": datetime.now(timezone.utc).isoformat(),
                          "points": [list(point) for point in positive],
                          "negative_points": [list(point) for point in negative],
                          "box": list(box) if box is not None else None}},
            "diagnostics": diagnostics}
