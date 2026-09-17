"""Lazy Hugging Face SAM2 runtime and dependency-free segmentation adapter."""

from __future__ import annotations

from contextlib import nullcontext
import gc
import threading
import time
from typing import Callable, Optional, Protocol

import numpy as np

from classical_segmentation import SegmentationPrompts, extract_contour_hierarchy

from .models import (
    Sam2Config,
    Sam2DependencyError,
    Sam2DeviceError,
    Sam2Diagnostics,
    Sam2Error,
    Sam2InferenceError,
    Sam2ModelLoadError,
    Sam2RawOutput,
    Sam2SegmentationResult,
)
from .postprocess import encode_sam2_prompts, finalize_sam2_mask


class Sam2RuntimeProtocol(Protocol):
    """Small injection seam around the heavyweight model implementation."""

    @property
    def device_name(self) -> str: ...

    @property
    def dtype_name(self) -> str: ...

    @property
    def model_load_time_ms(self) -> float: ...

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def infer(
        self,
        image_rgb: np.ndarray,
        *,
        input_points: list,
        input_labels: list,
        multimask_output: bool,
        mask_threshold: float,
        input_boxes: Optional[list] = None,
    ) -> Sam2RawOutput: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[[Sam2Config], Sam2RuntimeProtocol]


class HuggingFaceSam2Runtime:
    """A lazily loaded ``transformers.Sam2Model`` image runtime.

    Importing this module is safe when the optional ML stack is absent.  The
    imports, checkpoint download, CUDA initialization, and model transfer all
    happen inside :meth:`load`, which callers run on a background worker.
    """

    def __init__(self, config: Optional[Sam2Config] = None) -> None:
        self.config = config or Sam2Config()
        self._lock = threading.RLock()
        self._torch = None
        self._processor = None
        self._model = None
        self._device_name = "unloaded"
        self._dtype_name = "unloaded"
        self._model_load_time_ms = 0.0

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def dtype_name(self) -> str:
        return self._dtype_name

    @property
    def model_load_time_ms(self) -> float:
        return self._model_load_time_ms

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def _select_device(self, torch_module) -> str:
        preference = self.config.device_preference
        cuda_available = bool(torch_module.cuda.is_available())
        if preference == "cpu":
            return "cpu"
        if cuda_available:
            return "cuda:0"
        if preference == "cuda" and not self.config.allow_cpu_fallback:
            raise Sam2DeviceError(
                "SAM2 requested CUDA, but PyTorch cannot access an NVIDIA GPU"
            )
        return "cpu"

    def _install_loaded_model(self, torch_module, processor, model, device_name: str) -> None:
        if device_name.startswith("cuda"):
            dtype = getattr(torch_module, self.config.cuda_dtype)
        else:
            dtype = torch_module.float32
        model = model.to(device=device_name, dtype=dtype)
        model.eval()
        self._torch = torch_module
        self._processor = processor
        self._model = model
        self._device_name = device_name
        self._dtype_name = str(dtype).removeprefix("torch.")

    def load(self) -> None:
        with self._lock:
            if self.is_loaded:
                return
            started_at = time.perf_counter()
            try:
                import torch
                from transformers import Sam2Model, Sam2Processor
            except (ImportError, ModuleNotFoundError) as exc:
                raise Sam2DependencyError(
                    "SAM2 requires the optional PyTorch and Transformers packages"
                ) from exc

            try:
                cache_dir = None if self.config.cache_dir is None else str(self.config.cache_dir)
                processor = Sam2Processor.from_pretrained(
                    self.config.model_id,
                    cache_dir=cache_dir,
                    local_files_only=self.config.local_files_only,
                )
                # Force the safetensors checkpoint rather than the pickle-based
                # native SAM2 .pt file that is also present in the repository.
                model = Sam2Model.from_pretrained(
                    self.config.model_id,
                    cache_dir=cache_dir,
                    local_files_only=self.config.local_files_only,
                    use_safetensors=True,
                )
                device_name = self._select_device(torch)
                try:
                    self._install_loaded_model(torch, processor, model, device_name)
                except torch.OutOfMemoryError:
                    if device_name == "cpu" or not self.config.allow_cpu_fallback:
                        raise
                    gc.collect()
                    torch.cuda.empty_cache()
                    self._install_loaded_model(torch, processor, model, "cpu")
            except Sam2Error:
                raise
            except Exception as exc:
                raise Sam2ModelLoadError(
                    f"could not load SAM2 checkpoint {self.config.model_id!r}: {exc}"
                ) from exc
            finally:
                self._model_load_time_ms = (time.perf_counter() - started_at) * 1000.0

    def _move_to_cpu(self) -> None:
        torch_module = self._torch
        model = self._model
        if torch_module is None or model is None:
            raise Sam2InferenceError("SAM2 cannot fall back before the model is loaded")
        model.to(device="cpu", dtype=torch_module.float32)
        gc.collect()
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        self._device_name = "cpu"
        self._dtype_name = "float32"

    def _infer_loaded(
        self,
        image_rgb: np.ndarray,
        *,
        input_points: list,
        input_labels: list,
        multimask_output: bool,
        mask_threshold: float,
        input_boxes: Optional[list] = None,
    ) -> Sam2RawOutput:
        torch_module = self._torch
        processor = self._processor
        model = self._model
        if torch_module is None or processor is None or model is None:
            raise Sam2InferenceError("SAM2 runtime is not loaded")

        prompt_arguments = {}
        if input_points:
            prompt_arguments.update(input_points=input_points, input_labels=input_labels)
        if input_boxes is not None:
            prompt_arguments["input_boxes"] = input_boxes
        inputs = processor(images=image_rgb, return_tensors="pt", **prompt_arguments)
        original_sizes = inputs["original_sizes"].detach().cpu()
        inputs = inputs.to(self._device_name)
        autocast = (
            torch_module.autocast(
                device_type="cuda",
                dtype=getattr(torch_module, self.config.cuda_dtype),
            )
            if self._device_name.startswith("cuda")
            else nullcontext()
        )
        with torch_module.inference_mode(), autocast:
            outputs = model(**inputs, multimask_output=multimask_output)

        postprocessed = processor.post_process_masks(
            outputs.pred_masks.detach().float().cpu(),
            original_sizes,
            mask_threshold=mask_threshold,
            binarize=True,
        )[0]
        masks = postprocessed.detach().cpu()
        if masks.ndim == 4:
            if masks.shape[0] != 1:
                raise Sam2InferenceError(
                    "SAM2 returned multiple objects for a one-object prompt"
                )
            masks = masks[0]
        if masks.ndim != 3:
            raise Sam2InferenceError(
                f"SAM2 returned an unexpected mask shape {tuple(masks.shape)}"
            )
        candidate_masks = masks.numpy()
        iou_scores = outputs.iou_scores.detach().float().cpu().reshape(-1).numpy()
        object_score_logit = None
        object_scores = getattr(outputs, "object_score_logits", None)
        if object_scores is not None and int(object_scores.numel()) > 0:
            object_score_logit = float(
                object_scores.detach().float().cpu().reshape(-1)[0].item()
            )
        try:
            return Sam2RawOutput(
                candidate_masks=candidate_masks,
                iou_scores=iou_scores,
                object_score_logit=object_score_logit,
            )
        except (TypeError, ValueError) as exc:
            raise Sam2InferenceError(f"SAM2 returned invalid candidate masks: {exc}") from exc

    def infer(
        self,
        image_rgb: np.ndarray,
        *,
        input_points: list,
        input_labels: list,
        multimask_output: bool,
        mask_threshold: float,
        input_boxes: Optional[list] = None,
    ) -> Sam2RawOutput:
        with self._lock:
            self.load()
            assert self._torch is not None
            try:
                return self._infer_loaded(
                    image_rgb,
                    input_points=input_points,
                    input_labels=input_labels,
                    multimask_output=multimask_output,
                    mask_threshold=mask_threshold,
                    input_boxes=input_boxes,
                )
            except self._torch.OutOfMemoryError as exc:
                if not self._device_name.startswith("cuda") or not self.config.allow_cpu_fallback:
                    raise Sam2InferenceError("SAM2 ran out of device memory") from exc
                self._move_to_cpu()
                try:
                    return self._infer_loaded(
                        image_rgb,
                        input_points=input_points,
                        input_labels=input_labels,
                        multimask_output=multimask_output,
                        mask_threshold=mask_threshold,
                        input_boxes=input_boxes,
                    )
                except Exception as retry_exc:
                    raise Sam2InferenceError(
                        f"SAM2 CPU retry failed after CUDA ran out of memory: {retry_exc}"
                    ) from retry_exc
            except Sam2Error:
                raise
            except Exception as exc:
                raise Sam2InferenceError(f"SAM2 inference failed: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            torch_module = self._torch
            was_cuda = self._device_name.startswith("cuda")
            self._model = None
            self._processor = None
            self._torch = None
            self._device_name = "closed"
            self._dtype_name = "closed"
        gc.collect()
        if was_cuda and torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()


class Sam2Segmenter:
    """Convert application prompts into one full-resolution SAM2 object mask."""

    def __init__(
        self,
        config: Optional[Sam2Config] = None,
        *,
        runtime_factory: Optional[RuntimeFactory] = None,
    ) -> None:
        self.config = config or Sam2Config()
        if not isinstance(self.config, Sam2Config):
            raise TypeError("config must be a Sam2Config or None")
        self._runtime_factory = runtime_factory or HuggingFaceSam2Runtime
        if not callable(self._runtime_factory):
            raise TypeError("runtime_factory must be callable")
        self._runtime: Optional[Sam2RuntimeProtocol] = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._runtime is not None and self._runtime.is_loaded

    def load(self) -> None:
        with self._lock:
            if self._runtime is None:
                self._runtime = self._runtime_factory(self.config)
            self._runtime.load()

    @staticmethod
    def _validate_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame_bgr)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("frame_bgr must be a non-empty HxWx3 uint8 BGR image")
        return np.ascontiguousarray(frame)

    def segment(
        self,
        frame_bgr: np.ndarray,
        prompts: SegmentationPrompts,
    ) -> Sam2SegmentationResult:
        started_at = time.perf_counter()
        if not isinstance(prompts, SegmentationPrompts):
            raise TypeError("prompts must be a SegmentationPrompts")
        frame = self._validate_bgr_frame(frame_bgr)
        source_size = (int(frame.shape[1]), int(frame.shape[0]))
        encoding = encode_sam2_prompts(prompts, source_size)
        # NumPy slicing reverses BGR to RGB without depending on PIL.
        image_rgb = np.ascontiguousarray(frame[:, :, ::-1])

        with self._lock:
            self.load()
            assert self._runtime is not None
            kwargs = encoding.processor_kwargs()
            output = self._runtime.infer(
                image_rgb,
                input_points=kwargs["input_points"],
                input_labels=kwargs["input_labels"],
                multimask_output=self.config.multimask_output,
                mask_threshold=self.config.mask_threshold,
            )
            if output.candidate_masks.shape[1:] != (source_size[1], source_size[0]):
                raise Sam2InferenceError(
                    "SAM2 post-processing did not restore the source image resolution"
                )
            mask, hierarchy, selection = finalize_sam2_mask(
                output,
                encoding,
                prompt_search_radius_ratio=self.config.prompt_search_radius_ratio,
                iou_score_window=self.config.iou_score_window,
                smart_boundary_smoothing=self.config.smart_boundary_smoothing,
                boundary_smoothing_radius=self.config.boundary_smoothing_radius,
            )
            cleanup = {"holes_detected": 0, "holes_filled": 0, "pixels_filled": 0}
            if self.config.repair_tiny_holes:
                from composer_core.mask_cleanup import repair_tiny_holes
                mask, cleanup = repair_tiny_holes(
                    mask,
                    protected_background_points=encoding.background_pixels,
                    max_hole_pixels=self.config.tiny_hole_max_pixels,
                    max_total_pixels=self.config.tiny_hole_total_pixels,
                    max_ratio=self.config.tiny_hole_max_ratio,
                    max_dimension=self.config.tiny_hole_max_dimension,
                )
                if cleanup["pixels_filled"]:
                    # The reusable contour now describes exactly the canonical
                    # mask that the dataset writer will persist.
                    hierarchy = extract_contour_hierarchy(mask)
            diagnostics = Sam2Diagnostics(
                source_size=source_size,
                model_id=self.config.model_id,
                device=self._runtime.device_name,
                dtype=self._runtime.dtype_name,
                candidate_index=selection.candidate_index,
                candidate_count=int(output.candidate_masks.shape[0]),
                predicted_iou=selection.predicted_iou,
                object_score_logit=output.object_score_logit,
                prompt_violation_count=selection.prompt_violation_count,
                foreground_ratio=float(np.count_nonzero(mask)) / float(mask.size),
                foreground_point_count=len(encoding.foreground_pixels),
                background_point_count=len(encoding.background_pixels),
                processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
                model_load_time_ms=self._runtime.model_load_time_ms,
                holes_detected=cleanup["holes_detected"],
                holes_filled=cleanup["holes_filled"],
                hole_pixels_filled=cleanup["pixels_filled"],
            )
        return Sam2SegmentationResult(
            mask=mask,
            hierarchy=hierarchy,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        with self._lock:
            runtime, self._runtime = self._runtime, None
        if runtime is not None:
            runtime.close()


__all__ = [
    "HuggingFaceSam2Runtime",
    "RuntimeFactory",
    "Sam2RuntimeProtocol",
    "Sam2Segmenter",
]
