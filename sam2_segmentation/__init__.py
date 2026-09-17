"""Lazy compatibility exports; optional tools load on demand."""
from importlib import import_module

_EXPORTS = {
    "HuggingFaceSam2Runtime": "backend",
    "RuntimeFactory": "backend",
    "Sam2RuntimeProtocol": "backend",
    "Sam2Segmenter": "backend",
    "Sam2Config": "models",
    "Sam2DependencyError": "models",
    "Sam2DeviceError": "models",
    "Sam2Diagnostics": "models",
    "Sam2Error": "models",
    "Sam2InferenceError": "models",
    "Sam2ModelLoadError": "models",
    "Sam2NoMaskError": "models",
    "Sam2RawOutput": "models",
    "Sam2SegmentationResult": "models",
    "Sam2WorkerState": "models",
    "Sam2CandidateSelection": "postprocess",
    "Sam2PromptEncoding": "postprocess",
    "encode_sam2_prompts": "postprocess",
    "extract_curvature_adaptive_contour_hierarchy": "postprocess",
    "finalize_sam2_mask": "postprocess",
    "keep_main_prompted_component": "postprocess",
    "select_best_sam2_candidate": "postprocess",
    "smooth_binary_mask": "postprocess",
    "Sam2LayoutRepairPlan": "layout_repair",
    "find_misnamed_sam2_sessions": "layout_repair",
    "repair_misnamed_sam2_sessions": "layout_repair",
    "COCO_SCHEMA_VERSION": "training_dataset",
    "HUMAN_VERIFIED_REVIEW_STATUS": "training_dataset",
    "LABEL_SOURCE": "training_dataset",
    "MIXED_REVIEW_STATUS": "training_dataset",
    "REJECTED_REVIEW_STATUS": "training_dataset",
    "REVIEW_STATUS": "training_dataset",
    "SAMPLE_REVIEW_STATUSES": "training_dataset",
    "SESSION_REVIEW_STATUSES": "training_dataset",
    "TRAINING_MANIFEST_SCHEMA_VERSION": "training_dataset",
    "Sam2TrainingDatasetSession": "training_dataset",
    "Sam2TrainingSampleRecord": "training_dataset",
    "decode_coco_uncompressed_rle": "training_dataset",
    "encode_coco_uncompressed_rle": "training_dataset",
    "Sam2Worker": "worker",
    "Sam2WorkerResult": "worker",
    "SegmenterFactory": "worker"
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
