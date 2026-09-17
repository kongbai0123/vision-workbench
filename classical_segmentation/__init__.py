"""Lazy compatibility exports; optional legacy tools load only on demand."""
from importlib import import_module

_EXPORTS = {
    "BackgroundCalibrator": "background",
    "as_binary_mask": "contours",
    "clean_prompted_mask": "contours",
    "extract_contour_hierarchy": "contours",
    "fill_small_holes": "contours",
    "keep_prompt_components": "contours",
    "remove_small_components": "contours",
    "BackgroundSampleRecord": "dataset",
    "SceneRecord": "dataset",
    "SegmentationDatasetSession": "dataset",
    "AggregateSegmentationMetrics": "evaluation",
    "AlgorithmRecommendation": "evaluation",
    "EvaluationBatch": "evaluation",
    "EvaluationConfig": "evaluation",
    "RecommendationConfig": "evaluation",
    "RecommendationReport": "evaluation",
    "SegmentationMetrics": "evaluation",
    "aggregate_segmentation_metrics": "evaluation",
    "count_odd_depth_holes": "evaluation",
    "evaluate_mask_pairs": "evaluation",
    "evaluate_segmentation": "evaluation",
    "recommend_follow_up_algorithms": "evaluation",
    "BackgroundModel": "models",
    "ContourHierarchy": "models",
    "NormalizedPoint": "models",
    "NormalizedRect": "models",
    "SegmentationConfig": "models",
    "SegmentationDiagnostics": "models",
    "SegmentationPrompts": "models",
    "SegmentationResult": "models",
    "ClassicalSegmenter": "segmenter",
    "SegmentationWorker": "worker",
    "SegmentationWorkerResult": "worker",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
