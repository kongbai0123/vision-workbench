"""Classical, calibration-based prompted object segmentation.

This package intentionally has no GUI dependency. It can therefore be used by
the Tk application, offline capture processing, or focused unit tests.
"""

from .background import BackgroundCalibrator
from .contours import (
    as_binary_mask,
    clean_prompted_mask,
    extract_contour_hierarchy,
    fill_small_holes,
    keep_prompt_components,
    remove_small_components,
)
from .dataset import (
    BackgroundSampleRecord,
    SceneRecord,
    SegmentationDatasetSession,
)
from .evaluation import (
    AggregateSegmentationMetrics,
    AlgorithmRecommendation,
    EvaluationBatch,
    EvaluationConfig,
    RecommendationConfig,
    RecommendationReport,
    SegmentationMetrics,
    aggregate_segmentation_metrics,
    count_odd_depth_holes,
    evaluate_mask_pairs,
    evaluate_segmentation,
    recommend_follow_up_algorithms,
)
from .models import (
    BackgroundModel,
    ContourHierarchy,
    NormalizedPoint,
    NormalizedRect,
    SegmentationConfig,
    SegmentationDiagnostics,
    SegmentationPrompts,
    SegmentationResult,
)
from .segmenter import ClassicalSegmenter
from .worker import SegmentationWorker, SegmentationWorkerResult

__all__ = [
    "BackgroundCalibrator",
    "BackgroundSampleRecord",
    "BackgroundModel",
    "ClassicalSegmenter",
    "ContourHierarchy",
    "AggregateSegmentationMetrics",
    "AlgorithmRecommendation",
    "EvaluationBatch",
    "EvaluationConfig",
    "NormalizedPoint",
    "NormalizedRect",
    "RecommendationConfig",
    "RecommendationReport",
    "SegmentationConfig",
    "SegmentationDatasetSession",
    "SegmentationDiagnostics",
    "SegmentationPrompts",
    "SegmentationResult",
    "SegmentationMetrics",
    "SceneRecord",
    "SegmentationWorker",
    "SegmentationWorkerResult",
    "as_binary_mask",
    "aggregate_segmentation_metrics",
    "clean_prompted_mask",
    "count_odd_depth_holes",
    "evaluate_mask_pairs",
    "evaluate_segmentation",
    "extract_contour_hierarchy",
    "fill_small_holes",
    "keep_prompt_components",
    "remove_small_components",
    "recommend_follow_up_algorithms",
]
