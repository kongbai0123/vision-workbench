"""Quantitative evaluation for binary segmentation masks.

The evaluator is deliberately independent from the segmentation algorithm.  It
accepts only a predicted binary mask and its manually annotated ground truth,
which makes it suitable for comparing the MVP with later Chan--Vese or
parameter-sweep experiments.

Metric conventions
------------------
``false_positive_ratio`` and ``false_negative_ratio`` use the whole image as
their denominator.  ``under_segmentation_ratio`` and
``over_segmentation_ratio`` use the ground-truth foreground area, so (for a
non-empty ground truth) their difference equals ``signed_area_bias``.  When the
ground truth is empty, the image area is used as a finite fallback denominator.

Boundary precision measures predicted-boundary coverage and boundary recall
measures ground-truth-boundary coverage.  A pixel is matched when the nearest
opposite boundary is within ``boundary_tolerance_px`` Euclidean pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


_RECOMMENDATION_STATUSES = {
    "recommended",
    "not_recommended",
    "insufficient_evidence",
}


def _finite_non_negative(value: object, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(parsed)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for one-mask evaluation."""

    boundary_tolerance_px: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "boundary_tolerance_px",
            _finite_non_negative(self.boundary_tolerance_px, "boundary_tolerance_px"),
        )

    def to_dict(self) -> dict[str, float]:
        return {"boundary_tolerance_px": self.boundary_tolerance_px}


@dataclass(frozen=True)
class RecommendationConfig:
    """Conservative rule thresholds for follow-up algorithm experiments."""

    minimum_repeated_samples: int = 3
    low_iou: float = 0.90
    significant_under_segmentation: float = 0.05
    significant_over_segmentation: float = 0.05
    significant_missing_boundary: float = 0.08
    low_boundary_precision: float = 0.90
    threshold_iou_range: float = 0.04
    threshold_area_bias_range: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_repeated_samples",
            _positive_integer(self.minimum_repeated_samples, "minimum_repeated_samples"),
        )
        for name in (
            "low_iou",
            "significant_under_segmentation",
            "significant_over_segmentation",
            "significant_missing_boundary",
            "low_boundary_precision",
            "threshold_iou_range",
            "threshold_area_bias_range",
        ):
            value = _finite_non_negative(getattr(self, name), name)
            if value > 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum_repeated_samples": self.minimum_repeated_samples,
            "low_iou": self.low_iou,
            "significant_under_segmentation": self.significant_under_segmentation,
            "significant_over_segmentation": self.significant_over_segmentation,
            "significant_missing_boundary": self.significant_missing_boundary,
            "low_boundary_precision": self.low_boundary_precision,
            "threshold_iou_range": self.threshold_iou_range,
            "threshold_area_bias_range": self.threshold_area_bias_range,
        }


@dataclass(frozen=True)
class SegmentationMetrics:
    """All measurements for one prediction/ground-truth pair."""

    image_shape: Tuple[int, int]
    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    true_negative_pixels: int
    predicted_area_pixels: int
    ground_truth_area_pixels: int
    predicted_boundary_pixels: int
    ground_truth_boundary_pixels: int
    iou: float
    dice: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    missing_boundary_rate: float
    false_positive_ratio: float
    false_negative_ratio: float
    under_segmentation_ratio: float
    over_segmentation_ratio: float
    signed_area_bias: float
    boundary_mean_distance_px: float
    boundary_p95_distance_px: float
    predicted_hole_count: int
    ground_truth_hole_count: int
    hole_count_error: int
    absolute_hole_count_error: int

    def to_dict(self) -> dict[str, object]:
        return {
            "image_shape": list(self.image_shape),
            "true_positive_pixels": self.true_positive_pixels,
            "false_positive_pixels": self.false_positive_pixels,
            "false_negative_pixels": self.false_negative_pixels,
            "true_negative_pixels": self.true_negative_pixels,
            "predicted_area_pixels": self.predicted_area_pixels,
            "ground_truth_area_pixels": self.ground_truth_area_pixels,
            "predicted_boundary_pixels": self.predicted_boundary_pixels,
            "ground_truth_boundary_pixels": self.ground_truth_boundary_pixels,
            "iou": self.iou,
            "dice": self.dice,
            "boundary_precision": self.boundary_precision,
            "boundary_recall": self.boundary_recall,
            "boundary_f1": self.boundary_f1,
            "missing_boundary_rate": self.missing_boundary_rate,
            "false_positive_ratio": self.false_positive_ratio,
            "false_negative_ratio": self.false_negative_ratio,
            "under_segmentation_ratio": self.under_segmentation_ratio,
            "over_segmentation_ratio": self.over_segmentation_ratio,
            "signed_area_bias": self.signed_area_bias,
            "boundary_mean_distance_px": self.boundary_mean_distance_px,
            "boundary_p95_distance_px": self.boundary_p95_distance_px,
            "predicted_hole_count": self.predicted_hole_count,
            "ground_truth_hole_count": self.ground_truth_hole_count,
            "hole_count_error": self.hole_count_error,
            "absolute_hole_count_error": self.absolute_hole_count_error,
        }


@dataclass(frozen=True)
class AggregateSegmentationMetrics:
    """Macro means and useful totals over multiple image pairs."""

    sample_count: int
    total_true_positive_pixels: int
    total_false_positive_pixels: int
    total_false_negative_pixels: int
    total_ground_truth_area_pixels: int
    total_predicted_area_pixels: int
    micro_iou: float
    micro_dice: float
    mean_iou: float
    mean_dice: float
    mean_boundary_precision: float
    mean_boundary_recall: float
    mean_boundary_f1: float
    mean_missing_boundary_rate: float
    mean_false_positive_ratio: float
    mean_false_negative_ratio: float
    mean_under_segmentation_ratio: float
    mean_over_segmentation_ratio: float
    mean_signed_area_bias: float
    signed_area_bias_std: float
    mean_boundary_mean_distance_px: float
    mean_boundary_p95_distance_px: float
    total_predicted_hole_count: int
    total_ground_truth_hole_count: int
    total_hole_count_error: int
    mean_absolute_hole_count_error: float
    correct_hole_count_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class EvaluationBatch:
    """Per-image and aggregate results for a collection of paired masks."""

    metrics: Tuple[SegmentationMetrics, ...]
    aggregate: AggregateSegmentationMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": [metric.to_dict() for metric in self.metrics],
            "aggregate": self.aggregate.to_dict(),
        }


@dataclass(frozen=True)
class AlgorithmRecommendation:
    """One explainable rule assessment, including negative/unknown outcomes."""

    algorithm: str
    status: str
    rationale: str
    evidence: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _RECOMMENDATION_STATUSES:
            raise ValueError(f"unsupported recommendation status: {self.status}")
        if not self.algorithm:
            raise ValueError("algorithm must not be empty")
        if not self.rationale:
            raise ValueError("rationale must not be empty")
        object.__setattr__(self, "evidence", tuple(str(item) for item in self.evidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "status": self.status,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class RecommendationReport:
    """Assessments for every supported follow-up technique."""

    assessments: Tuple[AlgorithmRecommendation, ...]

    @property
    def recommended(self) -> Tuple[AlgorithmRecommendation, ...]:
        return tuple(item for item in self.assessments if item.status == "recommended")

    def for_algorithm(self, algorithm: str) -> AlgorithmRecommendation:
        for assessment in self.assessments:
            if assessment.algorithm == algorithm:
                return assessment
        raise KeyError(algorithm)

    def to_dict(self) -> dict[str, object]:
        return {
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "recommended_algorithms": [item.algorithm for item in self.recommended],
        }


def _validate_binary_mask(mask: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if array.dtype.kind not in "buif":
        raise ValueError(f"{name} must contain numeric or boolean values")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must not contain NaN or infinity")
    valid = (array == 0) | (array == 1) | (array == 255)
    if not bool(np.all(valid)):
        raise ValueError(f"{name} must contain only binary values 0/1/255")
    return np.ascontiguousarray(array != 0, dtype=bool)


def _extract_boundary(binary: np.ndarray) -> np.ndarray:
    contours, _hierarchy = cv2.findContours(
        binary.astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    boundary = np.zeros(binary.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(boundary, contours, -1, 1, thickness=1)
    return boundary.astype(bool)


def _distance_to_boundary(target_boundary: np.ndarray) -> np.ndarray:
    non_boundary = np.ascontiguousarray(~target_boundary, dtype=np.uint8)
    return cv2.distanceTransform(non_boundary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _directional_boundary_distances(
    source_boundary: np.ndarray,
    target_boundary: np.ndarray,
    absent_penalty: float,
) -> np.ndarray:
    source_count = int(np.count_nonzero(source_boundary))
    if source_count == 0:
        return np.empty(0, dtype=np.float32)
    if not np.any(target_boundary):
        return np.full(source_count, absent_penalty, dtype=np.float32)
    return _distance_to_boundary(target_boundary)[source_boundary]


def _coverage(
    source_boundary: np.ndarray,
    target_boundary: np.ndarray,
    tolerance: float,
) -> float:
    source_count = int(np.count_nonzero(source_boundary))
    if source_count == 0:
        return 1.0
    if not np.any(target_boundary):
        return 0.0
    distances = _distance_to_boundary(target_boundary)[source_boundary]
    return float(np.mean(distances <= tolerance + 1e-6))


def count_odd_depth_holes(mask: np.ndarray) -> int:
    """Count all RETR_TREE contours at odd parent depth."""

    binary = _validate_binary_mask(mask, "mask")
    contours, hierarchy = cv2.findContours(
        binary.astype(np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None or not contours:
        return 0
    count = 0
    for index in range(len(contours)):
        depth = 0
        parent = int(hierarchy[0, index, 3])
        seen = {index}
        while parent >= 0:
            if parent in seen:
                raise ValueError("OpenCV returned a cyclic contour hierarchy")
            seen.add(parent)
            depth += 1
            parent = int(hierarchy[0, parent, 3])
        if depth % 2 == 1:
            count += 1
    return count


def evaluate_segmentation(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    config: Optional[EvaluationConfig] = None,
) -> SegmentationMetrics:
    """Evaluate one predicted mask against one manually annotated mask."""

    config = config or EvaluationConfig()
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    predicted = _validate_binary_mask(prediction, "prediction")
    truth = _validate_binary_mask(ground_truth, "ground_truth")
    if predicted.shape != truth.shape:
        raise ValueError("prediction and ground_truth must have the same shape")

    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))
    true_negative = int(np.count_nonzero(~predicted & ~truth))
    predicted_area = true_positive + false_positive
    truth_area = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    combined_area = predicted_area + truth_area
    iou = 1.0 if union == 0 else true_positive / union
    dice = 1.0 if combined_area == 0 else 2.0 * true_positive / combined_area

    predicted_boundary = _extract_boundary(predicted)
    truth_boundary = _extract_boundary(truth)
    predicted_boundary_count = int(np.count_nonzero(predicted_boundary))
    truth_boundary_count = int(np.count_nonzero(truth_boundary))
    precision = _coverage(
        predicted_boundary,
        truth_boundary,
        config.boundary_tolerance_px,
    )
    recall = _coverage(
        truth_boundary,
        predicted_boundary,
        config.boundary_tolerance_px,
    )
    boundary_f1 = (
        1.0
        if predicted_boundary_count == 0 and truth_boundary_count == 0
        else (0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall))
    )

    height, width = predicted.shape
    absent_penalty = max(1.0, math.hypot(width - 1, height - 1))
    predicted_to_truth = _directional_boundary_distances(
        predicted_boundary,
        truth_boundary,
        absent_penalty,
    )
    truth_to_predicted = _directional_boundary_distances(
        truth_boundary,
        predicted_boundary,
        absent_penalty,
    )
    if predicted_to_truth.size == 0 and truth_to_predicted.size == 0:
        boundary_mean_distance = 0.0
        boundary_p95_distance = 0.0
    else:
        symmetric_distances = np.concatenate((predicted_to_truth, truth_to_predicted))
        boundary_mean_distance = float(np.mean(symmetric_distances))
        boundary_p95_distance = float(np.percentile(symmetric_distances, 95))

    image_area = int(predicted.size)
    reference_area = truth_area if truth_area > 0 else image_area
    under_segmentation = false_negative / reference_area
    over_segmentation = false_positive / reference_area
    signed_area_bias = (predicted_area - truth_area) / reference_area
    predicted_holes = count_odd_depth_holes(predicted)
    truth_holes = count_odd_depth_holes(truth)
    hole_error = predicted_holes - truth_holes

    return SegmentationMetrics(
        image_shape=(height, width),
        true_positive_pixels=true_positive,
        false_positive_pixels=false_positive,
        false_negative_pixels=false_negative,
        true_negative_pixels=true_negative,
        predicted_area_pixels=predicted_area,
        ground_truth_area_pixels=truth_area,
        predicted_boundary_pixels=predicted_boundary_count,
        ground_truth_boundary_pixels=truth_boundary_count,
        iou=float(iou),
        dice=float(dice),
        boundary_precision=precision,
        boundary_recall=recall,
        boundary_f1=boundary_f1,
        missing_boundary_rate=1.0 - recall,
        false_positive_ratio=false_positive / image_area,
        false_negative_ratio=false_negative / image_area,
        under_segmentation_ratio=under_segmentation,
        over_segmentation_ratio=over_segmentation,
        signed_area_bias=signed_area_bias,
        boundary_mean_distance_px=boundary_mean_distance,
        boundary_p95_distance_px=boundary_p95_distance,
        predicted_hole_count=predicted_holes,
        ground_truth_hole_count=truth_holes,
        hole_count_error=hole_error,
        absolute_hole_count_error=abs(hole_error),
    )


def aggregate_segmentation_metrics(
    metrics: Iterable[SegmentationMetrics],
) -> AggregateSegmentationMetrics:
    """Macro-average an iterable of per-image metrics (must be non-empty)."""

    items = tuple(metrics)
    if not items:
        raise ValueError("at least one SegmentationMetrics item is required")
    if not all(isinstance(item, SegmentationMetrics) for item in items):
        raise TypeError("metrics must contain only SegmentationMetrics items")

    def mean(attribute: str) -> float:
        return float(np.mean([float(getattr(item, attribute)) for item in items]))

    total_tp = sum(item.true_positive_pixels for item in items)
    total_fp = sum(item.false_positive_pixels for item in items)
    total_fn = sum(item.false_negative_pixels for item in items)
    total_truth = sum(item.ground_truth_area_pixels for item in items)
    total_prediction = sum(item.predicted_area_pixels for item in items)
    micro_union = total_tp + total_fp + total_fn
    micro_combined = total_prediction + total_truth
    micro_iou = 1.0 if micro_union == 0 else total_tp / micro_union
    micro_dice = 1.0 if micro_combined == 0 else 2.0 * total_tp / micro_combined
    area_biases = np.asarray([item.signed_area_bias for item in items], dtype=np.float64)

    return AggregateSegmentationMetrics(
        sample_count=len(items),
        total_true_positive_pixels=total_tp,
        total_false_positive_pixels=total_fp,
        total_false_negative_pixels=total_fn,
        total_ground_truth_area_pixels=total_truth,
        total_predicted_area_pixels=total_prediction,
        micro_iou=float(micro_iou),
        micro_dice=float(micro_dice),
        mean_iou=mean("iou"),
        mean_dice=mean("dice"),
        mean_boundary_precision=mean("boundary_precision"),
        mean_boundary_recall=mean("boundary_recall"),
        mean_boundary_f1=mean("boundary_f1"),
        mean_missing_boundary_rate=mean("missing_boundary_rate"),
        mean_false_positive_ratio=mean("false_positive_ratio"),
        mean_false_negative_ratio=mean("false_negative_ratio"),
        mean_under_segmentation_ratio=mean("under_segmentation_ratio"),
        mean_over_segmentation_ratio=mean("over_segmentation_ratio"),
        mean_signed_area_bias=mean("signed_area_bias"),
        signed_area_bias_std=float(np.std(area_biases)),
        mean_boundary_mean_distance_px=mean("boundary_mean_distance_px"),
        mean_boundary_p95_distance_px=mean("boundary_p95_distance_px"),
        total_predicted_hole_count=sum(item.predicted_hole_count for item in items),
        total_ground_truth_hole_count=sum(item.ground_truth_hole_count for item in items),
        total_hole_count_error=sum(item.hole_count_error for item in items),
        mean_absolute_hole_count_error=mean("absolute_hole_count_error"),
        correct_hole_count_rate=float(
            np.mean([item.hole_count_error == 0 for item in items])
        ),
    )


def evaluate_mask_pairs(
    pairs: Iterable[Tuple[np.ndarray, np.ndarray]],
    config: Optional[EvaluationConfig] = None,
) -> EvaluationBatch:
    """Evaluate ``(prediction, ground_truth)`` pairs and aggregate the results."""

    results = tuple(
        evaluate_segmentation(prediction, ground_truth, config)
        for prediction, ground_truth in pairs
    )
    return EvaluationBatch(
        metrics=results,
        aggregate=aggregate_segmentation_metrics(results),
    )


def _coerce_aggregate(
    metrics: SegmentationMetrics
    | AggregateSegmentationMetrics
    | Iterable[SegmentationMetrics],
) -> AggregateSegmentationMetrics:
    if isinstance(metrics, AggregateSegmentationMetrics):
        return metrics
    if isinstance(metrics, SegmentationMetrics):
        return aggregate_segmentation_metrics((metrics,))
    return aggregate_segmentation_metrics(metrics)


def recommend_follow_up_algorithms(
    metrics: SegmentationMetrics
    | AggregateSegmentationMetrics
    | Iterable[SegmentationMetrics],
    *,
    threshold_candidates: Optional[Sequence[SegmentationMetrics]] = None,
    config: Optional[RecommendationConfig] = None,
) -> RecommendationReport:
    """Produce conservative, explainable follow-up experiment assessments.

    ``threshold_candidates`` must contain results for the *same input* under at
    least two threshold settings.  Ordinary dataset samples are never treated
    as threshold sensitivity evidence.

    These rules identify error signatures, not physical causes.  In
    particular, excess foreground can make illumination leakage plausible but
    a mask alone cannot prove that the excess came from a shadow.
    """

    config = config or RecommendationConfig()
    if not isinstance(config, RecommendationConfig):
        raise TypeError("config must be a RecommendationConfig")
    aggregate = _coerce_aggregate(metrics)
    assessments = []

    shrink_signature = (
        aggregate.mean_iou <= config.low_iou
        and aggregate.mean_signed_area_bias <= -config.significant_under_segmentation
        and aggregate.mean_under_segmentation_ratio
        >= config.significant_under_segmentation
        and aggregate.mean_missing_boundary_rate
        >= config.significant_missing_boundary
    )
    if shrink_signature and aggregate.sample_count >= config.minimum_repeated_samples:
        assessments.append(
            AlgorithmRecommendation(
                algorithm="chan_vese",
                status="recommended",
                rationale=(
                    "Repeated inward/under-segmentation boundary errors justify a "
                    "controlled Chan--Vese experiment."
                ),
                evidence=(
                    f"samples={aggregate.sample_count}",
                    f"mean_iou={aggregate.mean_iou:.4f}",
                    f"mean_signed_area_bias={aggregate.mean_signed_area_bias:.4f}",
                    f"mean_missing_boundary_rate={aggregate.mean_missing_boundary_rate:.4f}",
                ),
            )
        )
    elif shrink_signature:
        assessments.append(
            AlgorithmRecommendation(
                algorithm="chan_vese",
                status="insufficient_evidence",
                rationale=(
                    "An inward error signature is present, but it has not yet been "
                    "observed on enough independent images to call it persistent."
                ),
                evidence=(
                    f"samples={aggregate.sample_count}",
                    f"required_samples={config.minimum_repeated_samples}",
                    f"mean_iou={aggregate.mean_iou:.4f}",
                ),
            )
        )
    else:
        assessments.append(
            AlgorithmRecommendation(
                algorithm="chan_vese",
                status="not_recommended",
                rationale="The evaluated masks do not show a persistent inward-boundary signature.",
                evidence=(
                    f"mean_under_segmentation_ratio={aggregate.mean_under_segmentation_ratio:.4f}",
                    f"mean_missing_boundary_rate={aggregate.mean_missing_boundary_rate:.4f}",
                ),
            )
        )

    if threshold_candidates is None or len(threshold_candidates) < 2:
        assessments.append(
            AlgorithmRecommendation(
                algorithm="threshold_sweep",
                status="insufficient_evidence",
                rationale=(
                    "Threshold sensitivity cannot be inferred from a single mask; "
                    "evaluate the same image under at least two threshold settings."
                ),
                evidence=("threshold_candidate_count=0" if threshold_candidates is None else f"threshold_candidate_count={len(threshold_candidates)}",),
            )
        )
    else:
        if not all(isinstance(item, SegmentationMetrics) for item in threshold_candidates):
            raise TypeError("threshold_candidates must contain SegmentationMetrics items")
        candidate_ious = [item.iou for item in threshold_candidates]
        candidate_biases = [item.signed_area_bias for item in threshold_candidates]
        iou_range = max(candidate_ious) - min(candidate_ious)
        bias_range = max(candidate_biases) - min(candidate_biases)
        sensitive = (
            iou_range >= config.threshold_iou_range
            or bias_range >= config.threshold_area_bias_range
        )
        assessments.append(
            AlgorithmRecommendation(
                algorithm="threshold_sweep",
                status="recommended" if sensitive else "not_recommended",
                rationale=(
                    "The same input changes materially across threshold candidates; "
                    "retain multi-candidate search."
                    if sensitive
                    else "The tested threshold candidates produce sufficiently stable masks."
                ),
                evidence=(
                    f"threshold_candidate_count={len(threshold_candidates)}",
                    f"iou_range={iou_range:.4f}",
                    f"signed_area_bias_range={bias_range:.4f}",
                ),
            )
        )

    leakage_signature = (
        aggregate.mean_over_segmentation_ratio
        >= config.significant_over_segmentation
        and aggregate.mean_boundary_precision <= config.low_boundary_precision
    )
    assessments.append(
        AlgorithmRecommendation(
            algorithm="illumination_compensation",
            status="recommended" if leakage_signature else "not_recommended",
            rationale=(
                "Excess foreground with weak boundary precision is consistent with "
                "illumination leakage; inspect overlays to confirm whether shadows are "
                "the cause before adding compensation."
                if leakage_signature
                else "The masks do not show the over-segmentation leakage signature."
            ),
            evidence=(
                f"mean_over_segmentation_ratio={aggregate.mean_over_segmentation_ratio:.4f}",
                f"mean_boundary_precision={aggregate.mean_boundary_precision:.4f}",
            ),
        )
    )
    return RecommendationReport(tuple(assessments))


# Concise aliases for callers that naturally use mask/metric terminology.
evaluate_mask = evaluate_segmentation
aggregate_metrics = aggregate_segmentation_metrics
recommend_algorithms = recommend_follow_up_algorithms
count_holes = count_odd_depth_holes


__all__ = [
    "AggregateSegmentationMetrics",
    "AlgorithmRecommendation",
    "EvaluationBatch",
    "EvaluationConfig",
    "RecommendationConfig",
    "RecommendationReport",
    "SegmentationMetrics",
    "aggregate_metrics",
    "aggregate_segmentation_metrics",
    "count_holes",
    "count_odd_depth_holes",
    "evaluate_mask",
    "evaluate_mask_pairs",
    "evaluate_segmentation",
    "recommend_algorithms",
    "recommend_follow_up_algorithms",
]
