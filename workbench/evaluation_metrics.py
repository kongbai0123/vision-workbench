"""Task-specific evaluation summaries with explicit undefined-score handling.

The helpers in this module deliberately separate model selection (Validation)
from final reporting (Test).  They also keep raw counts so a displayed average
can always be audited and recomputed.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


EVALUATION_SCHEMA_VERSION = 2


class PixelMetrics:
    """Accumulate class-union pixel counts across a complete dataset split."""

    def __init__(self, labels):
        self.labels = list(labels)
        self.counts = {label: {"tp": 0, "fp": 0, "fn": 0,
                               "images_with_ground_truth": 0,
                               "images_with_prediction": 0}
                       for label in self.labels}

    def update(self, label, predicted, expected):
        predicted = np.asarray(predicted, dtype=bool)
        expected = np.asarray(expected, dtype=bool)
        if predicted.shape != expected.shape:
            raise ValueError(f"預測與標註遮罩尺寸不符：{predicted.shape} / {expected.shape}")
        row = self.counts[label]
        row["tp"] += int(np.count_nonzero(predicted & expected))
        row["fp"] += int(np.count_nonzero(predicted & ~expected))
        row["fn"] += int(np.count_nonzero(~predicted & expected))
        row["images_with_ground_truth"] += int(expected.any())
        row["images_with_prediction"] += int(predicted.any())

    def summary(self, split, images, *, include_dice=False):
        per_class, ious, dices = {}, [], []
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for label in self.labels:
            raw = self.counts[label]
            tp, fp, fn = raw["tp"], raw["fp"], raw["fn"]
            union = tp + fp + fn
            denominator = 2 * tp + fp + fn
            iou = round(tp / union, 6) if union else None
            dice = round(2 * tp / denominator, 6) if denominator else None
            support = tp + fn
            predicted = tp + fp
            row = {**raw, "ground_truth_pixels": support,
                   "predicted_pixels": predicted, "iou": iou}
            if include_dice:
                row["dice"] = dice
            per_class[label] = row
            if iou is not None:
                ious.append(iou)
            if include_dice and dice is not None:
                dices.append(dice)
            for key in totals:
                totals[key] += raw[key]
        union = totals["tp"] + totals["fp"] + totals["fn"]
        result = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "split": split,
            "images": int(images),
            "aggregation": "dataset_class_union_pixels",
            "mean_iou": round(sum(ious) / len(ious), 6) if ious else None,
            "micro_iou": round(totals["tp"] / union, 6) if union else None,
            "per_class_iou": {label: row["iou"] for label, row in per_class.items()},
            "per_class": per_class,
            "totals": totals,
        }
        if include_dice:
            result["mean_dice"] = round(sum(dices) / len(dices), 6) if dices else None
            result["per_class_dice"] = {label: row["dice"] for label, row in per_class.items()}
        return result


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _average_precision(points, positives):
    if not positives:
        return None
    tp = fp = 0
    curve = []
    for matched in points:
        tp += int(matched)
        fp += int(not matched)
        curve.append((tp / positives, tp / (tp + fp)))
    # COCO-style 101-point interpolated AP for one IoU threshold.
    return sum(max((precision for recall, precision in curve if recall >= level), default=0.0)
               for level in np.linspace(0.0, 1.0, 101)) / 101


def detection_summary(labels, samples, split, images, *, score_threshold=.5, iou_threshold=.5):
    """Score detections with score-sorted one-to-one matching.

    ``samples`` entries contain ``expected`` and ``predicted`` mappings.  Each
    prediction is ``(score, xyxy)`` and each expected item is an ``xyxy`` box.
    AP uses every returned score; fixed-threshold precision/recall uses the
    configured deployment threshold.
    """
    labels = list(labels)
    per_class = {}
    aggregate = {"tp": 0, "fp": 0, "fn": 0}
    aps = []
    for label in labels:
        expected_by_image = {index: list(sample.get("expected", {}).get(label, []))
                             for index, sample in enumerate(samples)}
        predictions = sorted(((float(score), index, list(box))
                              for index, sample in enumerate(samples)
                              for score, box in sample.get("predicted", {}).get(label, [])), reverse=True)
        positives = sum(map(len, expected_by_image.values()))
        used_ap = defaultdict(set)
        ap_points = []
        fixed_tp = fixed_fp = 0
        matched_ious = []
        for score, image_index, box in predictions:
            candidates = [(idx, box_iou(box, expected))
                          for idx, expected in enumerate(expected_by_image[image_index])
                          if idx not in used_ap[image_index]]
            match, overlap = max(candidates, key=lambda row: row[1], default=(-1, 0.0))
            matched = match >= 0 and overlap >= iou_threshold
            ap_points.append(matched)
            if matched:
                used_ap[image_index].add(match)
            if score >= score_threshold:
                if matched:
                    fixed_tp += 1
                    matched_ious.append(overlap)
                else:
                    fixed_fp += 1
        fixed_fn = positives - fixed_tp
        ap50 = _average_precision(ap_points, positives)
        support = positives
        precision = fixed_tp / (fixed_tp + fixed_fp) if fixed_tp + fixed_fp else None
        recall = fixed_tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        row = {"support": support, "predictions": len(predictions),
               "tp": fixed_tp, "fp": fixed_fp, "fn": fixed_fn,
               "precision": round(precision, 6) if precision is not None else None,
               "recall": round(recall, 6) if recall is not None else None,
               "f1": round(f1, 6) if f1 is not None else None,
               "ap50": round(ap50, 6) if ap50 is not None else None,
               "matched_box_iou": round(sum(matched_ious) / len(matched_ious), 6) if matched_ious else None}
        per_class[label] = row
        if ap50 is not None:
            aps.append(ap50)
        for key in aggregate:
            aggregate[key] += row[key]
    precision = aggregate["tp"] / (aggregate["tp"] + aggregate["fp"]) if aggregate["tp"] + aggregate["fp"] else None
    recall = aggregate["tp"] / (aggregate["tp"] + aggregate["fn"]) if aggregate["tp"] + aggregate["fn"] else None
    return {"schema_version": EVALUATION_SCHEMA_VERSION, "split": split, "images": int(images),
            "score_threshold": score_threshold, "iou_threshold": iou_threshold,
            "box_map50": round(sum(aps) / len(aps), 6) if aps else None,
            "precision_50": round(precision, 6) if precision is not None else None,
            "recall_50": round(recall, 6) if recall is not None else None,
            "per_class": per_class, "totals": aggregate}


def _test_source_audit(manifest, has_test):
    if not has_test:
        return {"test_independent_sources": None, "test_source_overlap_groups": [],
                "test_source_tracking_complete": None}
    quality = manifest.get("data_quality") or {}
    declared = quality.get("independent_sources")
    assets = manifest.get("assets", [])
    tracked = all(asset.get("batch_id") for asset in assets)
    train_validation = {asset.get("batch_id") for asset in assets
                        if asset.get("split") in {"train", "val"} and asset.get("batch_id")}
    test = {asset.get("batch_id") for asset in assets
            if asset.get("split") == "test" and asset.get("batch_id")}
    overlap = sorted(train_validation & test)
    if declared is False or overlap:
        independent = False
    elif declared is True:
        independent = True
    else:
        independent = True if tracked else None
    return {"test_independent_sources": independent,
            "test_source_overlap_groups": overlap,
            "test_source_tracking_complete": tracked}


def evaluation_protocol(*, checkpoint, has_test, manifest=None, independent_test=None):
    source = (_test_source_audit(manifest, has_test) if manifest is not None else
              {"test_independent_sources": independent_test,
               "test_source_overlap_groups": [], "test_source_tracking_complete": None})
    return {"schema_version": EVALUATION_SCHEMA_VERSION,
            "selection_split": "val", "selection_checkpoint": checkpoint,
            "test_present": bool(has_test), **source}
