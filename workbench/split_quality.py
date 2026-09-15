"""Class coverage shared by editable projects and immutable dataset versions."""
from collections import Counter

from .splitting import SPLIT_ORDER


def split_class_coverage(assets, assignments=None, *, active_splits=SPLIT_ORDER):
    """Count labels and require every observed class in Train and Validation.

    ``assets`` may use project ``id`` or manifest ``asset_id`` identifiers.
    Unobserved catalog classes are deliberately excluded: a class with no labels
    anywhere cannot be repaired by moving images between dataset splits.
    """
    active = tuple(split for split in SPLIT_ORDER if split in active_splits)
    counts = {split: Counter() for split in SPLIT_ORDER}
    images = Counter()
    totals = Counter()
    for asset in assets:
        asset_id = asset.get("id") or asset.get("asset_id")
        split = assignments.get(asset_id, "") if assignments is not None else asset.get("split", "")
        labels = Counter(shape["label"] for shape in asset.get("shapes", []) if shape.get("label"))
        totals.update(labels)
        if split in counts:
            images[split] += 1
            counts[split].update(labels)
    labels = sorted(totals)
    missing_train = [label for label in labels if not counts["train"][label]]
    missing_evaluation = {split: [label for label in labels if not counts[split][label]]
                          for split in active if split != "train"}
    blockers = [{"code": "train_class_missing", "label": label,
                 "message": f"Train 缺少已有標註的類別「{label}」，模型無法學習此類別",
                 "action": "調整來源群組分配或補充獨立來源，讓 Train 涵蓋所有已有標註的類別"}
                for label in missing_train]
    blockers.extend({"code": "validation_class_missing", "label": label, "split": "val",
                     "message": f"Validation 缺少類別「{label}」，無法可靠選擇模型",
                     "action": "補充獨立來源，讓 Validation 涵蓋所有已有標註的類別"}
                    for label in missing_evaluation.get("val", []))
    warnings = [{"code": "evaluation_class_missing", "label": label, "split": split,
                 "message": f"{split} 缺少類別「{label}」，無法評估此類別在該集合的表現",
                 "action": "補充獨立來源以改善評估覆蓋；請勿只為湊比例拆開同來源群組"}
                for split, missing in missing_evaluation.items() if split != "val" for label in missing]
    return {"ready": not blockers, "blockers": blockers, "warnings": warnings,
            "class_counts": {split: {label: counts[split][label] for label in labels} for split in SPLIT_ORDER},
            "class_totals": dict(sorted(totals.items())),
            "image_counts": {split: images[split] for split in SPLIT_ORDER},
            "missing_train_classes": missing_train, "missing_evaluation_classes": missing_evaluation}
