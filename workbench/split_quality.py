"""Class coverage shared by editable projects and immutable dataset versions."""
from collections import Counter

from .splitting import SPLIT_ORDER


def split_purpose(plan, assets):
    """Return the persisted policy only when it matches the exact allocation.

    This prevents a stale (or copied) permissive plan from weakening coverage
    checks after somebody edits the project allocation.
    """
    if not plan or plan.get('assignments') != {
            a.get('id') or a.get('asset_id'): a.get('split') for a in assets}:
        return 'formal'
    purpose = plan.get('purpose')
    if purpose in {'formal', 'reviewed_independent', 'experimental', 'all_train'}:
        return purpose
    return 'experimental' if plan.get('strategy') == 'random_loose' else 'formal'


def loose_split_applies(plan, assets):
    """Backward-compatible predicate for an exact experimental allocation."""
    return split_purpose(plan, assets) == 'experimental'


def split_class_coverage(assets, assignments=None, *, active_splits=SPLIT_ORDER,
                         loose=False, policy=None):
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
    policy = policy or ('experimental' if loose else 'formal')
    if policy == 'reviewed_independent':
        validation = [item for item in blockers if item['code'] == 'validation_class_missing']
        blockers = [item for item in blockers if item['code'] != 'validation_class_missing']
        warnings = validation + warnings
        if validation:
            warnings.append({'code': 'reviewed_validation_gap',
                             'message': '已確認圖片彼此獨立，但 Validation 類別不完整；可訓練，類別指標須個別解讀',
                             'action': '增加 Validation 樣本可改善評估覆蓋'})
    elif policy == 'all_train':
        blockers = [item for item in blockers if item['code'] == 'train_class_missing']
        warnings = []
    elif policy == 'experimental':
        warnings = blockers + warnings
        blockers = []
        warnings.append({'code': 'loose_split', 'message': '寬鬆分割：缺少類別僅警告；Train 未見類別無法學習，來源可能跨集合，結果僅供實驗參考',
                         'action': '正式評估請補充獨立來源並使用多類別平衡分割'})
    return {"ready": not blockers, "blockers": blockers, "warnings": warnings,
            "class_counts": {split: {label: counts[split][label] for label in labels} for split in SPLIT_ORDER},
            "class_totals": dict(sorted(totals.items())),
            "image_counts": {split: images[split] for split in SPLIT_ORDER},
            "missing_train_classes": missing_train, "missing_evaluation_classes": missing_evaluation}
