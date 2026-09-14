"""Deterministic class-count stratification for training dataset splits."""
from __future__ import annotations

from collections import Counter
import math


SPLIT_ORDER = ("train", "val", "test")


def _ratios(values):
    if isinstance(values, dict):
        raw = {name: float(values.get(name, 0)) for name in SPLIT_ORDER}
    else:
        items = list(values)
        if len(items) != 3:
            raise ValueError("分割比例必須包含 train、val、test")
        raw = dict(zip(SPLIT_ORDER, map(float, items)))
    if any(not math.isfinite(value) or value < 0 for value in raw.values()) or sum(raw.values()) <= 0:
        raise ValueError("分割比例必須為非負數且總和大於 0")
    total = sum(raw.values())
    return {name: raw[name] / total for name in SPLIT_ORDER}


def stratified_split(assets, ratios=(0.7, 0.2, 0.1)):
    """Assign assets by per-class instance counts without any randomness.

    Exact duplicate images (same SHA-256) remain an atomic unit. Source batches
    are deliberately not atomic because a class may occur in only one capture
    batch; keeping that batch intact would make class coverage impossible.
    """
    assets = list(assets)
    weights = _ratios(ratios)
    active = [name for name in SPLIT_ORDER if weights[name] > 0]
    if len(assets) < len(active):
        raise ValueError(f"至少需要 {len(active)} 張已核准影像才能建立 {' / '.join(active)}")
    ids = [str(asset.get("id") or "") for asset in assets]
    if any(not asset_id for asset_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("每張圖片必須有唯一 id 才能自動分割")

    grouped = {}
    for asset in assets:
        key = f"sha:{asset.get('sha256')}" if asset.get("sha256") else f"id:{asset['id']}"
        unit = grouped.setdefault(key, {"key": key, "ids": [], "images": 0, "classes": Counter()})
        unit["ids"].append(asset["id"])
        unit["images"] += 1
        unit["classes"].update(
            str(shape.get("label")) for shape in asset.get("shapes", []) if shape.get("label")
        )
    units = list(grouped.values())
    if len(units) < len(active):
        raise ValueError("相同影像不可跨集合；可用的唯一影像不足以建立所有分割")

    totals = Counter()
    for unit in units:
        totals.update(unit["classes"])
    targets = {
        split: {label: total * weights[split] for label, total in totals.items()}
        for split in SPLIT_ORDER
    }
    image_targets = {split: len(assets) * weights[split] for split in SPLIT_ORDER}
    class_counts = {split: Counter() for split in SPLIT_ORDER}
    image_counts = Counter()
    unit_split = {}

    def cost():
        value = 0.0
        for split in active:
            value += 0.35 * ((image_counts[split] - image_targets[split]) ** 2) / max(1.0, image_targets[split])
            for label in totals:
                value += ((class_counts[split][label] - targets[split][label]) ** 2) / max(1.0, targets[split][label])
        return value

    def place(unit, split, direction=1):
        image_counts[split] += direction * unit["images"]
        for label, count in unit["classes"].items():
            class_counts[split][label] += direction * count

    def rarity(unit):
        return sum(count / max(1, totals[label]) for label, count in unit["classes"].items())

    ordered = sorted(units, key=lambda unit: (-rarity(unit), -sum(unit["classes"].values()), unit["key"]))
    for unit in ordered:
        choices = []
        for index, split in enumerate(active):
            place(unit, split)
            score = cost()
            deficit = image_targets[split] - image_counts[split]
            place(unit, split, -1)
            choices.append((score, -deficit, index, split))
        split = min(choices)[-1]
        unit_split[unit["key"]] = split
        place(unit, split)

    # Small datasets can leave a split empty after greedy placement. Move the
    # least costly unit so every requested split remains usable.
    for empty in [split for split in active if image_counts[split] == 0]:
        moves = []
        for unit in ordered:
            source = unit_split[unit["key"]]
            if image_counts[source] <= unit["images"]:
                continue
            place(unit, source, -1)
            place(unit, empty)
            score = cost()
            place(unit, empty, -1)
            place(unit, source)
            moves.append((score, unit["key"], unit, source))
        if not moves:
            raise ValueError("資料量不足，無法建立非空的 train / val / test")
        _, _, unit, source = min(moves)
        place(unit, source, -1)
        place(unit, empty)
        unit_split[unit["key"]] = empty

    assignments = {}
    for unit in units:
        for asset_id in unit["ids"]:
            assignments[asset_id] = unit_split[unit["key"]]
    warnings = []
    for label in sorted(totals):
        supporting_units = sum(bool(unit["classes"].get(label)) for unit in units)
        if supporting_units < len(active):
            warnings.append(f"類別「{label}」只出現在 {supporting_units} 張唯一影像，無法涵蓋所有集合")
    report = {
        "strategy": "class_count_stratified",
        "deterministic": True,
        "ratios": {split: round(weights[split] * 100, 6) for split in SPLIT_ORDER},
        "image_counts": {split: image_counts[split] for split in SPLIT_ORDER},
        "class_totals": dict(sorted(totals.items())),
        "class_counts": {
            split: {label: class_counts[split][label] for label in sorted(totals)}
            for split in SPLIT_ORDER
        },
        "warnings": warnings,
    }
    return assignments, report
