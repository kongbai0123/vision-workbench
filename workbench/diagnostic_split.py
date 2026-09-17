"""Explicit, limited-data pipeline verification; never an independent test split."""
from collections import defaultdict
import math

from .split_quality import split_class_coverage


def diagnostic_temporal_split(assets, gap_images=1):
    """Use consecutive capture blocks with excluded boundary images per class.

    This intentionally reuses capture sessions and must carry a diagnostic label.
    It makes no claim that different blocks are statistically independent.
    """
    if type(gap_images) is not int or gap_images < 1:
        raise ValueError("流程驗證的區段邊界至少需保留一張間隔圖片")
    assets = list(assets)
    if not assets:
        raise ValueError("沒有圖片可供流程驗證")
    groups = defaultdict(list)
    for asset in assets:
        labels = {s.get("label") for s in asset.get("shapes", []) if s.get("label")}
        if len(labels) != 1 or not asset.get("batch_id"):
            raise ValueError("時間區段流程驗證要求每張圖片有來源批次及單一類別")
        groups[(asset["batch_id"], next(iter(labels)))].append(asset)
    assignments, excluded, blocks = {}, [], []
    for (batch, label), members in sorted(groups.items()):
        members.sort(key=lambda a: (a.get("name", ""), a.get("asset_id", a.get("id", ""))))
        usable = len(members) - 2 * gap_images
        evaluation = max(2, math.floor(usable * .2))
        train_count = usable - 2 * evaluation
        if train_count < 2:
            raise ValueError(f"類別 {label} 的來源 {batch} 圖片不足以保留 Train、Val、Test 與區段間隔")
        position = 0
        for split, count in (("train", train_count), ("val", evaluation), ("test", evaluation)):
            selected = members[position:position + count]
            ids = [a.get("asset_id", a.get("id")) for a in selected]
            assignments.update({aid: split for aid in ids})
            blocks.append({"batch_id": batch, "label": label, "split": split, "asset_ids": ids})
            position += count
            if split != "test":
                excluded.extend({"asset_id": a.get("asset_id", a.get("id")), "name": a.get("name"),
                                 "reason": "temporal_boundary_gap"} for a in members[position:position + gap_images])
                position += gap_images
    selected = [a for a in assets if a.get("asset_id", a.get("id")) in assignments]
    hashes = defaultdict(set)
    for a in selected:
        if a.get("sha256"):
            hashes[a["sha256"]].add(assignments[a.get("asset_id", a.get("id"))])
    if any(len(splits) > 1 for splits in hashes.values()):
        raise ValueError("時間區段存在跨集合的完全相同圖片，請先整理重複來源")
    coverage = split_class_coverage(selected, assignments)
    if not coverage["ready"] or coverage["warnings"]:
        raise ValueError("流程驗證仍缺少類別覆蓋，無法建立資料版本")
    return {"assignments": assignments, "blocks": blocks, "coverage": coverage,
            "data_quality": {"purpose": "diagnostic", "independent_sources": False,
                             "method": "temporal_blocks_with_gap", "gap_images": gap_images,
                             "excluded_assets": excluded,
                             "warnings": ["流程驗證：同拍攝批次跨越集合，區段間隔不能保證場景獨立；分數不代表新場景泛化能力。"]}}
