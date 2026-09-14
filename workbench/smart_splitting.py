"""Previewable, reproducible split allocation with hard source-group constraints."""
from collections import Counter
import hashlib
import json
import math
import random

from .splitting import SPLIT_ORDER, _ratios


def source_groups(assets, overrides=None, strategy="smart"):
    assets = sorted(assets, key=lambda a: a["id"])
    overrides = overrides or {}
    if not isinstance(overrides, dict) or set(overrides) - {a["id"] for a in assets}:
        raise ValueError("手動群組包含不存在或未核准的圖片")
    if any(not isinstance(v, str) or len(v) > 128 for v in overrides.values()):
        raise ValueError("手動群組名稱最多 128 字")
    parent = list(range(len(assets)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, asset in enumerate(assets):
        tokens = ["sha:" + asset["sha256"]] if asset.get("sha256") else []
        source = asset.get("source") or {}
        while isinstance(source, dict):
            if strategy == "smart":
                for key in ("video_sha256", "video_id", "video"):
                    if source.get(key):
                        tokens.append(key + ":" + str(source[key]).replace("\\", "/").casefold())
            source = source.get("original")
        group = overrides.get(asset["id"], "").strip()
        if group:
            tokens.append("manual:" + group)
        elif strategy == "smart" and asset.get("batch_id"):
            tokens.append("batch:" + asset["batch_id"])
        for token in tokens:
            if token in seen:
                parent[find(i)] = find(seen[token])
            else:
                seen[token] = i
    groups = {}
    for i, asset in enumerate(assets):
        groups.setdefault(find(i), []).append(asset)
    result = []
    for members in groups.values():
        ids = sorted(a["id"] for a in members)
        labels = Counter(s["label"] for a in members for s in a.get("shapes", []) if s.get("label"))
        result.append({"id": hashlib.sha256("|".join(ids).encode()).hexdigest()[:16], "asset_ids": ids,
                       "images": len(ids), "classes": dict(labels),
                       "names": sorted({overrides.get(a["id"], "").strip() or a.get("batch_id") or a["id"][:8] for a in members}),
                       "sources": sorted({a.get("batch_id") or "未指定來源" for a in members}),
                       "current": dict(Counter(a.get("split") or "未分配" for a in members))})
    return sorted(result, key=lambda g: g["id"])


def smart_split(assets, options=None):
    options = options or {}
    if not isinstance(options, dict):
        raise ValueError("分割設定必須是物件")
    strategy = options.get("strategy", "smart")
    if strategy not in {"smart", "class_balanced"}:
        raise ValueError("不支援的分割策略")
    seed = options.get("seed", 42)
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("分割種子必須為 0–2147483647 的整數")
    try:
        ratios = _ratios(options.get("ratios", [70, 20, 10]))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("請填入有效的 Train / Validation / Test 比例") from None
    if any(not math.isfinite(x) for x in ratios.values()) or ratios["train"] <= 0:
        raise ValueError("比例必須有限，且 Train 必須大於 0")
    active = [s for s in SPLIT_ORDER if ratios[s] > 0]
    if len(active) < 2:
        raise ValueError("至少需要 Train 及 Validation 或 Test")
    assets = list(assets)
    groups = source_groups(assets, options.get("group_overrides"), strategy)
    if len(groups) < len(active):
        raise ValueError(f"只有 {len(groups)} 個獨立來源群組，無法建立 {len(active)} 個非空集合；請補充獨立來源、整理手動群組或減少集合，系統不會拆開群組")
    locks = options.get("locks") or {}
    if not isinstance(locks, dict) or set(locks) - {g["id"] for g in groups} or any(v not in active for v in locks.values()):
        raise ValueError("群組鎖定已過期或目標集合比例為 0；請重新預覽")
    locks = dict(locks)
    if options.get("preserve_test"):
        for g in groups:
            if g["current"].get("test"):
                if "test" not in active or locks.get(g["id"], "test") != "test":
                    raise ValueError("保留 Test 與手動鎖定衝突")
                locks[g["id"]] = "test"
    totals = Counter()
    for g in groups: totals.update(g["classes"])
    n = len(assets)
    def cost(counts, classes):
        value = 0.
        for split in active:
            value += .7 * ((counts[split] - n * ratios[split]) / max(1, n * ratios[split])) ** 2
            if not counts[split]: value += 1000
            for label, total in totals.items():
                value += ((classes[split][label] - total * ratios[split]) / max(1, total * ratios[split])) ** 2
                if not classes[split][label]: value += 100 if split == "train" else 4
        return value
    rng = random.Random(seed); best = None
    free = [g for g in groups if g["id"] not in locks]
    # Candidate moves update aggregates, not the full dataset, so large imports
    # remain bounded by group count times class count.
    for attempt in range(min(24, max(4, 1200 // len(groups)))):
        ordered = sorted(free, key=lambda g: (-sum(v / totals[k] for k,v in g["classes"].items()), -g["images"], g["id"]))
        if attempt: rng.shuffle(ordered)
        assignments = dict(locks); counts = Counter(); classes = {s: Counter() for s in active}
        def place(group, split, direction):
            counts[split] += direction * group["images"]
            for label, value in group["classes"].items(): classes[split][label] += direction * value
        for group in groups:
            if group["id"] in locks: place(group,locks[group["id"]],1)
        def choose(group, preferred=None):
            choices=[]
            for split in active:
                place(group,split,1);value=cost(counts,classes);place(group,split,-1)
                choices.append((value,split != preferred,SPLIT_ORDER.index(split),split))
            return min(choices)[-1]
        for group in ordered:
            split=choose(group);assignments[group["id"]]=split;place(group,split,1)
        for _ in range(3):
            changed=False
            for group in ordered:
                old=assignments[group["id"]];place(group,old,-1)
                split=choose(group,old);place(group,split,1);assignments[group["id"]]=split
                changed |= old != split
            if not changed: break
        value=cost(counts,classes)
        if all(counts[split] for split in active) and (best is None or value < best[0]):
            best = value, assignments, counts, classes
    if best is None:
        raise ValueError("群組鎖定或資料量使集合無法非空；請解除部分鎖定或補充資料")
    _, assignments, counts, classes = best
    warnings = []
    for label in sorted(totals):
        missing = [s for s in active if not classes[s][label]]
        if missing: warnings.append(f"類別「{label}」未涵蓋 {' / '.join(missing)}；保持群組完整，建議補充獨立來源")
    if strategy != "smart": warnings.append("類別平衡模式可能拆開拍攝來源，請確認圖片彼此獨立")
    untracked = sum(not a.get("batch_id") and not a.get("source") for a in assets)
    if untracked: warnings.append(f"{untracked} 張圖片缺少來源資訊，請補定義手動群組")
    output = {aid: assignments[g["id"]] for g in groups for aid in g["asset_ids"]}
    for g in groups:
        g["proposed"] = assignments[g["id"]]; g["locked"] = g["id"] in locks
    return {"algorithm_version": 1, "strategy": strategy, "seed": seed, "options": options,
            "assignments": output, "groups": groups, "warnings": warnings,
            "image_counts": {s: counts[s] for s in SPLIT_ORDER},
            "ratios": {s: ratios[s] * 100 for s in SPLIT_ORDER},
            "actual_ratios": {s: counts[s] / n * 100 for s in SPLIT_ORDER},
            "class_counts": {s: dict(classes.get(s, {})) for s in SPLIT_ORDER},
            "class_totals": dict(totals), "changed": sum(a.get("split") != output[a["id"]] for a in assets)}
