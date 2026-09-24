"""Previewable, reproducible split allocation with hard source-group constraints."""
from collections import Counter
from itertools import combinations
import hashlib
import json
import math
import random

from .splitting import SPLIT_ORDER, _ratios
from .split_quality import split_class_coverage


def _covered_seed(groups, locks, active, ordered):
    """Find a feasible starting point before optimizing soft balance targets.

    All unlocked groups can initially go to Train. At most two groups then need
    reserving for empty evaluation splits. Trying these reservations completely
    decides feasibility for the hard nonempty-split and Train-coverage rules.
    """
    assignments = {g["id"]: locks.get(g["id"], "train") for g in groups}
    train = [g for g in groups if assignments[g["id"]] == "train"]
    totals = Counter()
    train_classes = Counter()
    for group in groups:
        totals.update(group["classes"])
    for group in train:
        train_classes.update(group["classes"])
    train_images = sum(g["images"] for g in train)
    if not train_images or any(not train_classes[label] for label in totals):
        return None
    occupied = set(assignments.values())
    reserve = [split for split in active if split != "train" and split not in occupied]
    if not reserve:
        return assignments

    def removable(*selected):
        if sum(g["images"] for g in selected) >= train_images:
            return False
        removed = Counter()
        for group in selected:
            removed.update(group["classes"])
        return all(train_classes[label] > count for label, count in removed.items())

    candidates = [g for g in ordered if removable(g)]
    selected = None
    if len(reserve) == 1 and candidates:
        selected = candidates[:1]
    elif len(reserve) == 2:
        for index, first in enumerate(candidates):
            second = next((g for g in candidates[index + 1:] if removable(first, g)), None)
            if second is not None:
                selected = [first, second]
                break
    if selected is None:
        return None
    for group, split in zip(selected, reserve):
        assignments[group["id"]] = split
    return assignments


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
        presence, pairs = Counter(), Counter()
        for asset in members:
            present = sorted({s['label'] for s in asset.get('shapes', []) if s.get('label')})
            presence.update(present)
            pairs.update(json.dumps(pair, ensure_ascii=False) for pair in combinations(present, 2))
        result.append({"id": hashlib.sha256("|".join(ids).encode()).hexdigest()[:16], "asset_ids": ids,
                       "images": len(ids), "classes": dict(labels),
                       "class_images": dict(presence), "class_pairs": dict(pairs),
                       "names": sorted({overrides.get(a["id"], "").strip() or a.get("batch_id") or a["id"][:8] for a in members}),
                       "sources": sorted({a.get("batch_id") or "未指定來源" for a in members}),
                       "current": dict(Counter(a.get("split") or "未分配" for a in members))})
    return sorted(result, key=lambda g: g["id"])


def smart_split(assets, options=None):
    options = dict(options or {})
    if not isinstance(options, dict):
        raise ValueError("分割設定必須是物件")
    purpose_explicit = 'purpose' in options
    purpose = options.get('purpose')
    if purpose is None:
        purpose = 'experimental' if options.get('strategy') == 'random_loose' else 'formal'
    if purpose not in {'formal', 'reviewed_independent', 'experimental', 'all_train'}:
        raise ValueError('不支援的資料用途')
    strategy = options.get("strategy", "smart")
    if purpose in {'reviewed_independent', 'formal'} and strategy == 'random_loose':
        strategy = 'multilabel'
    elif purpose == 'experimental':
        strategy = 'random_loose'
    if strategy not in {"smart", "class_balanced", "multilabel", "random_loose"}:
        raise ValueError("不支援的分割策略")
    isolate = options.get('source_isolation', strategy != 'class_balanced')
    if type(isolate) is not bool:
        raise ValueError('來源隔離必須為布林值')
    loose = purpose == 'experimental'
    if loose:
        isolate = False
    elif purpose == 'reviewed_independent':
        isolate = False
    elif purpose == 'formal' and purpose_explicit:
        isolate = True
    mode = options.get('balance_mode', 'hybrid')
    if mode not in {'hybrid', 'presence', 'instances', 'cooccurrence'}:
        raise ValueError('不支援的多類別平衡目標')
    multilabel = strategy == 'multilabel'
    seed = options.get("seed", 42)
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("分割種子必須為 0–2147483647 的整數")
    try:
        ratios = _ratios([100, 0, 0] if purpose == 'all_train' else options.get("ratios", [70, 20, 10]))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("請填入有效的 Train / Validation / Test 比例") from None
    if any(not math.isfinite(x) for x in ratios.values()) or ratios["train"] <= 0:
        raise ValueError("比例必須有限，且 Train 必須大於 0")
    active = [s for s in SPLIT_ORDER if ratios[s] > 0]
    if ratios["val"] <= 0 and not loose and purpose != 'all_train':
        raise ValueError("Validation 比例必須大於 0；Test 不可替代 Validation")
    assets = list(assets)
    if not assets or any(not a.get('id') for a in assets) or len({a['id'] for a in assets}) != len(assets):
        raise ValueError('需要已核准圖片，且每張圖片必須有唯一 id')
    groups = source_groups(assets, options.get("group_overrides"), 'smart' if isolate else 'class_balanced')
    if purpose == 'all_train':
        output = {asset['id']: 'train' for asset in assets}
        totals = Counter(shape['label'] for asset in assets for shape in asset.get('shapes', []) if shape.get('label'))
        images = Counter()
        pairs = Counter()
        for asset in assets:
            present = sorted({shape['label'] for shape in asset.get('shapes', []) if shape.get('label')})
            images.update(present)
            pairs.update(json.dumps(pair, ensure_ascii=False) for pair in combinations(present, 2))
        for group in groups:
            group['proposed'] = 'train'; group['locked'] = True
        quality = split_class_coverage(assets, output, active_splits=['train'], policy='all_train')
        warnings = ['全部圖片只用於最終訓練；沒有獨立 Validation／Test，不能產生可比較的泛化評估']
        return {'algorithm_version': 6, 'purpose': purpose, 'strategy': 'all_train', 'seed': seed,
                'options': options, 'source_isolation': False, 'balance_mode': mode,
                'class_image_totals': dict(images),
                'class_image_counts': {'train': dict(images), 'val': {}, 'test': {}},
                'class_source_counts': dict(Counter(label for group in source_groups(assets) for label in group['classes'])),
                'class_group_counts': dict(Counter(label for group in groups for label in group['classes'])),
                'scarcity': [], 'pair_distribution': [],
                'ratio_deviation': {'train': 0, 'val': 0, 'test': 0},
                'assignments': output, 'groups': groups, 'warnings': warnings,
                'ready': quality['ready'], 'blockers': quality['blockers'], 'coverage': quality,
                'image_counts': {'train': len(assets), 'val': 0, 'test': 0},
                'ratios': {'train': 100, 'val': 0, 'test': 0},
                'actual_ratios': {'train': 100, 'val': 0, 'test': 0},
                'class_counts': {'train': dict(totals), 'val': {}, 'test': {}},
                'class_totals': dict(totals),
                'changed': sum(asset.get('split') != 'train' for asset in assets)}
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
    presence_totals, pair_totals = Counter(), Counter()
    for group in groups:
        presence_totals.update(group['class_images']); pair_totals.update(group['class_pairs'])
    # Only recurring pairs are useful balance targets; keep optimization bounded.
    selected_pairs = dict(sorted(((k,v) for k,v in pair_totals.items() if v >= len(active)),
                                 key=lambda item: (-item[1], item[0]))[:256])
    carrier_counts = Counter(label for group in groups for label in group['classes'])
    n = len(assets)
    def cost(counts, classes, presence, pairs):
        value = 0.
        for split in active:
            value += .7 * ((counts[split] - n * ratios[split]) / max(1, n * ratios[split])) ** 2
            if not counts[split]: value += 1000
            for label, total in totals.items():
                weight = 1 if not multilabel or mode == 'instances' else .35 if mode == 'hybrid' else 0
                value += weight * ((classes[split][label] - total * ratios[split]) / max(1, total * ratios[split])) ** 2
                if not classes[split][label]:
                    value += (10000 if split in {'train', 'val'} else 100) if multilabel else (100 if split == 'train' else 4)
                if multilabel and mode != 'instances':
                    target = presence_totals[label] * ratios[split]
                    value += ((presence[split][label] - target) / max(1, target)) ** 2
            if multilabel and mode in {'hybrid', 'cooccurrence'} and selected_pairs:
                weight = (.25 if mode == 'hybrid' else .75) * max(1, len(totals)) / len(selected_pairs)
                for pair,total in selected_pairs.items():
                    target = total * ratios[split]
                    value += weight * ((pairs[split][pair] - target) / max(1, target)) ** 2
        return value
    rng = random.Random(seed); best = None; fallback = None
    free = [g for g in groups if g["id"] not in locks]
    if loose:
        # Shuffle whole image units, then fill image-count deficits. Labels do
        # not influence allocation; duplicate/manual groups and locks survive.
        ordered = list(free)
        rng.shuffle(ordered)
        assignments = dict(locks)
        counts = Counter()
        classes = {s: Counter() for s in active}
        for group in groups:
            if group['id'] in locks:
                counts[locks[group['id']]] += group['images']
        for index, group in enumerate(ordered):
            empty = [s for s in active if not counts[s]]
            choices = empty if len(ordered)-index <= len(empty) else active
            split = max(choices, key=lambda s: (n*ratios[s]-counts[s], -SPLIT_ORDER.index(s)))
            assignments[group['id']] = split
            counts[split] += group['images']
        if all(counts[s] for s in active):
            for group in groups:
                classes[assignments[group['id']]].update(group['classes'])
            best = 0, assignments, counts, classes
    # Candidate moves update aggregates, not the full dataset, so large imports
    # remain bounded by group count times class count.
    for attempt in range(0 if loose else min(24, max(4, 1200 // len(groups)))):
        ordered = sorted(free, key=lambda g: (-sum(v / totals[k] for k,v in g["classes"].items()), -g["images"], g["id"]))
        if attempt: rng.shuffle(ordered)
        if multilabel:
            ordered.sort(key=lambda g: min((carrier_counts[k] for k in g['classes']), default=len(groups)+1))
        covered = _covered_seed(groups, locks, active, ordered)
        assignments = covered or dict(locks); counts = Counter(); classes = {s: Counter() for s in active}
        presence = {s: Counter() for s in active}; pairs = {s: Counter() for s in active}
        def place(group, split, direction):
            counts[split] += direction * group["images"]
            for label, value in group["classes"].items(): classes[split][label] += direction * value
            for label, value in group['class_images'].items(): presence[split][label] += direction * value
            for pair, value in group['class_pairs'].items():
                if pair in selected_pairs: pairs[split][pair] += direction * value
        for group in groups:
            if group["id"] in assignments: place(group,assignments[group["id"]],1)
        def choose(group, preferred=None):
            choices=[]
            for split in active:
                place(group,split,1)
                valid = covered is None or (all(counts[s] for s in active) and
                                            all(classes["train"][label] for label in totals))
                value=cost(counts,classes,presence,pairs);place(group,split,-1)
                if valid: choices.append((value,split != preferred,SPLIT_ORDER.index(split),split))
            return min(choices)[-1]
        if covered is None:
            for group in ordered:
                split=choose(group);assignments[group["id"]]=split;place(group,split,1)
        for _ in range(3):
            changed=False
            for group in ordered:
                old=assignments[group["id"]];place(group,old,-1)
                split=choose(group,old);place(group,split,1);assignments[group["id"]]=split
                changed |= old != split
            if not changed: break
        if multilabel:
            # Pair swaps escape local optima that single-group moves cannot fix.
            candidates = ordered if len(ordered) <= 48 else rng.sample(ordered, 48)
            current_cost = cost(counts, classes, presence, pairs)
            for left, right in combinations(candidates, 2):
                a, b = assignments[left['id']], assignments[right['id']]
                if a == b: continue
                place(left,a,-1); place(right,b,-1); place(left,b,1); place(right,a,1)
                valid = all(counts[s] for s in active) and all(classes['train'][k] for k in totals)
                candidate_cost = cost(counts, classes, presence, pairs)
                if valid and candidate_cost < current_cost - 1e-9:
                    assignments[left['id']], assignments[right['id']] = b, a
                    current_cost = candidate_cost
                else:
                    place(left,b,-1); place(right,a,-1); place(left,a,1); place(right,b,1)
        value=cost(counts,classes,presence,pairs)
        nonempty = all(counts[split] for split in active)
        if nonempty and (fallback is None or value < fallback[0]):
            fallback = value, assignments, counts, classes
        complete_training_validation = (
            all(classes["train"][label] for label in totals)
            and all(classes["val"][label] for label in totals)
        )
        if nonempty and complete_training_validation and (best is None or value < best[0]):
            best = value, assignments, counts, classes
    best = best or fallback
    if best is None:
        raise ValueError("群組鎖定或資料量使集合無法非空；請解除部分鎖定或補充資料")
    _, assignments, counts, classes = best
    output = {aid: assignments[g["id"]] for g in groups for aid in g["asset_ids"]}
    quality = split_class_coverage(assets, output, active_splits=active, policy=purpose)
    warnings = [item["message"] for item in quality["warnings"]]
    for blocker in quality["blockers"]:
        label = blocker["label"]
        carriers = [g for g in groups if g["classes"].get(label)]
        blocker["source_group_count"] = len(carriers)
        if blocker["code"] == "train_class_missing":
            locked_out = all(locks.get(g["id"]) in {"val", "test"} for g in carriers)
            if locked_out:
                blocker["action"] = "此類別的所有來源群組均鎖定在評估集合；請解除鎖定或補充可用於 Train 的獨立來源"
            elif len(carriers) == 1:
                blocker["action"] = "此類別只有一個來源群組；無法在保持群組完整時同時提供 Train 與 Validation，請補充另一個獨立來源"
        elif blocker["code"] == "validation_class_missing":
            locked_out = all(locks.get(g["id"]) in {"train", "test"} for g in carriers)
            if locked_out:
                blocker["action"] = "此類別的所有來源群組均鎖定在 Train 或 Test；請解除鎖定或補充可用於 Validation 的獨立來源"
            elif len(carriers) == 1:
                blocker["action"] = "此類別只有一個來源群組；無法在保持群組完整時同時提供 Train 與 Validation，請補充另一個獨立來源"
    if purpose == 'reviewed_independent':
        warnings.append('圖片層級平衡分配；完全相同的圖片仍保持同一集合')
    elif not isolate:
        warnings.append("未啟用來源隔離：同次拍攝可能跨集合，評估結果不代表新來源表現")
    untracked = sum(not a.get("batch_id") and not a.get("source") for a in assets)
    if untracked: warnings.append(f"{untracked} 張圖片缺少來源資訊，請補定義手動群組")
    for g in groups:
        g["proposed"] = assignments[g["id"]]; g["locked"] = g["id"] in locks
    class_images = {s: Counter() for s in SPLIT_ORDER}
    pair_counts = {s: Counter() for s in SPLIT_ORDER}
    for group in groups:
        split = assignments[group['id']]
        class_images[split].update(group['class_images']); pair_counts[split].update(group['class_pairs'])
    source_counts = Counter(label for group in source_groups(assets) for label in group['classes'])
    scarcity = []
    for label in sorted(totals):
        details = []
        if presence_totals[label] < len(active): details.append('含此類別的圖片不足以覆蓋所有集合')
        if source_counts[label] < len(active): details.append('來源群組不足以在各集合提供獨立來源')
        if class_images['train'][label] < 5: details.append('Train 少於 5 張，學習樣本有限')
        for split in active:
            if split != 'train' and class_images[split][label] < 5: details.append(f'{split} 少於 5 張，評估可能不穩定')
        if details: scarcity.append({'label': label, 'messages': details})
    return {"algorithm_version": 6, "purpose": purpose, "strategy": strategy, "seed": seed, "options": options,
            'source_isolation': isolate, 'balance_mode': mode if multilabel else 'instances',
            'class_image_totals': dict(presence_totals),
            'class_image_counts': {s: dict(class_images[s]) for s in SPLIT_ORDER},
            'class_source_counts': dict(source_counts), 'class_group_counts': dict(carrier_counts),
            'scarcity': scarcity,
            'pair_distribution': [{'labels': json.loads(pair), 'total': total,
                                   'counts': {s: pair_counts[s][pair] for s in SPLIT_ORDER}} for pair,total in selected_pairs.items()],
            'ratio_deviation': {s: (counts[s]/n-ratios[s])*100 for s in SPLIT_ORDER},
            "assignments": output, "groups": groups, "warnings": warnings,
            "ready": quality["ready"], "blockers": quality["blockers"], "coverage": quality,
            "image_counts": {s: counts[s] for s in SPLIT_ORDER},
            "ratios": {s: ratios[s] * 100 for s in SPLIT_ORDER},
            "actual_ratios": {s: counts[s] / n * 100 for s in SPLIT_ORDER},
            "class_counts": {s: dict(classes.get(s, {})) for s in SPLIT_ORDER},
            "class_totals": dict(totals), "changed": sum(a.get("split") != output[a["id"]] for a in assets)}
