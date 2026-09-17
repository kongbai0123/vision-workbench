"""Auditable YOLO Seg conversion without modifying source annotations."""
from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np

from composer_core.geometry import decode_rle, encode_rle, shape_polygons
from composer_core.mask_cleanup import enclosed_background_components


DEFAULT_POLICY = {
    "yolo_mask_policy": "repair_tiny_holes",
    "tiny_hole_max_pixels": 16,
    "tiny_hole_total_pixels": 16,
    "tiny_hole_max_ratio": .0001,
    "tiny_hole_max_dimension": 16,
}


def policy_from_config(config=None):
    config = config or {}
    return {key: config.get(key, value) for key, value in DEFAULT_POLICY.items()}


def _hole_components(mask):
    """Return every pixel that an outer YOLO polygon would fill as a hole."""
    return enclosed_background_components(mask, include_points=True)


def _issue(asset, shape, shape_index, **details):
    return {
        "asset_id": asset.get("asset_id"),
        "name": asset.get("name") or asset.get("asset_id"),
        "split": asset.get("split"),
        "width": int(asset["width"]),
        "height": int(asset["height"]),
        "shape_id": shape.get("id"),
        "shape_index": shape_index,
        "label": shape.get("label"),
        **details,
    }


def compatible_shape(asset, shape, shape_index, config=None):
    """Return a Run-copy shape and one repair or blocker description."""
    policy = policy_from_config(config)
    width, height = int(asset["width"]), int(asset["height"])
    if shape.get("type") != "mask":
        try:
            polygons, diagnostics = shape_polygons(shape, width, height, tolerance=0)
        except ValueError as exc:
            return shape, None, _issue(asset, shape, shape_index, code="invalid_geometry", message=str(exc))
        if len(polygons) != 1 or diagnostics.get("pixel_iou", 1) < .999:
            return shape, None, _issue(asset, shape, shape_index, code="lossy_geometry",
                message="標註無法無損表示為單一 YOLO Seg 多邊形", diagnostics=diagnostics)
        return shape, None, None

    mask = decode_rle(shape.get("counts"), width, height) > 0
    foreground = int(np.count_nonzero(mask))
    component_count = int(cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8,
    )[0] - 1)
    holes = _hole_components(mask)
    if component_count != 1:
        return shape, None, _issue(asset, shape, shape_index, code="multiple_components",
            message=f"遮罩包含 {component_count} 個分離區塊，無法無損表示為單一 YOLO Seg 實例",
            components=component_count, holes=[{k: v for k, v in hole.items() if k != "points"} for hole in holes])
    if not holes:
        try:
            polygons, diagnostics = shape_polygons(shape, width, height, tolerance=0)
        except ValueError as exc:
            return shape, None, _issue(asset, shape, shape_index, code="invalid_geometry", message=str(exc))
        if (len(polygons) != 1 or diagnostics.get("holes_omitted")
                or diagnostics.get("pixel_iou", 1) < .999):
            return shape, None, _issue(asset, shape, shape_index, code="lossy_geometry",
                message="遮罩轉為 YOLO Seg 多邊形後的像素 IoU 低於 99.9%", diagnostics=diagnostics)
        return shape, None, None

    public_holes = [{k: v for k, v in hole.items() if k != "points"} for hole in holes]
    total = sum(hole["pixels"] for hole in holes)
    ratio = total / max(1, foreground)
    if policy["yolo_mask_policy"] == "strict":
        return shape, None, _issue(asset, shape, shape_index, code="holes_strict",
            message=f"嚴格無損模式偵測到 {len(holes)} 個封閉孔洞（共 {total} px）", holes=public_holes,
            hole_pixels=total, original_foreground_pixels=foreground, fill_ratio=ratio)
    oversized = [hole for hole in holes if hole["pixels"] > int(policy["tiny_hole_max_pixels"])
                 or hole["bbox"][2] > int(policy["tiny_hole_max_dimension"])
                 or hole["bbox"][3] > int(policy["tiny_hole_max_dimension"])]
    if oversized or total > int(policy["tiny_hole_total_pixels"]) or ratio > float(policy["tiny_hole_max_ratio"]):
        return shape, None, _issue(asset, shape, shape_index, code="holes_above_limit",
            message=f"孔洞超過相容修補門檻（{len(holes)} 個，共 {total} px，占遮罩 {ratio:.5%}）",
            holes=public_holes, hole_pixels=total, original_foreground_pixels=foreground, fill_ratio=ratio,
            limits=policy)

    repaired_mask = mask.copy()
    for hole in holes:
        for x, y in hole["points"]:
            repaired_mask[y, x] = True
    repaired = deepcopy(shape)
    repaired["counts"] = encode_rle(repaired_mask)
    polygons, diagnostics = shape_polygons(repaired, width, height, tolerance=0)
    remaining_holes = _hole_components(repaired_mask)
    if len(polygons) != 1 or remaining_holes or diagnostics.get("pixel_iou", 1) < .999:
        return shape, None, _issue(asset, shape, shape_index, code="repair_still_lossy",
            message="修補微小孔洞後仍無法安全轉成 YOLO Seg 多邊形", holes=public_holes,
            remaining_holes=[{k: v for k, v in hole.items() if k != "points"} for hole in remaining_holes],
            diagnostics=diagnostics)
    repair = _issue(asset, shape, shape_index, code="tiny_holes_repaired",
        message=f"Run 相容副本將修補 {len(holes)} 個微小孔洞（共 {total} px）", holes=public_holes,
        hole_pixels=total, original_foreground_pixels=foreground, fill_ratio=ratio,
        source_annotations_unchanged=True, converted_pixel_iou=foreground / (foreground + total))
    return repaired, repair, None


def analyze_manifest(manifest, config=None):
    policy = policy_from_config(config)
    repairs, blockers = [], []
    shape_count = 0
    for asset in manifest.get("assets", []):
        for index, shape in enumerate(asset.get("shapes", [])):
            shape_count += 1
            _copy, repair, blocker = compatible_shape(asset, shape, index, policy)
            if repair:
                repairs.append(repair)
            if blocker:
                blockers.append(blocker)
    return {
        "schema_version": 1,
        "compatible": not blockers,
        "policy": policy,
        "summary": {
            "assets_scanned": len(manifest.get("assets", [])),
            "shapes_scanned": shape_count,
            "affected_assets": len({item["asset_id"] for item in repairs}),
            "affected_shapes": len(repairs),
            "holes_repaired": sum(len(item.get("holes", [])) for item in repairs),
            "pixels_repaired": sum(item.get("hole_pixels", 0) for item in repairs),
            "blocked_assets": len({item["asset_id"] for item in blockers}),
        },
        "repairs": repairs,
        "blockers": blockers,
        "source_annotations_unchanged": True,
        "metric_label_space": "yolo_compatible_copy" if repairs else "source_annotations",
    }


def blocker_message(report):
    if report.get("compatible"):
        return ""
    first = report["blockers"][0]
    count = report["summary"]["blocked_assets"]
    return (f"YOLO Seg 無法無損或在安全門檻內相容：{count} 張圖片需要處理；"
            f"{first['name']}／{first.get('label') or '未命名標註'}：{first['message']}。請查看相容檢查的位置明細。")
