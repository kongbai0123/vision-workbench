"""Mask conversion core adapted from Dataset Composer's sam2_adaptive.py.

The canonical representation is full-image, column-major COCO RLE. Polygon
conversion always starts from that canonical mask and records the resulting loss.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def decode_rle(counts, width: int, height: int) -> np.ndarray:
    """Decode COCO uncompressed or compressed counts without an external model."""
    if isinstance(counts, (str, bytes)):
        encoded = counts.decode('ascii') if isinstance(counts, bytes) else counts
        decoded, index = [], 0
        while index < len(encoded):
            value, shift, more = 0, 0, True
            while more:
                if index >= len(encoded) or shift > 60:
                    raise ValueError('壓縮 RLE 不完整')
                code = ord(encoded[index]) - 48
                index += 1
                if not 0 <= code <= 63:
                    raise ValueError('壓縮 RLE 字元無效')
                value |= (code & 31) << shift
                more = bool(code & 32)
                shift += 5
                if not more and code & 16:
                    value |= -1 << shift
            if len(decoded) > 2:
                value += decoded[-2]
            decoded.append(value)
        counts = decoded
    if (not isinstance(counts, list) or width <= 0 or height <= 0
            or width * height > 200_000_000
            or any(type(n) is not int or n < 0 for n in counts)
            or sum(counts) != width * height):
        raise ValueError('RLE counts 與圖片尺寸不一致')
    flat = np.zeros(width * height, dtype=np.uint8)
    offset = 0
    for index, length in enumerate(counts):
        if index % 2:
            flat[offset:offset + length] = 255
        offset += length
    return flat.reshape((height, width), order='F').copy(order='C')


def encode_rle(mask: np.ndarray) -> list[int]:
    flat = (mask.reshape(-1, order='F') > 0).astype(np.uint8)
    if not flat.size:
        raise ValueError('空白影像')
    boundaries = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.concatenate(([0], boundaries, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return counts


def validate_shape(shape: dict, width: int, height: int) -> None:
    if not isinstance(shape, dict):
        raise ValueError('標註必須是物件')
    if not isinstance(shape.get('id'), str) or not shape['id']:
        raise ValueError('標註缺少穩定 ID')
    label = shape.get('label')
    if not isinstance(label, str) or not label.strip() or '\n' in label or '\r' in label:
        raise ValueError('標註類別不可為空或包含換行')
    kind = shape.get('type')
    number = lambda n: type(n) in (float, int) and math.isfinite(n)
    if kind == 'mask':
        mask = decode_rle(shape.get('counts'), width, height)
        if not np.any(mask):
            raise ValueError('Mask 沒有前景')
        return
    if kind in {'polygon', 'obb', 'linestrip', 'point'}:
        points = shape.get('points')
        minimum = 1 if kind == 'point' else 2 if kind == 'linestrip' else 3
        if (not isinstance(points, list) or len(points) < minimum
                or (kind == 'point' and len(points) != 1)
                or (kind == 'obb' and len(points) != 4)):
            raise ValueError('標註頂點數量無效')
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 or
               not all(number(v) for v in p) or not 0 <= p[0] <= width or
               not 0 <= p[1] <= height for p in points):
            raise ValueError('標註頂點超出圖片或不是有限數值')
        if kind in {'polygon', 'obb'} and abs(cv2.contourArea(np.asarray(points, np.float32))) <= 0:
            raise ValueError('多邊形面積必須大於零')
        return
    values = [shape.get(k) for k in ('x', 'y', 'width', 'height')]
    if kind != 'rectangle' or not all(number(v) for v in values):
        raise ValueError(f'不支援的標註類型或座標：{kind}')
    x, y, w, h = values
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
        raise ValueError('矩形超出圖片或面積為零')


def bounds(shape: dict, width: int, height: int) -> list[float]:
    if shape['type'] == 'mask':
        return list(cv2.boundingRect(decode_rle(shape['counts'], width, height)))
    if shape['type'] == 'rectangle':
        return [shape[k] for k in ('x', 'y', 'width', 'height')]
    points = shape['points']
    x, y = min(p[0] for p in points), min(p[1] for p in points)
    return [x, y, max(p[0] for p in points) - x, max(p[1] for p in points) - y]


def shape_polygons(shape: dict, width: int, height: int, tolerance=0) -> tuple[list, dict]:
    """Return polygons plus explicit loss diagnostics (never edit the input)."""
    kind = shape['type']
    if kind == 'rectangle':
        x, y, w, h = bounds(shape, width, height)
        return [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], {}
    if kind in {'polygon', 'obb'}:
        return [shape['points']], {}
    if kind != 'mask':
        raise ValueError(f'{kind} 無法表示為面積多邊形')
    mask = decode_rle(shape['counts'], width, height)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outer, holes, original_points = [], 0, 0
    if hierarchy is None:
        raise ValueError('Mask 沒有輪廓')
    tiny_components = 0
    for contour, relation in zip(contours, hierarchy[0]):
        original_points += len(contour)
        if int(relation[3]) >= 0:
            holes += 1
            continue
        # Reused Composer two-stage simplification, always from the source mask.
        compact = cv2.approxPolyDP(contour, 0.0, True)
        if len(compact) < 3:
            compact = contour
        simplified = cv2.approxPolyDP(compact, float(tolerance), True) if tolerance > 0 else compact
        if len(simplified) < 3:
            simplified = compact
        if len(simplified) < 3 or cv2.contourArea(simplified) == 0:
            tiny_components += 1
            continue
        outer.append(simplified.reshape(-1, 2).astype(float).tolist())
    if tiny_components:
        raise ValueError('Mask 含有無法以 polygon 表示的單像素／細線區塊；請使用 Native 或 COCO')
    if not outer:
        raise ValueError('Mask 無法產生有效多邊形')
    approximate = np.zeros_like(mask)
    cv2.fillPoly(approximate, [np.asarray(p, np.int32) for p in outer], 255)
    intersection = int(np.count_nonzero((mask > 0) & (approximate > 0)))
    union = int(np.count_nonzero((mask > 0) | (approximate > 0)))
    return outer, {'holes_omitted': holes, 'components': len(outer),
                   'original_points': original_points,
                   'output_points': sum(len(p) for p in outer),
                   'pixel_iou': intersection / union if union else 1.0,
                   'rle_to_polygon': True}
