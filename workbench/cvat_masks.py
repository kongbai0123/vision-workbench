"""Lossless conversion between full-image COCO RLE and CVAT cropped row RLE."""
import numpy as np

from composer_core.geometry import decode_rle, encode_rle


def to_cvat_mask(counts, width, height):
    bitmap = decode_rle(counts, width, height)
    ys, xs = np.nonzero(bitmap)
    if len(xs):
        left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    else:
        left = top = right = bottom = 0
    # Transpose makes the existing column-major encoder traverse CVAT's rows.
    runs = encode_rle(bitmap[top:bottom + 1, left:right + 1].T)
    return runs + [left, top, right, bottom]


def from_cvat_mask(points, width, height):
    if width <= 0 or height <= 0 or width * height > 16777216:
        raise ValueError("CVAT 遮罩影像尺寸無效。")
    if len(points) < 5 or any(type(n) not in (int, float) or not float(n).is_integer() for n in points):
        raise ValueError("CVAT 遮罩 RLE 無效。")
    *counts, left, top, right, bottom = map(int, points)
    if not (0 <= left <= right < width and 0 <= top <= bottom < height):
        raise ValueError("CVAT 遮罩邊界超出影像。")
    crop_width, crop_height = right - left + 1, bottom - top + 1
    crop = decode_rle(counts, crop_height, crop_width).T
    bitmap = np.zeros((height, width), dtype=np.uint8)
    bitmap[top:bottom + 1, left:right + 1] = crop
    return encode_rle(bitmap)
