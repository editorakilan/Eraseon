"""
Letterbox Black Bars & Container Aspect-Ratio Badge Module.
Handles full-width top/bottom letterbox bars via fast directional edge extrapolation
(zero checkerboard/mosaic, zero CPU stall) and dark aspect badges via logo_module.
"""

import cv2
import numpy as np
from backend.watermark_engine.modules.common import ensure_bgr, match_sensor_noise
from backend.watermark_engine.modules.logo_module import remove as logo_remove


def _extrapolate_horizontal_bar(bgr: np.ndarray, bar_mask: np.ndarray, match_grain: bool = True) -> np.ndarray:
    """
    Cleanly replaces full-width top or bottom letterbox bars by extrapolating
    adjacent active image rows with subtle progressive smoothing and grain matching.
    """
    h, w = bgr.shape[:2]
    out = bgr.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bar_mask)

    for i in range(1, num_labels):
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bw <= w * 0.65:
            continue

        is_top = by <= int(h * 0.12)
        is_bot = (by + bh) >= int(h * 0.88)

        if is_top and (by + bh) < h - 4:
            ref_y0 = min(h - 1, by + bh)
            ref_y1 = min(h, ref_y0 + 6)
            ref_band = np.median(out[ref_y0:ref_y1, bx:bx + bw].astype(np.float32), axis=0)
            for y in range(by, by + bh):
                dist = (by + bh) - y
                sigma = max(0.5, min(9.0, dist * 0.25))
                ksize = int(sigma * 3) | 1
                row_smooth = cv2.GaussianBlur(ref_band[np.newaxis, :, :], (ksize, 1), sigma)[0]
                m_row = labels[y, bx:bx + bw] == i
                out[y, bx:bx + bw][m_row] = np.clip(row_smooth, 0, 255).astype(np.uint8)[m_row]

        elif is_bot and by >= 4:
            ref_y1 = max(1, by)
            ref_y0 = max(0, ref_y1 - 6)
            ref_band = np.median(out[ref_y0:ref_y1, bx:bx + bw].astype(np.float32), axis=0)
            for y in range(by, by + bh):
                dist = (y - by) + 1
                sigma = max(0.5, min(9.0, dist * 0.25))
                ksize = int(sigma * 3) | 1
                row_smooth = cv2.GaussianBlur(ref_band[np.newaxis, :, :], (ksize, 1), sigma)[0]
                m_row = labels[y, bx:bx + bw] == i
                out[y, bx:bx + bw][m_row] = np.clip(row_smooth, 0, 255).astype(np.uint8)[m_row]

    if match_grain:
        out = match_sensor_noise(bgr, out, bar_mask)
    return out


def remove(
    img: np.ndarray,
    mask: np.ndarray,
    match_grain: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Separates full-width letterbox bars from compact container badges/pills:
    - Edge bars: directional vertical extrapolation (instant, mosaic-free).
    - Compact badges: clustered Big-LaMa neural inpainting via logo_module.
    """
    bgr = ensure_bgr(img)
    if mask is None or np.count_nonzero(mask) == 0:
        return bgr

    h, w = bgr.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))

    bar_mask = np.zeros((h, w), dtype=np.uint8)
    badge_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_labels):
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        is_edge_bar = (bw > w * 0.65) and (bh < h * 0.22) and (by <= int(h * 0.12) or (by + bh) >= int(h * 0.88))
        if is_edge_bar:
            bar_mask[labels == i] = 255
        else:
            badge_mask[labels == i] = 255

    result = bgr
    if np.count_nonzero(bar_mask) > 0:
        result = _extrapolate_horizontal_bar(result, bar_mask, match_grain=match_grain)
    if np.count_nonzero(badge_mask) > 0:
        result = logo_remove(result, badge_mask, pad=56, match_grain=match_grain, max_dim=640)

    return result
