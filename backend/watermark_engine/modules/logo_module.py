"""
Logo / Solid Badge / Opaque Object Removal Module (Zero-Mosaic).
Isolated strictly for opaque logos, stickers, corner badges, and manual object masks.
Never touches repeating grid/deblend logic.
"""

import cv2
import numpy as np
from backend.watermark_engine.modules.common import ensure_bgr, lama_inpaint_patch


def remove(
    img: np.ndarray,
    mask: np.ndarray,
    pad: int = 56,
    match_grain: bool = True,
    max_dim: int = 640,
    **kwargs
) -> np.ndarray:
    """
    Clustered Big-LaMa Neural Inpainting for logos, badges, and solid objects:
    - Clusters nearby components and feeds context-padded patches (<= max_dim).
    - Preserves 100% bit-exact original pixels outside the masked area.
    """
    bgr = ensure_bgr(img)
    if mask is None or np.count_nonzero(mask) == 0:
        return bgr

    h, w = bgr.shape[:2]
    result = bgr.copy()

    merge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    dilated_mask = cv2.dilate(mask, merge_k, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_mask)

    total_pixels = h * w
    mask_pixels = np.count_nonzero(mask)

    if mask_pixels > total_pixels * 0.45 or num_labels > 25:
        return lama_inpaint_patch(bgr, mask, match_grain=match_grain, max_dim=max_dim)

    for i in range(1, num_labels):
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)

        patch_img = result[y1:y2, x1:x2]
        comp_mask = ((labels[y1:y2, x1:x2] == i) & (mask[y1:y2, x1:x2] > 0)).astype(np.uint8) * 255

        if np.count_nonzero(comp_mask) == 0:
            continue

        result[y1:y2, x1:x2] = lama_inpaint_patch(
            patch_img, comp_mask, match_grain=match_grain, max_dim=max_dim
        )

    return result
