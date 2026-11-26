"""
Single Translucent Text / Signature / Stamp Deblending Module.
Preserves underlying fine textures (hair, skin, sharp waves) via stroke-aware YCrCb
luminance subtraction and heals smooth backgrounds (sky, calm water) with thin-seam Telea.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from backend.watermark_engine.modules.common import ensure_bgr
from backend.watermark_engine.modules.logo_module import remove as logo_remove


def remove(
    img: np.ndarray,
    mask: np.ndarray,
    alpha_hint: float = 0.20,
    watermark_color: Optional[Tuple[int, int, int]] = None,
    **kwargs
) -> np.ndarray:
    bgr = ensure_bgr(img)
    mask_bool = mask > 0
    if np.count_nonzero(mask_bool) == 0:
        return bgr

    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_ch = ycc[:, :, 0]
    b_y = cv2.medianBlur(y_ch.astype(np.uint8), 11).astype(np.float32)
    diff_y = y_ch - b_y

    if watermark_color is not None:
        wm_b, wm_g, wm_r = watermark_color
        is_light = (wm_b + wm_g + wm_r) > 384
    else:
        is_light = float(np.mean(diff_y[mask_bool])) >= 0.0

    mag_y = diff_y if is_light else -diff_y
    stroke_mask = mask_bool & (mag_y >= 5.0)
    if np.count_nonzero(stroke_mask) == 0:
        return logo_remove(bgr, mask)

    b_blur = cv2.blur(b_y, (7, 7))
    b_sq = cv2.blur(b_y ** 2, (7, 7))
    bg_std = np.sqrt(np.maximum(0, b_sq - b_blur ** 2))

    diff_capped = np.clip(mag_y, 0.0, 26.0)
    y_rec = y_ch.copy()
    y_rec[stroke_mask] -= diff_capped[stroke_mask] if is_light else -diff_capped[stroke_mask]
    ycc[:, :, 0] = np.clip(y_rec, 0.0, 255.0)
    recovered = cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    seam_mask = cv2.dilate(stroke_mask.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    smooth_seam = seam_mask & ((bg_std < 6.0).astype(np.uint8) * 255)
    if np.count_nonzero(smooth_seam) > 0:
        recovered = cv2.inpaint(recovered, smooth_seam, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    return recovered
