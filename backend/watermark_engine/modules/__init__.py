"""
Modular Operation Registry for Eraseon Watermark Removal.
Each watermark/operation type has a completely isolated module so tuning or fixing
one operation (e.g., Logo, Grid, Sparkles, Letterbox, Text) never introduces regressions in another.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from backend.watermark_engine.modules.common import ensure_bgr
from backend.watermark_engine.modules import (
    logo_module,
    grid_module,
    sparkles_module,
    letterbox_module,
    text_module,
    inpaint_module,
)


MODULE_DESCRIPTIONS = {
    "auto": "Smart Auto-Router (routes each detected region to its isolated module)",
    "logo": "Logo / Solid Badge / Sticker Module (Zero-Mosaic Big-LaMa)",
    "grid": "Translucent Grid / Stock Text Module (Zero-Blur Dekel 2D Lattice)",
    "sparkles": "AI Corner Sparkles / Star Module (Gemini / Imagen / SynthID)",
    "letterbox": "Letterbox Bars & Aspect Badges Module (Edge Extrapolation + Badge Fill)",
    "text": "Single Translucent Text / Signature Module (Stroke-Aware YCrCb Deblend)",
    "inpaint_telea": "Classical Patch Module (Telea Fast Marching)",
    "inpaint_ns": "Smooth Gradient Module (Navier-Stokes PDE)",
}


def run_module_removal(
    img: np.ndarray,
    mask: np.ndarray,
    mode: str = "auto",
    alpha_hint: float = 0.5,
    watermark_color: Optional[Tuple[int, int, int]] = None,
    inpaint_radius: int = 5,
    match_grain: bool = True,
    boxes_hint: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    """
    Dispatches watermark removal strictly to the requested isolated operation module.
    In 'auto' mode, separates solid boxes (letterbox, sparkles, container badges) from
    repeating grids so each region is cleaned by its dedicated module independently.
    """
    bgr = ensure_bgr(img)
    if mask is None or np.count_nonzero(mask) == 0:
        return bgr

    m = (mode or "auto").lower().strip()

    # Explicit module routing (100% isolated — zero cross-module interference)
    if m in ("logo", "lama"):
        return logo_module.remove(bgr, mask, match_grain=match_grain)
    if m in ("grid", "deblend"):
        return grid_module.remove(bgr, mask, alpha_hint=alpha_hint, watermark_color=watermark_color)
    if m in ("sparkles", "ai_sparkles"):
        return sparkles_module.remove(bgr, mask, match_grain=match_grain)
    if m == "letterbox":
        return letterbox_module.remove(bgr, mask, match_grain=match_grain)
    if m == "text":
        return text_module.remove(bgr, mask, alpha_hint=alpha_hint, watermark_color=watermark_color)
    if m == "inpaint_telea":
        return inpaint_module.remove_telea(bgr, mask, inpaint_radius=inpaint_radius)
    if m == "inpaint_ns":
        return inpaint_module.remove_ns(bgr, mask, inpaint_radius=inpaint_radius)

    # Smart Auto Mode: route cleanly per region type if boxes_hint is available
    h, w = bgr.shape[:2]
    working = bgr
    remaining_mask = mask.copy()

    if boxes_hint:
        lb_mask = np.zeros((h, w), dtype=np.uint8)
        sp_mask = np.zeros((h, w), dtype=np.uint8)
        solid_mask = np.zeros((h, w), dtype=np.uint8)
        has_grid = False

        for b in boxes_hint:
            b_type = b.get("type", "")
            x0 = max(0, int(b.get("x", 0)) - 6)
            y0 = max(0, int(b.get("y", 0)) - 6)
            x1 = min(w, int(b.get("x", 0) + b.get("width", 0)) + 6)
            y1 = min(h, int(b.get("y", 0) + b.get("height", 0)) + 6)
            if x1 <= x0 or y1 <= y0:
                continue

            if b_type in ("letterbox_bar", "container_badge"):
                lb_mask[y0:y1, x0:x1] = np.where(remaining_mask[y0:y1, x0:x1] > 0, 255, lb_mask[y0:y1, x0:x1])
            elif b_type == "ai_sparkles":
                sp_mask[y0:y1, x0:x1] = np.where(remaining_mask[y0:y1, x0:x1] > 0, 255, sp_mask[y0:y1, x0:x1])
            elif b_type == "translucent_grid":
                has_grid = True
            else:
                solid_mask[y0:y1, x0:x1] = np.where(remaining_mask[y0:y1, x0:x1] > 0, 255, solid_mask[y0:y1, x0:x1])

        if np.count_nonzero(lb_mask) > 0:
            working = letterbox_module.remove(working, lb_mask, match_grain=match_grain)
            remaining_mask = cv2.bitwise_and(remaining_mask, cv2.bitwise_not(lb_mask))
        if np.count_nonzero(sp_mask) > 0:
            working = sparkles_module.remove(working, sp_mask, match_grain=match_grain)
            remaining_mask = cv2.bitwise_and(remaining_mask, cv2.bitwise_not(sp_mask))
        if has_grid and np.count_nonzero(remaining_mask) > 0:
            return grid_module.remove(working, remaining_mask, alpha_hint=alpha_hint, watermark_color=watermark_color)
        if np.count_nonzero(remaining_mask) > 0:
            return logo_module.remove(working, remaining_mask, match_grain=match_grain)
        return working

    # Manual mask in Auto mode: only trigger grid module if >=5 components AND verified lattice exists
    num_labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
    if num_labels >= 6:
        cleaned, lattice_mask, applied = grid_module.dekel_repeating_removal(bgr, user_mask=mask)
        if applied:
            dil_lat = cv2.dilate(lattice_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            rem_mask = cv2.bitwise_and(mask, cv2.bitwise_not(dil_lat))
            if np.count_nonzero(rem_mask) < 80:
                return cleaned
            return logo_module.remove(cleaned, rem_mask, match_grain=match_grain)

    return logo_module.remove(bgr, mask, match_grain=match_grain)
