"""
Watermark Removal and Restoration Engine (Modular Facade)
Delegates each operation strictly to its isolated module in backend.watermark_engine.modules.
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np

from backend.watermark_engine.modules import (
    run_module_removal,
    MODULE_DESCRIPTIONS,
    logo_module,
    grid_module,
    sparkles_module,
    letterbox_module,
    text_module,
    inpaint_module,
)
from backend.watermark_engine.modules.common import (
    get_lama_model,
    ensure_bgr,
    match_sensor_noise,
    lama_inpaint_patch as lama_inpaint_single_patch,
    classical_inpaint as content_aware_inpaint,
)


def lama_deep_inpaint(img: np.ndarray, mask: np.ndarray, pad: int = 56, match_grain: bool = True) -> np.ndarray:
    return logo_module.remove(img, mask, pad=pad, match_grain=match_grain)


def alpha_deblend(
    img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.20,
    watermark_color: Optional[Tuple[int, int, int]] = None,
    adaptive_alpha: bool = True
) -> np.ndarray:
    return grid_module.remove(img, mask, alpha_hint=alpha, watermark_color=watermark_color)


def hybrid_watermark_removal(
    img: np.ndarray,
    mask: np.ndarray,
    mode: str = "auto",
    alpha_hint: float = 0.5,
    watermark_color: Optional[Tuple[int, int, int]] = None,
    inpaint_radius: int = 5,
    match_grain: bool = True,
    boxes_hint: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    return run_module_removal(
        img=img,
        mask=mask,
        mode=mode,
        alpha_hint=alpha_hint,
        watermark_color=watermark_color,
        inpaint_radius=inpaint_radius,
        match_grain=match_grain,
        boxes_hint=boxes_hint,
    )
