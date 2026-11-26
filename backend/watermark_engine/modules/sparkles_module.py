"""
AI Corner Sparkles / Star Logo Module (Gemini / Imagen / SynthID).
Dedicated localized patch restoration for compact corner diamond stars.
"""

import numpy as np
from backend.watermark_engine.modules.logo_module import remove as logo_remove


def remove(
    img: np.ndarray,
    mask: np.ndarray,
    match_grain: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Removes AI corner sparkles with tight context padding (44px) and sensor grain matching.
    """
    return logo_remove(img, mask, pad=44, match_grain=match_grain, max_dim=512, **kwargs)
