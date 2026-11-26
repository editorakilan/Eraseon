"""
Classical Inpainting Module (Telea FMM & Navier-Stokes PDE).
"""

import numpy as np
from backend.watermark_engine.modules.common import classical_inpaint


def remove_telea(img: np.ndarray, mask: np.ndarray, inpaint_radius: int = 5, **kwargs) -> np.ndarray:
    return classical_inpaint(img, mask, method="telea", inpaint_radius=inpaint_radius, dilate_mask=2)


def remove_ns(img: np.ndarray, mask: np.ndarray, inpaint_radius: int = 5, **kwargs) -> np.ndarray:
    return classical_inpaint(img, mask, method="ns", inpaint_radius=inpaint_radius, dilate_mask=2)
