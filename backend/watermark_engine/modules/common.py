"""
Shared low-level image, Big-LaMa, and classical inpainting utilities
used across isolated watermark removal modules.
"""

import os
import cv2
import numpy as np
import torch
from typing import Optional

_lama_model = None


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel BGR."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def get_lama_model():
    """Lazy load Big-LaMa TorchScript model."""
    global _lama_model
    if _lama_model is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "models", "lama.pt")
        if not os.path.exists(model_path):
            cache_path = os.path.expanduser(
                r"~/.cache/huggingface/hub/models--JosephCatrambone--big-lama-torchscript/snapshots/7dfbdc6dc27602d370909e8688f8f84404a1f995/lama.pt"
            )
            if os.path.exists(cache_path):
                model_path = cache_path

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _lama_model = torch.jit.load(model_path, map_location=device)
            _lama_model.eval()
        except Exception as e:
            print(f"Warning: Failed to load LaMa model ({e}), falling back to harmonic/classical inpainting.")
            _lama_model = False

    return _lama_model if _lama_model is not False else None


def match_sensor_noise(patch_img: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Measures high-frequency sensor noise & film grain from the unmasked border region
    and synthesizes matching grain inside the inpainted area so it blends naturally.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return inpainted

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    border = cv2.subtract(cv2.dilate(mask, k, iterations=1), mask)
    if np.count_nonzero(border) < 30:
        return inpainted

    gray_patch = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_patch.astype(np.float32), (5, 5), 0)
    noise_field = gray_patch.astype(np.float32) - blur
    border_noise = noise_field[border > 0]

    std_noise = float(np.std(border_noise))
    if std_noise > 1.2:
        rng = np.random.default_rng(42)
        noise_synth = rng.normal(0, std_noise * 0.70, inpainted.shape).astype(np.float32)
        m_float = cv2.GaussianBlur((mask > 0).astype(np.float32), (3, 3), 1.0)[:, :, np.newaxis]
        return np.clip(inpainted.astype(np.float32) + noise_synth * m_float, 0, 255).astype(np.uint8)

    return inpainted


def classical_inpaint(
    img: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    inpaint_radius: int = 5,
    dilate_mask: int = 2
) -> np.ndarray:
    """Classical Telea FMM or Navier-Stokes PDE inpainting."""
    bgr = ensure_bgr(img)
    if mask is None or np.count_nonzero(mask) == 0:
        return bgr
    if dilate_mask > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_mask * 2 + 1, dilate_mask * 2 + 1))
        proc_mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        proc_mask = mask.copy()

    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(bgr, proc_mask, inpaintRadius=inpaint_radius, flags=flag)


def lama_inpaint_patch(
    patch_img: np.ndarray,
    patch_mask: np.ndarray,
    match_grain: bool = True,
    max_dim: int = 640
) -> np.ndarray:
    """
    Runs Big-LaMa neural inpainting on a single BGR patch.
    Automatically bounds oversized patches (> max_dim) to keep LaMa within its optimal
    receptive field (preventing FFC spectral checkerboard/mosaic ringing and CPU stalls).
    """
    model = get_lama_model()
    if model is None:
        return classical_inpaint(patch_img, patch_mask, method="telea", inpaint_radius=5, dilate_mask=1)

    ph, pw = patch_img.shape[:2]
    scale = 1.0
    if max(ph, pw) > max_dim:
        scale = max_dim / float(max(ph, pw))
        sw = max(16, int(round(pw * scale)))
        sh = max(16, int(round(ph * scale)))
        work_img = cv2.resize(patch_img, (sw, sh), interpolation=cv2.INTER_AREA)
        work_mask = cv2.resize(patch_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        # Ensure thin mask features survive downscaling
        work_mask = cv2.dilate(work_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    else:
        work_img = patch_img
        work_mask = patch_mask

    wh, ww = work_img.shape[:2]
    pad_h = (8 - wh % 8) % 8
    pad_w = (8 - ww % 8) % 8

    rgb = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    m_norm = (work_mask > 0).astype(np.float32)

    if pad_h > 0 or pad_w > 0:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        m_norm = np.pad(m_norm, ((0, pad_h), (0, pad_w)), mode="reflect")

    device = next(model.parameters()).device
    t_img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    t_mask = torch.from_numpy(m_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(t_img, t_mask)

    out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if pad_h > 0:
        out_np = out_np[:-pad_h, :]
    if pad_w > 0:
        out_np = out_np[:, :-pad_w]

    out_bgr = cv2.cvtColor(np.clip(out_np * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    if scale < 1.0:
        out_bgr = cv2.resize(out_bgr, (pw, ph), interpolation=cv2.INTER_LANCZOS4)

    if match_grain:
        out_bgr = match_sensor_noise(patch_img, out_bgr, patch_mask)

    soft_mask = cv2.GaussianBlur((patch_mask > 0).astype(np.float32), (5, 5), 1.2)[:, :, np.newaxis]
    blended = ((1.0 - soft_mask) * patch_img.astype(np.float32) + soft_mask * out_bgr.astype(np.float32)).astype(np.uint8)
    return blended
