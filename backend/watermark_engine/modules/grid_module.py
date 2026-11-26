"""
Repeating Translucent Grid & Stock Watermark Removal Module.
Uses Dekel-style 2D Bravais Lattice Basis Recovery & YCrCb Sub-Pixel Median Matte Inversion.
Strictly isolated from solid logo / letterbox / sparkle modules.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from backend.watermark_engine.modules.common import ensure_bgr
from backend.watermark_engine.modules import text_module


def _find_lattice_basis(points: list) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, list]]:
    """Find dominant 2D Bravais lattice basis (v1, v2, origin, inliers) from 2D peak points."""
    pts = np.array(points, dtype=np.float32)
    n = len(pts)
    if n < 3:
        return None

    diffs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[j] - pts[i]
            length = float(np.linalg.norm(d))
            if length >= 50.0:
                diffs.append((length, d))
    diffs.sort(key=lambda x: x[0])

    clusters = []
    for _, d in diffs:
        placed = False
        for c in clusters:
            if np.linalg.norm(d - c[0]) < 10.0:
                c.append(d)
                placed = True
                break
        if not placed:
            clusters.append([d])

    cand_vecs = []
    for c in clusters:
        mean_v = np.mean(c, axis=0)
        if mean_v[1] < -5.0 or (abs(mean_v[1]) <= 5.0 and mean_v[0] < 0):
            mean_v = -mean_v
        cand_vecs.append((len(c), float(np.linalg.norm(mean_v)), mean_v))

    uniq_vecs = []
    for votes, ln, v in cand_vecs:
        if not any(np.linalg.norm(v - uv[2]) < 12.0 for uv in uniq_vecs):
            uniq_vecs.append((votes, ln, v))

    uniq_vecs.sort(key=lambda x: (-min(x[0], 4), x[1]))
    vecs = [v for _, _, v in uniq_vecs[:15]]

    best_score = -1.0
    best_model = None
    for i in range(len(vecs)):
        v1 = vecs[i]
        for j in range(i + 1, len(vecs)):
            v2 = vecs[j]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(det) < 0.25 * np.linalg.norm(v1) * np.linalg.norm(v2):
                continue
            inv_m = np.linalg.inv(np.column_stack([v1, v2]))
            for p0_idx in range(min(n, 5)):
                p0 = pts[p0_idx]
                inliers = []
                for k in range(n):
                    uv = inv_m @ (pts[k] - p0)
                    mn = np.round(uv)
                    err = np.linalg.norm((pts[k] - p0) - (mn[0] * v1 + mn[1] * v2))
                    if err < 8.0:
                        mi, ni = int(mn[0]), int(mn[1])
                        if not any(im == mi and in_ == ni for im, in_, _ in inliers):
                            inliers.append((mi, ni, pts[k]))
                score = len(inliers) * 1000.0 - abs(det) * 0.002
                if len(inliers) >= 3 and score > best_score:
                    best_score = score
                    best_model = (v1, v2, p0, inliers)

    if best_model is None:
        return None

    _, _, _, inliers = best_model
    a_mat = np.array([[m, nn, 1.0] for m, nn, _ in inliers], dtype=np.float64)
    bx = np.array([p[0] for _, _, p in inliers], dtype=np.float64)
    by = np.array([p[1] for _, _, p in inliers], dtype=np.float64)
    px, _, _, _ = np.linalg.lstsq(a_mat, bx, rcond=None)
    py, _, _, _ = np.linalg.lstsq(a_mat, by, rcond=None)
    return (
        np.array([px[0], py[0]], dtype=np.float32),
        np.array([px[1], py[1]], dtype=np.float32),
        np.array([px[2], py[2]], dtype=np.float32),
        inliers,
    )


def dekel_repeating_removal(
    bgr: np.ndarray,
    user_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Dekel-style Multi-Instance Matte & 2D Lattice Recovery in YCrCb.
    Returns (cleaned_bgr, cleaned_area_mask_u8, applied_bool).
    """
    h, w = bgr.shape[:2]
    empty_mask = np.zeros((h, w), dtype=np.uint8)

    try:
        from backend.watermark_engine.detector import detect_text_watermarks
        _, ocr_items = detect_text_watermarks(bgr, sensitivity=0.5)
    except Exception:
        return bgr, empty_mask, False

    if not ocr_items or len(ocr_items) < 3:
        return bgr, empty_mask, False

    items = [{
        "pts": np.array(it["polygon"], dtype=np.float32),
        "text": it["text"],
        "score": float(it["confidence"]),
        "ang": float(it.get("angle", 0.0)),
        "bbox": it["bbox"],
    } for it in ocr_items]

    clusters = []
    for it in items:
        placed = False
        for c in clusters:
            if abs(it["ang"] - c[0]["ang"]) < 8.0:
                c.append(it)
                placed = True
                break
        if not placed:
            clusters.append([it])
    clusters.sort(key=len, reverse=True)
    dom_cluster = clusters[0]
    if len(dom_cluster) < 3:
        return bgr, empty_mask, False

    best_seed = None
    best_rank = -1e9
    for it in dom_cluster:
        bx, by, bw, bh = it["bbox"]
        border_pen = 0.5 if (bx <= 2 or by <= 2 or bx + bw >= w - 2 or by + bh >= h - 2) else 0.0
        length = float(np.linalg.norm(it["pts"][1] - it["pts"][0]))
        rank = it["score"] * 2.0 + min(length / 100.0, 2.0) - border_pen
        if rank > best_rank:
            best_rank = rank
            best_seed = it

    bx, by, bw, bh = best_seed["bbox"]
    pad = max(140, int(max(bw, bh) * 1.4))
    img_pad = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    ycrcb_pad = cv2.cvtColor(img_pad, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_pad = ycrcb_pad[:, :, 0]
    hp_y = y_pad - cv2.GaussianBlur(y_pad, (19, 19), 0)

    margin = 10
    rx0, ry0 = bx + pad - margin, by + pad - margin
    rw, rh = bw + 2 * margin, bh + 2 * margin
    seed_hp = hp_y[ry0:ry0 + rh, rx0:rx0 + rw]
    seed_mask = np.zeros((rh, rw), dtype=np.float32)
    poly_local = (best_seed["pts"] - np.array([bx - margin, by - margin])).astype(np.int32)
    cv2.fillPoly(seed_mask, [poly_local], 1.0)
    seed_mask = cv2.dilate(seed_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    seed_masked = seed_hp * seed_mask

    res_map = cv2.matchTemplate(hp_y, seed_masked, cv2.TM_CCOEFF_NORMED)
    peaks = []
    res_work = res_map.copy()
    nms_r = max(40, min(rw, rh) // 2)
    while len(peaks) < 60:
        _, max_v, _, max_loc = cv2.minMaxLoc(res_work)
        if max_v < 0.30:
            break
        px, py = max_loc
        cx_real = px + rw / 2.0 - pad
        cy_real = py + rh / 2.0 - pad
        if -bw * 0.4 <= cx_real <= w + bw * 0.4 and -bh * 0.4 <= cy_real <= h + bh * 0.4:
            peaks.append((max_v, px, py))
        x0, y0 = max(0, px - nms_r), max(0, py - nms_r)
        x1, y1 = min(res_work.shape[1], px + nms_r + 1), min(res_work.shape[0], py + nms_r + 1)
        res_work[y0:y1, x0:x1] = -1.0

    if len(peaks) < 3:
        return bgr, empty_mask, False

    seed_4x = cv2.resize(seed_masked, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    m = 4
    sub_peaks = []
    for sc, px, py in peaks:
        if py - m < 0 or px - m < 0 or py + rh + m > hp_y.shape[0] or px + rw + m > hp_y.shape[1]:
            continue
        cand_s = hp_y[py - m:py + rh + m, px - m:px + rw + m]
        cand_4x = cv2.resize(cand_s, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        r4 = cv2.matchTemplate(cand_4x, seed_4x, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc4 = cv2.minMaxLoc(r4)
        sub_peaks.append((sc, px + (loc4[0] / 4.0) - m, py + (loc4[1] / 4.0) - m))

    seed_pts = [(px, py) for sc, px, py in sub_peaks if sc >= 0.35]
    lattice = _find_lattice_basis(seed_pts)
    if lattice is None:
        return bgr, empty_mask, False
    v1, v2, p0, _ = lattice

    exp = int(max(rw, rh) * 0.55)
    gw, gh = rw + 2 * exp, rh + 2 * exp
    aligned_hp_big = []
    for sc, px, py in sub_peaks[:10]:
        if sc < max(0.38, peaks[0][0] * 0.50):
            continue
        ix, iy = int(np.floor(px)), int(np.floor(py))
        fx, fy = px - ix, py - iy
        gx0, gy0 = ix - exp, iy - exp
        if gx0 < 0 or gy0 < 0 or gx0 + gw + 2 > hp_y.shape[1] or gy0 + gh + 2 > hp_y.shape[0]:
            continue
        big_hp = hp_y[gy0:gy0 + gh + 2, gx0:gx0 + gw + 2]
        aligned_hp_big.append(
            cv2.warpAffine(big_hp, np.float32([[1, 0, -fx], [0, 1, -fy]]), (gw, gh), flags=cv2.INTER_CUBIC)
        )

    if len(aligned_hp_big) < 3:
        return bgr, empty_mask, False

    med_big_hp = np.median(np.stack(aligned_hp_big, axis=0), axis=0)
    is_light = float(np.percentile(med_big_hp, 99.2)) >= -float(np.percentile(med_big_hp, 0.8))
    act_map = med_big_hp if is_light else -med_big_hp

    seed_center = np.array([exp + rw / 2.0, exp + rh / 2.0], dtype=np.float32)
    yy_g, xx_g = np.mgrid[0:gh, 0:gw].astype(np.float32)
    d0 = (xx_g - seed_center[0]) ** 2 + (yy_g - seed_center[1]) ** 2
    voronoi_mask = np.ones((gh, gw), dtype=bool)
    for dm in (-1, 0, 1):
        for dn in (-1, 0, 1):
            if dm == 0 and dn == 0:
                continue
            nb = seed_center + dm * v1 + dn * v2
            d_nb = (xx_g - nb[0]) ** 2 + (yy_g - nb[1]) ** 2
            voronoi_mask &= (d0 <= d_nb * 1.02)

    active = ((act_map > 2.5) & voronoi_mask).astype(np.uint8) * 255
    ys, xs = np.where(active > 0)
    if len(xs) < 20:
        return bgr, empty_mask, False

    ax0, ay0 = max(0, int(xs.min()) - 6), max(0, int(ys.min()) - 6)
    ax1, ay1 = min(gw, int(xs.max()) + 7), min(gh, int(ys.max()) + 7)
    pw, ph = ax1 - ax0, ay1 - ay0
    off_x, off_y = ax0 - exp, ay0 - exp

    active_crop = active[ay0:ay1, ax0:ax1]
    text_inpaint_mask = cv2.dilate(active_crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    tile_strip_mask = text_inpaint_mask.astype(np.float32) / 255.0

    a_y_list, a_cr_list, a_cb_list = [], [], []
    for sc, px, py in sub_peaks[:10]:
        if len(a_y_list) >= 8 or sc < max(0.38, peaks[0][0] * 0.50):
            continue
        tx, ty = px + off_x, py + off_y
        ix, iy = int(np.floor(tx)), int(np.floor(ty))
        fx, fy = tx - ix, ty - iy
        if ix < 0 or iy < 0 or ix + pw + 2 > ycrcb_pad.shape[1] or iy + ph + 2 > ycrcb_pad.shape[0]:
            continue
        cand_ycc = ycrcb_pad[iy:iy + ph + 2, ix:ix + pw + 2]
        t_ycc = cv2.warpAffine(cand_ycc, np.float32([[1, 0, -fx], [0, 1, -fy]]), (pw, ph), flags=cv2.INTER_CUBIC)
        t_bgr = cv2.cvtColor(np.clip(t_ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        b_bgr = cv2.inpaint(t_bgr, text_inpaint_mask, 4, cv2.INPAINT_TELEA)
        b_ycc = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        d_ycc = (t_ycc - b_ycc) * tile_strip_mask[:, :, np.newaxis]

        denom_y = np.maximum(255.0 - b_ycc[:, :, 0], 45.0) if is_light else np.maximum(b_ycc[:, :, 0], 45.0)
        dy_signed = d_ycc[:, :, 0] if is_light else -d_ycc[:, :, 0]
        a_y_list.append(np.clip(dy_signed / denom_y, -0.05, 0.50))

        for ch_idx, acc in ((1, a_cr_list), (2, a_cb_list)):
            dc, bc = d_ycc[:, :, ch_idx], b_ycc[:, :, ch_idx]
            vc = np.abs(128.0 - bc) > 12.0
            ac = np.full_like(dc, np.nan)
            ac[vc] = np.clip(dc[vc] / (128.0 - bc[vc]), -0.15, 0.50)
            acc.append(ac)

    if not a_y_list:
        return bgr, empty_mask, False

    alpha_y = np.clip(np.median(np.stack(a_y_list, axis=0), axis=0) * tile_strip_mask, -0.04, 0.48)
    alpha_y[np.abs(alpha_y) < 0.015] = 0.0

    with np.errstate(all="ignore"):
        alpha_cr = np.nan_to_num(np.nanmedian(np.stack(a_cr_list, axis=0), axis=0), nan=0.0) * tile_strip_mask
        alpha_cb = np.nan_to_num(np.nanmedian(np.stack(a_cb_list, axis=0), axis=0), nan=0.0) * tile_strip_mask
    alpha_c = np.clip(0.5 * (alpha_cr + alpha_cb), 0.0, 0.45)
    alpha_c[alpha_c < 0.015] = 0.0

    global_ay = np.zeros(ycrcb_pad.shape[:2], dtype=np.float32)
    global_ac = np.zeros(ycrcb_pad.shape[:2], dtype=np.float32)

    m_lat = np.column_stack([v1, v2])
    inv_m = np.linalg.inv(m_lat)
    corners = np.array([
        [pad - pw, pad - ph],
        [pad + w + pw, pad - ph],
        [pad - pw, pad + h + ph],
        [pad + w + pw, pad + h + ph],
    ], dtype=np.float32)
    uv_c = np.array([inv_m @ (c - p0) for c in corners])
    m_min, m_max = int(np.floor(uv_c[:, 0].min())) - 1, int(np.ceil(uv_c[:, 0].max())) + 1
    n_min, n_max = int(np.floor(uv_c[:, 1].min())) - 1, int(np.ceil(uv_c[:, 1].max())) + 1

    user_mask_dil = None
    overlap_tiles = 0
    if user_mask is not None and np.count_nonzero(user_mask) > 0:
        user_mask_dil = cv2.dilate((user_mask > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    candidate_placements = []
    for m_i in range(m_min, m_max + 1):
        for n_i in range(n_min, n_max + 1):
            seed_pos = p0 + m_i * v1 + n_i * v2
            snapped = seed_pos
            for _, px, py in sub_peaks:
                cand_p = np.array([px, py], dtype=np.float32)
                if np.linalg.norm(cand_p - seed_pos) < 7.0:
                    snapped = cand_p
                    break
            tx, ty = float(snapped[0] + off_x), float(snapped[1] + off_y)
            ix, iy = int(np.floor(tx)), int(np.floor(ty))
            fx, fy = tx - ix, ty - iy
            x0, y0 = max(pad, ix), max(pad, iy)
            x1, y1 = min(pad + w, ix + pw), min(pad + h, iy + ph)
            if x1 <= x0 + 8 or y1 <= y0 + 8:
                continue

            a_y_crop = cv2.warpAffine(alpha_y, np.float32([[1, 0, fx], [0, 1, fy]]), (pw, ph), flags=cv2.INTER_CUBIC)[y0 - iy:y1 - iy, x0 - ix:x1 - ix]
            a_c_crop = cv2.warpAffine(alpha_c, np.float32([[1, 0, fx], [0, 1, fy]]), (pw, ph), flags=cv2.INTER_CUBIC)[y0 - iy:y1 - iy, x0 - ix:x1 - ix]
            if np.count_nonzero(a_y_crop > 0.04) < 15:
                continue

            hp_crop = hp_y[y0:y1, x0:x1]
            hp_sub = hp_crop - np.mean(hp_crop)
            a_sub = a_y_crop - np.mean(a_y_crop)
            sign_hp = hp_sub if is_light else -hp_sub
            corr = float(np.sum(sign_hp * a_sub) / ((np.linalg.norm(sign_hp) + 1e-6) * (np.linalg.norm(a_sub) + 1e-6)))
            if corr < 0.08:
                continue

            if corr < 0.55:
                local_dot = cv2.boxFilter(sign_hp * np.maximum(a_y_crop, 0), -1, (29, 29))
                local_a_norm = cv2.boxFilter(np.maximum(a_y_crop, 0) ** 2, -1, (29, 29))
                empty_half = (local_a_norm > 1e-4) & (local_dot <= 0.0)
                a_y_crop = a_y_crop.copy()
                a_c_crop = a_c_crop.copy()
                a_y_crop[empty_half] = 0.0
                a_c_crop[empty_half] = 0.0

            if user_mask_dil is not None:
                um_slice = user_mask_dil[y0 - pad:y1 - pad, x0 - pad:x1 - pad]
                if np.count_nonzero((um_slice > 0) & (a_y_crop > 0.04)) >= 12:
                    overlap_tiles += 1

            candidate_placements.append((x0, y0, x1, y1, a_y_crop, a_c_crop))

    if not candidate_placements:
        return bgr, empty_mask, False

    for x0, y0, x1, y1, a_y_crop, a_c_crop in candidate_placements:
        pos_mask = a_y_crop > 0
        cur_y = global_ay[y0:y1, x0:x1]
        cur_y[pos_mask] = np.maximum(cur_y[pos_mask], a_y_crop[pos_mask])
        neg_mask = (a_y_crop < 0) & (cur_y == 0)
        cur_y[neg_mask] = a_y_crop[neg_mask]
        global_ac[y0:y1, x0:x1] = np.maximum(global_ac[y0:y1, x0:x1], a_c_crop)

    g_y = global_ay[pad:pad + h, pad:pad + w]
    g_c = global_ac[pad:pad + h, pad:pad + w]

    if user_mask_dil is not None and overlap_tiles <= 1:
        um_f = (user_mask_dil > 0).astype(np.float32)
        g_y = g_y * um_f
        g_c = g_c * um_f

    out_ycc = ycrcb_pad[pad:pad + h, pad:pad + w].copy()
    target_y = 255.0 if is_light else 0.0
    out_ycc[:, :, 0] = (out_ycc[:, :, 0] - g_y * target_y) / np.maximum(1.0 - g_y, 0.45)
    out_ycc[:, :, 1] = (out_ycc[:, :, 1] - g_c * 128.0) / np.maximum(1.0 - g_c, 0.45)
    out_ycc[:, :, 2] = (out_ycc[:, :, 2] - g_c * 128.0) / np.maximum(1.0 - g_c, 0.45)

    res_bgr = cv2.cvtColor(np.clip(out_ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    med3 = cv2.medianBlur(res_bgr, 3)
    diff3 = np.max(np.abs(res_bgr.astype(np.float32) - med3.astype(np.float32)), axis=2)
    gray_m = cv2.cvtColor(med3, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b_m = cv2.blur(gray_m, (7, 7))
    b_m2 = cv2.blur(gray_m ** 2, (7, 7))
    bg_std = np.sqrt(np.maximum(0, b_m2 - b_m ** 2))

    smooth_stroke = ((g_y > 0.045) & (bg_std < 5.0)).astype(np.uint8) * 255
    smooth_stroke = cv2.dilate(smooth_stroke, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if np.count_nonzero(smooth_stroke) > 0:
        res_bgr = cv2.inpaint(res_bgr, smooth_stroke, 2, cv2.INPAINT_TELEA)

    spike_mask = (g_y > 0.14) & (bg_std >= 5.0) & (diff3 > 7.5)
    res_bgr[spike_mask] = med3[spike_mask]

    cleaned_area_mask = (g_y > 0.015).astype(np.uint8) * 255
    return res_bgr, cleaned_area_mask, True


def remove(
    img: np.ndarray,
    mask: np.ndarray,
    alpha_hint: float = 0.5,
    watermark_color: Optional[Tuple[int, int, int]] = None,
    **kwargs
) -> np.ndarray:
    """
    Executes isolated Repeating Grid removal:
    1. Runs 2D lattice Dekel YCrCb matte inversion.
    2. Cleans any remaining masked fragments with stroke-aware deblending.
    """
    bgr = ensure_bgr(img)
    if mask is None or np.count_nonzero(mask) == 0:
        return bgr

    cleaned, lattice_mask, applied = dekel_repeating_removal(bgr, user_mask=mask)
    if applied:
        dil_lat = cv2.dilate(lattice_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        rem_mask = cv2.bitwise_and(mask, cv2.bitwise_not(dil_lat))
        if np.count_nonzero(rem_mask) < 80:
            return cleaned
        return text_module.remove(cleaned, rem_mask, alpha_hint=alpha_hint, watermark_color=watermark_color)

    return text_module.remove(bgr, mask, alpha_hint=alpha_hint, watermark_color=watermark_color)
