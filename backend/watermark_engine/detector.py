"""
Watermark Auto-Detection Engine
Detects watermarks using:
1. RapidOCR for rotated/diagonal text, stamps, copyright lines, and signatures.
2. Multi-scale High-Pass / Morphological Gradient filtering for faint/translucent logos and lines.
3. Connected Component Analysis with spatial heuristics.
4. AI Corner Sparkles & Star Logo Detector (Google Gemini / Imagen / SynthID).
5. Letterbox black bars & dark container overlay badges.
6. Translucent repeating grid & stock photography watermark detector.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

# Lazy-loaded RapidOCR instance to optimize startup time
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception as e:
            print(f"Warning: RapidOCR could not be initialized: {e}")
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None


STRICT_WATERMARK_KEYWORDS = [
    "watermark", "shutterstock", "getty", "istock", "alamy", "adobe stock",
    "confidential", "draft", "do not copy", "all rights reserved",
    "copyright", "(c)", "©", "synthid", "imagen", "mark jacob", "jane doe", "zoner"
]

WATERMARK_KEYWORDS = STRICT_WATERMARK_KEYWORDS + [
    "preview", "sample", "copy", "proof", "stock",
    "letterbox", "black bar", "epiq", "container", "presentation",
    "photographer", "photography", "photo", "shot on",
    "camera", "shot with", "nikon", "canon", "sony", "fujifilm", "hasselblad", "leica"
]


def _is_white_paper_body_text(gray: np.ndarray, bx: int, by: int, bw: int, bh: int, angle: float, text_lower: str) -> bool:
    """
    Returns True when an OCR box is normal dark-ink horizontal document body text on white paper
    (e.g. a PDF paragraph or table cell), so document text is never flagged as a watermark.
    """
    if abs(angle) > 8.0:
        return False
    if any(kw in text_lower for kw in STRICT_WATERMARK_KEYWORDS):
        return False
    h, w = gray.shape[:2]
    pad = 4
    crop = gray[max(0, by - pad):min(h, by + bh + pad), max(0, bx - pad):min(w, bx + bw + pad)]
    if crop.size == 0:
        return False
    bg_p85 = float(np.percentile(crop, 85))
    ink_p2 = float(np.percentile(crop, 2))
    # Bright white paper (>222) + dark crisp ink (<125) = standard PDF/document body text
    return (bg_p85 > 222.0) and (ink_p2 < 125.0)


def detect_text_watermarks(img: np.ndarray, sensitivity: float = 0.5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detect text-based watermarks using RapidOCR.
    Handles rotated, diagonal, and horizontal text.
    Strictly distinguishes watermarks from genuine scene signage and white-paper PDF body text.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    ocr = get_ocr_engine()
    results = []

    if ocr is None:
        return mask, results

    try:
        ocr_res, _ = ocr(img)
        if not ocr_res:
            return mask, results

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Pre-filter normal white-paper document paragraphs before counting repeating words
        filtered_items = []
        for item in ocr_res:
            pts, text, score = item[0], item[1], float(item[2])
            pts_np = np.array(pts, dtype=np.int32)
            bx, by, bw, bh = cv2.boundingRect(pts_np)
            dx = float(pts_np[1][0] - pts_np[0][0])
            dy = float(pts_np[1][1] - pts_np[0][1])
            angle = float(np.degrees(np.arctan2(dy, dx)))
            text_lower = text.lower()
            if _is_white_paper_body_text(gray, bx, by, bw, bh, angle, text_lower):
                continue
            filtered_items.append((pts_np, bx, by, bw, bh, angle, text, text_lower, score))

        # Count occurrences of words among non-body-text items to detect repeating watermark grids
        word_counts = {}
        for _, _, _, _, _, _, _, text_lower, _ in filtered_items:
            for token in text_lower.split():
                clean_tok = "".join(ch for ch in token if ch.isalnum())
                if len(clean_tok) >= 3:
                    word_counts[clean_tok] = word_counts.get(clean_tok, 0) + 1

        for pts_np, bx, by, bw, bh, angle, text, text_lower, score in filtered_items:
            min_score = max(0.18, 0.60 - (sensitivity * 0.40))
            if score >= min_score:
                is_kw = any(kw in text_lower for kw in WATERMARK_KEYWORDS)

                # Check if any token in this text repeats >= 2 times across the image (repeating grid)
                tokens = ["".join(ch for ch in t if ch.isalnum()) for t in text_lower.split()]
                is_repeating = any(word_counts.get(t, 0) >= 2 for t in tokens if len(t) >= 3)

                # Outer margin / corner placement (typical for watermarks, timestamps, signatures, corner logos)
                in_corner_margin = (by < h * 0.20) or ((by + bh) > h * 0.70) or (bx < w * 0.28) or ((bx + bw) > w * 0.70)

                # Center scene (where genuine highway signs, storefronts, billboards, and subjects appear)
                is_center_scene = (w * 0.22 <= bx <= w * 0.78) and (h * 0.18 <= by <= h * 0.70)

                should_include = False
                if is_center_scene:
                    if is_kw or is_repeating:
                        should_include = True
                else:
                    if is_kw or is_repeating or (in_corner_margin and score >= 0.35):
                        should_include = True

                if should_include:
                    cv2.fillPoly(mask, [pts_np], 255)
                    results.append({
                        "text": text,
                        "confidence": round(score, 3),
                        "polygon": pts_np.tolist(),
                        "bbox": (int(bx), int(by), int(bw), int(bh)),
                        "angle": round(angle, 1),
                        "is_keyword_match": is_kw
                    })
    except Exception as e:
        print(f"OCR detection error: {e}")

    return mask, results


def detect_ai_corner_sparkles(img: np.ndarray, sensitivity: float = 0.5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detects AI watermark sparkles and star logos (e.g. Google Gemini / Imagen / SynthID 
    4-pointed diamond stars) placed in the bottom-right corner.
    Uses multi-scale tophat local contrast and luminance thresholding.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []

    # Standard placement for AI generation stars (SynthID / Gemini / Imagen) is bottom-right
    corners = [
        ("bottom_right", int(h * 0.60), h, int(w * 0.68), w)
    ]

    for corner_name, y1, y2, x1, x2 in corners:
        crop = img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 30 or cw < 30:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Skip white-paper PDF margins
        if float(np.median(gray)) > 235.0:
            continue

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        thresh_th_val = max(14, int(20 - sensitivity * 10))
        _, th_thresh = cv2.threshold(th, thresh_th_val, 255, cv2.THRESH_BINARY)
        _, th_sharp = cv2.threshold(th, max(24, thresh_th_val + 12), 255, cv2.THRESH_BINARY)

        thresh_br_val = max(150, int(180 - sensitivity * 30))
        _, br_thresh = cv2.threshold(gray, thresh_br_val, 255, cv2.THRESH_BINARY)

        clean_th = cv2.morphologyEx(th_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        clean_th_sharp = cv2.morphologyEx(th_sharp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        clean_br = cv2.morphologyEx(br_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        for m_cand in [clean_th, clean_th_sharp, clean_br]:
            contours, _ = cv2.findContours(m_cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh) if bh > 0 else 0

                # SynthID / AI diamond stars are compact and near-symmetric
                if 12 <= bw <= 55 and 12 <= bh <= 55 and 0.75 <= aspect <= 1.35 and 40 <= area <= 1800:
                    sc = crop[by:by + bh, bx:bx + bw]
                    b, g, r = sc[:, :, 0], sc[:, :, 1], sc[:, :, 2]
                    color_diff = np.mean(np.abs(b.astype(int) - g.astype(int)) + np.abs(g.astype(int) - r.astype(int)))
                    if color_diff < 55:
                        gx = x1 + bx
                        gy = y1 + by
                        pad = 4
                        px = max(0, gx - pad)
                        py = max(0, gy - pad)
                        pw = min(w - px, bw + pad * 2)
                        ph = min(h - py, bh + pad * 2)
                        cv2.rectangle(mask, (px, py), (px + pw, py + ph), 255, -1)
                        boxes.append({
                            "x": int(px),
                            "y": int(py),
                            "width": int(pw),
                            "height": int(ph),
                            "type": "ai_sparkles",
                            "label": "AI Corner Sparkle"
                        })

    return mask, boxes


def detect_letterbox_overlays(img: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detects letterbox black bars and overlay tags in or near margins.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    row_means = np.mean(gray, axis=1)

    # 1. Top Letterbox Bar
    top_bar_h = 0
    for r in range(int(h * 0.15)):
        if row_means[r] < 35:
            top_bar_h = r + 1
    if top_bar_h > 5:
        cv2.rectangle(mask, (0, 0), (w, top_bar_h + 3), 255, -1)
        boxes.append({
            "x": 0,
            "y": 0,
            "width": int(w),
            "height": int(top_bar_h + 3),
            "type": "letterbox_bar",
            "label": "Top Letterbox Bar"
        })

    # 2. Bottom Letterbox Bar
    bot_bar_h = 0
    for r in range(h - 1, int(h * 0.85), -1):
        if row_means[r] < 35:
            bot_bar_h = h - r
    if bot_bar_h > 5:
        cv2.rectangle(mask, (0, h - bot_bar_h - 3), (w, h), 255, -1)
        boxes.append({
            "x": 0,
            "y": int(h - bot_bar_h - 3),
            "width": int(w),
            "height": int(bot_bar_h + 3),
            "type": "letterbox_bar",
            "label": "Bottom Letterbox Bar"
        })

    return mask, boxes


def detect_translucent_grid_watermarks(img: np.ndarray, sensitivity: float = 0.5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detects repeating translucent emblems and stock photography stamps.
    Ensures bounded bounding boxes that do not swallow entire scenic backgrounds.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    th1 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k1)
    th2 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k2)
    tophat = cv2.max(th1, th2)

    bh1 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k1)
    bh2 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k2)
    blackhat = cv2.max(bh1, bh2)

    combined = cv2.max(tophat, blackhat)

    thresh_val = max(14, int(22 - sensitivity * 10))
    _, bin_strokes = cv2.threshold(combined, thresh_val, 255, cv2.THRESH_BINARY)
    clean = cv2.morphologyEx(bin_strokes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    connect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    connected = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, connect_k)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        # Bounded emblem/stamp size (prevents merging whole waves/skies)
        if 40 <= area <= 6000 and 15 <= bw <= int(w * 0.25) and 12 <= bh <= int(h * 0.18):
            aspect = bw / float(bh) if bh > 0 else 0
            if 0.35 <= aspect <= 3.0:
                pad = 4
                px = max(0, bx - pad)
                py = max(0, by - pad)
                pw = min(w - px, bw + pad * 2)
                ph = min(h - py, bh + pad * 2)
                cv2.rectangle(mask, (px, py), (px + pw, py + ph), 255, -1)
                boxes.append({
                    "x": int(px),
                    "y": int(py),
                    "width": int(pw),
                    "height": int(ph),
                    "type": "translucent_grid",
                    "label": "Translucent Emblem"
                })

    return mask, boxes


def merge_bounding_boxes(boxes: List[Dict[str, Any]], dist_thresh: int = 10, img_width: int = 1000, img_height: int = 1000) -> List[Dict[str, Any]]:
    """
    Merges overlapping and closely adjacent bounding boxes (e.g. 'Mark' + 'Jacob').
    Strictly prevents chaining from creating large screen-filling boxes.
    """
    if not boxes:
        return []

    rects = [[
        b["x"],
        b["y"],
        b["x"] + b["width"],
        b["y"] + b["height"],
        b.get("type", "watermark"),
        b.get("label", "Watermark Box")
    ] for b in boxes]

    total_img_area = img_width * img_height

    changed = True
    while changed:
        changed = False
        new_rects = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            r1 = rects[i]
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                r2 = rects[j]

                # Check if either box is a full-width letterbox bar
                w1 = r1[2] - r1[0]
                w2 = r2[2] - r2[0]
                is_bar1 = w1 > img_width * 0.75
                is_bar2 = w2 > img_width * 0.75

                if is_bar1 or is_bar2:
                    if is_bar1 and is_bar2 and (abs(r1[1] - r2[1]) < 20):
                        r1 = [min(r1[0], r2[0]), min(r1[1], r2[1]), max(r1[2], r2[2]), max(r1[3], r2[3]), "letterbox_bar", "Letterbox Bar"]
                        used[j] = True
                        changed = True
                    continue

                # Never merge different distinct categories (e.g. AI sparkles should not merge with badges or text)
                if r1[4] != r2[4] and "ai_sparkles" in (r1[4], r2[4]):
                    continue

                # Intersection area
                ix1 = max(r1[0], r2[0])
                iy1 = max(r1[1], r2[1])
                ix2 = min(r1[2], r2[2])
                iy2 = min(r1[3], r2[3])
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                area1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
                area2 = (r2[2] - r2[0]) * (r2[3] - r2[1])

                # Union bounds
                ux1 = min(r1[0], r2[0])
                uy1 = min(r1[1], r2[1])
                ux2 = max(r1[2], r2[2])
                uy2 = max(r1[3], r2[3])
                uw = ux2 - ux1
                uh = uy2 - uy1
                union_area = uw * uh

                # Significant direct overlap
                is_overlapping = inter_area > 0 and (inter_area / float(min(area1, area2)) > 0.20)

                # Are they adjacent words on the same line (like "Mark" + "Jacob")?
                v_overlap = max(0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
                min_h = min(r1[3] - r1[1], r2[3] - r2[1])
                h_gap = max(0, max(r1[0], r2[0]) - min(r1[2], r2[2]))
                same_line_words = (v_overlap >= 0.5 * min_h) and (h_gap <= 10) and (union_area <= 1.30 * (area1 + area2))

                # Allow merge only if compact and bounded
                is_aspect_badge = any("container" in str(t) or "badge" in str(t) for t in (r1[4], r2[4]))
                max_allowed_dim = max(img_width * 0.45, 450) if is_aspect_badge else max(img_width * 0.25, 220)

                if (same_line_words or is_overlapping) and (uw <= max_allowed_dim) and (uh <= 160) and (union_area < total_img_area * 0.12):
                    r1 = [
                        ux1, uy1, ux2, uy2,
                        r1[4] if r1[4] != "watermark" else r2[4],
                        r1[5] if r1[5] != "Watermark Box" else r2[5]
                    ]
                    used[j] = True
                    changed = True

            new_rects.append(r1)
        rects = new_rects

    out = []
    for r in rects:
        out.append({
            "x": int(r[0]),
            "y": int(r[1]),
            "width": int(r[2] - r[0]),
            "height": int(r[3] - r[1]),
            "type": r[4],
            "label": r[5]
        })
    return out


def _find_embedded_photo_regions(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    On white-background PDF/document pages, locates embedded rectangular image regions
    (e.g. figures/screenshots) so corner stars and letterboxes inside figures are also detected.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # Only run on white-margin document pages
    border_vals = np.concatenate([gray[:10, :].ravel(), gray[-10:, :].ravel(), gray[:, :10].ravel(), gray[:, -10:].ravel()])
    if np.median(border_vals) < 235:
        return []

    non_white = (gray < 238).astype(np.uint8) * 255
    # Close tiny gaps inside photos without merging separate paragraphs
    closed = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    min_area = h * w * 0.04
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area and bw >= w * 0.35 and bh >= h * 0.14:
            roi = gray[by:by + bh, bx:bx + bw]
            # Dense figure/photo has lower brightness and higher fill than sparse text tables
            fill_ratio = float(np.count_nonzero(roi < 235)) / float(roi.size)
            if fill_ratio > 0.72 and float(np.std(roi)) > 18.0:
                regions.append((bx, by, bw, bh))
    return regions


def auto_detect_watermark(
    img: np.ndarray,
    sensitivity: float = 0.5,
    dilate_pixels: int = 4,
    target_type: str = "all"
) -> Dict[str, Any]:
    """
    Combined auto-detector:
    Detects watermark regions as clean individual bounding boxes.
    Returns:
    - mask: binary mask image with filled boxes
    - boxes: list of bounding boxes with labels
    - detected_count: number of boxes found
    - recommended_mode: "auto" (LaMa) or "deblend" (Zero-blur inverse alpha)
    """
    h, w = img.shape[:2]
    candidate_boxes = []
    ocr_results = []
    has_translucent_grid = False

    # Scan embedded figures on white-paper PDF pages, or full image for regular photos
    embedded = _find_embedded_photo_regions(img)
    scan_regions = embedded if embedded else [(0, 0, w, h)]

    # 1. AI Corner Sparkles Detector (Google Gemini / Imagen / SynthID)
    if target_type in ("all", "ai_sparkles"):
        for rx, ry, rw, rh in scan_regions:
            sub = img[ry:ry + rh, rx:rx + rw]
            _, sparkle_boxes = detect_ai_corner_sparkles(sub, sensitivity=sensitivity)
            for b in sparkle_boxes:
                b["x"] += rx
                b["y"] += ry
                candidate_boxes.append(b)

    # 2. Letterbox & Banner Overlay Detector
    if target_type in ("all", "letterbox"):
        for rx, ry, rw, rh in scan_regions:
            sub = img[ry:ry + rh, rx:rx + rw]
            _, lb_boxes = detect_letterbox_overlays(sub)
            for b in lb_boxes:
                b["x"] += rx
                b["y"] += ry
                candidate_boxes.append(b)

    # 3. OCR and Text Watermarks
    diagonal_boxes = []
    if target_type in ("all", "text", "translucent_grid"):
        ocr_mask, ocr_results = detect_text_watermarks(img, sensitivity=sensitivity)
        for item in ocr_results:
            bx, by, bw, bh = item["bbox"]
            text_l = item["text"].lower()
            angle = item.get("angle", 0.0)
            poly = item.get("polygon")

            if any(k in text_l for k in ["epiq", "container", "presentation", "1.89:1", "2.00:1", "2.10:1", "2.17:1", "2.20:1"]):
                candidate_boxes.append({
                    "x": max(0, bx - 10),
                    "y": max(0, by - 8),
                    "width": min(w - bx + 10, bw + 20),
                    "height": min(h - by + 8, bh + 16),
                    "type": "container_badge",
                    "label": "Aspect Ratio Badge"
                })
            elif any(k in text_l for k in ["letterbox", "black bar"]):
                pass
            else:
                is_diagonal = abs(angle) > 10.0
                box_entry = {
                    "x": max(0, bx - 4),
                    "y": max(0, by - 4),
                    "width": min(w - bx + 4, bw + 8),
                    "height": min(h - by + 4, bh + 8),
                    "type": "watermark_text",
                    "label": f"Text: {item['text'][:18]}",
                    "angle": angle
                }
                if is_diagonal and poly:
                    box_entry["polygon"] = poly
                    diagonal_boxes.append(box_entry)
                else:
                    candidate_boxes.append(box_entry)
        if len(ocr_results) >= 4:
            has_translucent_grid = True

    # 4. Translucent Grid Emblems (only when requested, or if target_type == "all" and NO OCR text was found)
    if target_type == "translucent_grid" or (target_type == "all" and not ocr_results and has_translucent_grid):
        _, grid_boxes = detect_translucent_grid_watermarks(img, sensitivity=sensitivity)
        candidate_boxes.extend(grid_boxes)

    # 5. Merge nearby / overlapping horizontal boxes
    merged_boxes = merge_bounding_boxes(candidate_boxes, dist_thresh=10, img_width=w, img_height=h)

    # Combine merged horizontal boxes with tight diagonal polygon boxes
    final_boxes = merged_boxes + diagonal_boxes

    # 6. Create box-based mask: tight rotated polygons for diagonal text, boxes for others
    box_mask = np.zeros((h, w), dtype=np.uint8)
    for b in final_boxes:
        if b.get("polygon") and len(b["polygon"]) >= 3:
            pts_np = np.array(b["polygon"], dtype=np.int32)
            poly_single = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_single, [pts_np], 255)
            k_poly = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            poly_dilated = cv2.dilate(poly_single, k_poly, iterations=1)
            box_mask = cv2.bitwise_or(box_mask, poly_dilated)
        else:
            pad = max(2, dilate_pixels)
            x1 = max(0, b["x"] - pad)
            y1 = max(0, b["y"] - pad)
            x2 = min(w, b["x"] + b["width"] + pad)
            y2 = min(h, b["y"] + b["height"] + pad)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)

    recommended_mode = "deblend" if has_translucent_grid else "auto"

    return {
        "mask": box_mask,
        "boxes": final_boxes,
        "ocr_items": ocr_results,
        "detected_count": len(final_boxes),
        "recommended_mode": recommended_mode
    }
