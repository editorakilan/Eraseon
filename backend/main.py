"""
FastAPI Backend Server for Watermark Removal Application
Supports single and multi-page PDFs, high-resolution images, auto-detection,
interactive canvas masking, and zero-artifact hybrid removal.
"""

import os
import io
import uuid
import base64
import json
import shutil
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.watermark_engine.remover import hybrid_watermark_removal
from backend.watermark_engine.detector import auto_detect_watermark
from backend.watermark_engine.pdf_handler import PDFDocument, save_images_to_pdf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Eraseon", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session registry (with file persistence)
SESSIONS: Dict[str, Dict[str, Any]] = {}


def get_session_dir(session_id: str) -> str:
    s_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(s_dir, exist_ok=True)
    return s_dir


def get_output_dir(session_id: str) -> str:
    o_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(o_dir, exist_ok=True)
    return o_dir


def parse_color_hex(hex_str: Optional[str]) -> Optional[tuple]:
    if not hex_str or hex_str.lower() in ("auto", "none", ""):
        return None
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        # return BGR
        return (b, g, r)
    return None


@app.get("/")
async def root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Eraseon Backend is Running."}


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "app": "Eraseon", "version": "1.0.0"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads image or PDF and initializes session."""
    session_id = str(uuid.uuid4())
    s_dir = get_session_dir(session_id)
    get_output_dir(session_id)

    filename = file.filename or "uploaded_file"
    ext = os.path.splitext(filename)[1].lower()
    saved_path = os.path.join(s_dir, f"original{ext}")

    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    is_pdf = ext == ".pdf"
    session_info: Dict[str, Any] = {
        "session_id": session_id,
        "filename": filename,
        "file_type": "pdf" if is_pdf else "image",
        "file_path": saved_path,
        "page_count": 1,
        "pages": []
    }

    if is_pdf:
        try:
            pdf_doc = PDFDocument(saved_path)
            meta = pdf_doc.get_info()
            session_info["page_count"] = meta["page_count"]
            session_info["pages"] = meta["pages"]
            pdf_doc.close()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")
    else:
        try:
            with Image.open(saved_path) as pil_img:
                w, h = pil_img.size
                session_info["pages"] = [{"page_num": 0, "width": w, "height": h}]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    SESSIONS[session_id] = session_info
    return session_info


@app.get("/api/page/{session_id}/{page_num}")
async def get_page(session_id: str, page_num: int = 0, scale: float = 2.0):
    """Returns rendered page image (PDF page or uploaded image)."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSIONS[session_id]
    s_dir = get_session_dir(session_id)
    cached_page_path = os.path.join(s_dir, f"rendered_page_{page_num}.png")

    if os.path.exists(cached_page_path) and scale == 2.0:
        return FileResponse(cached_page_path, media_type="image/png")

    if session["file_type"] == "pdf":
        try:
            pdf_doc = PDFDocument(session["file_path"])
            bgr = pdf_doc.render_page(page_num, scale=scale)
            pdf_doc.close()
            cv2.imwrite(cached_page_path, bgr)
            return FileResponse(cached_page_path, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Render error: {str(e)}")
    else:
        # Image
        img_bgr = cv2.imread(session["file_path"], cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise HTTPException(status_code=500, detail="Could not read image file")
        cv2.imwrite(cached_page_path, img_bgr)
        return FileResponse(cached_page_path, media_type="image/png")


class AutoDetectRequest(BaseModel):
    session_id: str
    page_num: int = 0
    sensitivity: float = 0.5
    dilate_pixels: int = 4
    target_type: Optional[str] = "all"


@app.post("/api/auto-detect")
async def auto_detect(req: AutoDetectRequest):
    """Runs automated watermark detection on the given page."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSIONS[req.session_id]
    s_dir = get_session_dir(req.session_id)
    page_img_path = os.path.join(s_dir, f"rendered_page_{req.page_num}.png")

    if not os.path.exists(page_img_path):
        # Render page
        if session["file_type"] == "pdf":
            pdf_doc = PDFDocument(session["file_path"])
            img_bgr = pdf_doc.render_page(req.page_num, scale=2.0)
            pdf_doc.close()
            cv2.imwrite(page_img_path, img_bgr)
        else:
            img_bgr = cv2.imread(session["file_path"])
            cv2.imwrite(page_img_path, img_bgr)
    else:
        img_bgr = cv2.imread(page_img_path)

    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Could not load page image")

    # Run auto detector with specified target_type
    detection_res = auto_detect_watermark(
        img_bgr,
        sensitivity=req.sensitivity,
        dilate_pixels=req.dilate_pixels,
        target_type=req.target_type or "all"
    )

    mask = detection_res["mask"]
    # Encode mask to base64 PNG so frontend canvas can directly draw it
    _, mask_buf = cv2.imencode(".png", mask)
    mask_b64 = "data:image/png;base64," + base64.b64encode(mask_buf).decode("utf-8")

    return {
        "detected_count": detection_res["detected_count"],
        "boxes": detection_res["boxes"],
        "ocr_items": detection_res["ocr_items"],
        "recommended_mode": detection_res.get("recommended_mode", "auto"),
        "mask_b64": mask_b64
    }


class ProcessPageRequest(BaseModel):
    session_id: str
    page_num: int = 0
    mask_b64: str  # Base64 data URL of user mask
    mode: str = "auto"  # auto, deblend, inpaint_telea, inpaint_ns
    alpha_hint: float = 0.5
    watermark_color: Optional[str] = "auto"
    inpaint_radius: int = 5
    match_grain: bool = True


@app.post("/api/process")
async def process_page(req: ProcessPageRequest):
    """Processes a single page/image with the specified mask and algorithm."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    s_dir = get_session_dir(req.session_id)
    o_dir = get_output_dir(req.session_id)
    page_img_path = os.path.join(s_dir, f"rendered_page_{req.page_num}.png")

    if not os.path.exists(page_img_path):
        raise HTTPException(status_code=400, detail="Page not rendered yet")

    img_bgr = cv2.imread(page_img_path)
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to read page image")

    # Decode mask
    try:
        if "," in req.mask_b64:
            header, encoded = req.mask_b64.split(",", 1)
        else:
            encoded = req.mask_b64
        mask_bytes = base64.b64decode(encoded)
        mask_pil = Image.open(io.BytesIO(mask_bytes))
        
        # Check if mask has alpha channel (canvas drawing usually has transparency)
        mask_arr = np.array(mask_pil)
        if len(mask_arr.shape) == 3 and mask_arr.shape[2] == 4:
            # Alpha channel > 0 represents painted areas
            binary_mask = (mask_arr[:, :, 3] > 10).astype(np.uint8) * 255
        elif len(mask_arr.shape) == 3:
            # Grayscale or RGB mask
            gray = cv2.cvtColor(mask_arr, cv2.COLOR_RGB2GRAY)
            binary_mask = (gray > 10).astype(np.uint8) * 255
        else:
            binary_mask = (mask_arr > 10).astype(np.uint8) * 255

        # Resize mask to match image size exactly if necessary
        if binary_mask.shape[:2] != img_bgr.shape[:2]:
            binary_mask = cv2.resize(binary_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Check if mask is completely empty
        if np.count_nonzero(binary_mask) == 0:
            raise HTTPException(
                status_code=400,
                detail="No watermark selected. Please mark the watermark with the Brush or Box tool, or click 'Auto-Detect' / 'AI Sparkles' first."
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid mask: {str(e)}")

    # Parse color
    wm_color = parse_color_hex(req.watermark_color)

    # Run hybrid removal
    cleaned = hybrid_watermark_removal(
        img=img_bgr,
        mask=binary_mask,
        mode=req.mode,
        alpha_hint=req.alpha_hint,
        watermark_color=wm_color,
        inpaint_radius=req.inpaint_radius,
        match_grain=req.match_grain
    )

    out_file = os.path.join(o_dir, f"cleaned_page_{req.page_num}.png")
    cv2.imwrite(out_file, cleaned)

    # Encode cleaned result to base64 for instant preview
    _, buf = cv2.imencode(".png", cleaned)
    cleaned_b64 = "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

    return {
        "status": "success",
        "page_num": req.page_num,
        "cleaned_b64": cleaned_b64,
        "download_url": f"/api/download/{req.session_id}?type=page&page_num={req.page_num}"
    }


class ProcessAllPDFRequest(BaseModel):
    session_id: str
    template_mask_b64: str = ""
    page_masks: Optional[Dict[str, str]] = None
    mode: str = "auto"
    alpha_hint: float = 0.5
    watermark_color: Optional[str] = "auto"
    inpaint_radius: int = 5
    dpi: int = 150


def _decode_b64_mask(mask_b64: str) -> Optional[np.ndarray]:
    if not mask_b64:
        return None
    encoded = mask_b64.split(",", 1)[1] if "," in mask_b64 else mask_b64
    if not encoded.strip():
        return None
    mask_bytes = base64.b64decode(encoded)
    mask_arr = np.array(Image.open(io.BytesIO(mask_bytes)))
    if len(mask_arr.shape) == 3 and mask_arr.shape[2] == 4:
        return (mask_arr[:, :, 3] > 10).astype(np.uint8) * 255
    gray = cv2.cvtColor(mask_arr, cv2.COLOR_RGB2GRAY) if len(mask_arr.shape) == 3 else mask_arr
    return (gray > 10).astype(np.uint8) * 255


@app.post("/api/process-all-pages")
async def process_all_pages(req: ProcessAllPDFRequest):
    """Batch processes all pages of a multi-page PDF and compiles a clean PDF."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSIONS[req.session_id]
    if session["file_type"] != "pdf":
        raise HTTPException(status_code=400, detail="This operation is for PDF files")

    o_dir = get_output_dir(req.session_id)
    pdf_doc = PDFDocument(session["file_path"])
    total_pages = pdf_doc.page_count

    template_mask = _decode_b64_mask(req.template_mask_b64)
    decoded_page_masks: Dict[int, np.ndarray] = {}
    if req.page_masks:
        for k, v in req.page_masks.items():
            try:
                m = _decode_b64_mask(v)
                if m is not None and np.count_nonzero(m) > 0:
                    decoded_page_masks[int(k)] = m
            except Exception:
                pass

    if template_mask is None and not decoded_page_masks:
        pdf_doc.close()
        raise HTTPException(status_code=400, detail="No watermark mask provided for PDF batch cleaning.")

    wm_color = parse_color_hex(req.watermark_color)
    cleaned_images = []
    scale = max(1.5, req.dpi / 72.0)

    for p in range(total_pages):
        page_bgr = pdf_doc.render_page(p, scale=scale)
        src_mask = decoded_page_masks.get(p, template_mask)
        if src_mask is None or np.count_nonzero(src_mask) == 0:
            cleaned_images.append(page_bgr)
            continue

        cur_mask = cv2.resize(src_mask, (page_bgr.shape[1], page_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        cleaned_page = hybrid_watermark_removal(
            img=page_bgr,
            mask=cur_mask,
            mode=req.mode,
            alpha_hint=req.alpha_hint,
            watermark_color=wm_color,
            inpaint_radius=req.inpaint_radius
        )
        cleaned_images.append(cleaned_page)

    pdf_doc.close()

    # Reconstruct PDF
    out_pdf_path = os.path.join(o_dir, "cleaned_document.pdf")
    save_images_to_pdf(cleaned_images, out_pdf_path, dpi=req.dpi)

    return {
        "status": "success",
        "total_pages": total_pages,
        "download_url": f"/api/download/{req.session_id}?type=pdf"
    }


@app.get("/api/download/{session_id}")
async def download_output(
    session_id: str,
    type: str = Query("image", enum=["image", "pdf", "page"]),
    page_num: int = 0
):
    """Downloads cleaned image, single page, or full PDF."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSIONS[session_id]
    o_dir = get_output_dir(session_id)
    base_name = os.path.splitext(session["filename"])[0]

    if type == "pdf":
        pdf_path = os.path.join(o_dir, "cleaned_document.pdf")
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="Cleaned PDF not generated yet")
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"{base_name}_cleaned.pdf"
        )
    else:
        # Single page / Image
        img_path = os.path.join(o_dir, f"cleaned_page_{page_num}.png")
        if not os.path.exists(img_path):
            raise HTTPException(status_code=404, detail="Cleaned page not generated yet")
        return FileResponse(
            img_path,
            media_type="image/png",
            filename=f"{base_name}_page_{page_num + 1}_cleaned.png"
        )


# Mount static directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
