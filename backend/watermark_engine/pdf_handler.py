"""
PDF Processing and Reconstruction Engine
Powered by pypdfium2 for ultra-fast, high-fidelity rasterization and PDF generation.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import pypdfium2 as pdfium
import numpy as np
from PIL import Image
import cv2


class PDFDocument:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.pdf = pdfium.PdfDocument(file_path)
        self.page_count = len(self.pdf)

    def get_info(self) -> Dict[str, Any]:
        """Return document metadata and page dimensions."""
        pages_info = []
        for i in range(self.page_count):
            page = self.pdf.get_page(i)
            w, h = page.get_size()
            pages_info.append({"page_num": i, "width": int(w), "height": int(h)})
        return {
            "page_count": self.page_count,
            "pages": pages_info
        }

    def render_page(self, page_index: int, scale: float = 2.0) -> np.ndarray:
        """
        Renders a PDF page to a high-quality BGR numpy array.
        scale=2.0 roughly corresponds to 144-150 DPI (crisp text & photos).
        """
        if page_index < 0 or page_index >= self.page_count:
            raise IndexError(f"Page index {page_index} out of range [0, {self.page_count - 1}]")

        page = self.pdf.get_page(page_index)
        # Render page with high quality anti-aliasing
        bitmap = page.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()
        # Convert PIL (RGB) to OpenCV (BGR)
        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return bgr

    def close(self):
        try:
            self.pdf.close()
        except Exception:
            pass


def save_images_to_pdf(image_list: List[np.ndarray], output_pdf_path: str, dpi: int = 150):
    """
    Assembles a list of BGR numpy images into a multi-page PDF.
    """
    if not image_list:
        raise ValueError("No images provided for PDF assembly.")

    pil_images = []
    for img_bgr in image_list:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))

    # Save first page and append remaining
    first_page = pil_images[0]
    rest_pages = pil_images[1:] if len(pil_images) > 1 else []

    first_page.save(
        output_pdf_path,
        "PDF",
        resolution=float(dpi),
        save_all=True,
        append_images=rest_pages
    )
