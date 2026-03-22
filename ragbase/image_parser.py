"""
OCR-based image parser for RagBase.

Extracts text from image files (PNG/JPG/JPEG) using pytesseract + Pillow and
returns a LangChain Document so the result can flow through the existing
chunking / embedding / vector-store pipeline unchanged.

Graceful fallback
-----------------
If Tesseract is not installed the module raises ``TesseractNotFoundError``
(a subclass of ``RuntimeError``) so the caller can display a clear error
in the Streamlit UI.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def is_image_path(path: Path) -> bool:
    """Return True if *path* points to a supported image file."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


class TesseractNotFoundError(RuntimeError):
    """Raised when the Tesseract OCR binary cannot be found."""


def extract_text_from_image(image_path: Path) -> Document:
    """Run OCR on *image_path* and return a :class:`~langchain_core.documents.Document`.

    The document's ``metadata`` includes:
    - ``source``: the file name
    - ``type``: ``"image"``
    - ``ocr_engine``: ``"pytesseract"``

    Raises
    ------
    TesseractNotFoundError
        When the Tesseract binary is not installed or not found on ``PATH``.
    ImportError
        When ``pytesseract`` or ``Pillow`` are not installed in the Python env.
    """
    try:
        import pytesseract
        from PIL import Image, ImageEnhance, ImageOps
    except ImportError as exc:
        raise ImportError(
            "pytesseract and Pillow are required for image ingestion. "
            "Install them with: pip install pytesseract Pillow"
        ) from exc

    try:
        image = Image.open(image_path)
        # Minimal preprocessing: grayscale + slight contrast boost
        image = ImageOps.grayscale(image)
        image = ImageEnhance.Contrast(image).enhance(1.5)
        text = pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractNotFoundError(
            "Tesseract OCR is not installed or not found on PATH.\n"
            "Install it with:\n"
            "  macOS:   brew install tesseract\n"
            "  Ubuntu:  sudo apt-get install tesseract-ocr\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc

    return Document(
        page_content=text,
        metadata={
            "source": image_path.name,
            "type": "image",
            "ocr_engine": "pytesseract",
        },
    )
