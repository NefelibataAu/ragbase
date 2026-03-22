"""
Unit tests for ragbase.image_parser — image type detection and OCR wrapper.

OCR is mocked so these tests run without Tesseract installed.
"""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from ragbase.image_parser import (
    IMAGE_EXTENSIONS,
    TesseractNotFoundError,
    extract_text_from_image,
    is_image_path,
)


# ---------------------------------------------------------------------------
# is_image_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("photo.png", True),
        ("scan.jpg", True),
        ("picture.jpeg", True),
        ("UPPER.PNG", True),
        ("report.pdf", False),
        ("data.txt", False),
        ("archive.zip", False),
        ("noextension", False),
    ],
)
def test_is_image_path(filename, expected):
    assert is_image_path(Path(filename)) == expected


def test_image_extensions_set():
    """Ensure the canonical extension set is complete."""
    assert IMAGE_EXTENSIONS == {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# extract_text_from_image — success path (mocked pytesseract + Pillow)
# ---------------------------------------------------------------------------


def _make_pil_mock():
    """Return a minimal mock of the PIL.Image / PIL namespace."""
    pil_mock = ModuleType("PIL")
    image_mod = ModuleType("PIL.Image")
    imageops_mod = ModuleType("PIL.ImageOps")
    imageenhance_mod = ModuleType("PIL.ImageEnhance")

    fake_image = MagicMock()
    image_mod.open = MagicMock(return_value=fake_image)
    imageops_mod.grayscale = MagicMock(return_value=fake_image)

    enhance_instance = MagicMock()
    enhance_instance.enhance = MagicMock(return_value=fake_image)
    imageenhance_mod.Contrast = MagicMock(return_value=enhance_instance)

    pil_mock.Image = image_mod
    pil_mock.ImageOps = imageops_mod
    pil_mock.ImageEnhance = imageenhance_mod

    return pil_mock, image_mod, imageops_mod, imageenhance_mod


def test_extract_text_success(tmp_path):
    """extract_text_from_image returns a Document with expected metadata."""
    fake_image_path = tmp_path / "test.png"
    fake_image_path.write_bytes(b"fake-png-data")

    pil_mock, image_mod, imageops_mod, imageenhance_mod = _make_pil_mock()

    pytesseract_mock = ModuleType("pytesseract")
    pytesseract_mock.image_to_string = MagicMock(return_value="Hello OCR world")
    pytesseract_mock.TesseractNotFoundError = type(
        "TesseractNotFoundError", (EnvironmentError,), {}
    )

    with patch.dict(
        sys.modules,
        {
            "pytesseract": pytesseract_mock,
            "PIL": pil_mock,
            "PIL.Image": image_mod,
            "PIL.ImageOps": imageops_mod,
            "PIL.ImageEnhance": imageenhance_mod,
        },
    ):
        doc = extract_text_from_image(fake_image_path)

    assert doc.page_content == "Hello OCR world"
    assert doc.metadata["source"] == "test.png"
    assert doc.metadata["type"] == "image"
    assert doc.metadata["ocr_engine"] == "pytesseract"


def test_extract_text_tesseract_not_found(tmp_path):
    """extract_text_from_image raises TesseractNotFoundError when Tesseract is missing."""
    fake_image_path = tmp_path / "test.png"
    fake_image_path.write_bytes(b"fake-png-data")

    pil_mock, image_mod, imageops_mod, imageenhance_mod = _make_pil_mock()

    class FakeTesseractNotFoundError(EnvironmentError):
        pass

    pytesseract_mock = ModuleType("pytesseract")
    pytesseract_mock.TesseractNotFoundError = FakeTesseractNotFoundError
    pytesseract_mock.image_to_string = MagicMock(
        side_effect=FakeTesseractNotFoundError("not found")
    )

    with patch.dict(
        sys.modules,
        {
            "pytesseract": pytesseract_mock,
            "PIL": pil_mock,
            "PIL.Image": image_mod,
            "PIL.ImageOps": imageops_mod,
            "PIL.ImageEnhance": imageenhance_mod,
        },
    ):
        with pytest.raises(TesseractNotFoundError):
            extract_text_from_image(fake_image_path)


def test_extract_text_import_error(tmp_path, monkeypatch):
    """extract_text_from_image raises ImportError when pytesseract is missing."""
    fake_image_path = tmp_path / "test.png"
    fake_image_path.write_bytes(b"fake-png-data")

    # Remove pytesseract from sys.modules to force ImportError
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "PIL.Image", None)

    with pytest.raises(ImportError, match="pytesseract"):
        extract_text_from_image(fake_image_path)
