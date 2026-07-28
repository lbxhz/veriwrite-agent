import sys
from zipfile import ZipFile

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from veriwrite_agent.services import requirement_input
from veriwrite_agent.services.ocr import OCRNoTextError, OCRTextResult
from veriwrite_agent.services.requirement_input import (
    RequirementTextExtractionError,
    UnsupportedRequirementFileError,
    extract_requirement_text,
    load_requirement_text,
)


def test_loads_gb18030_plain_text(tmp_path) -> None:
    path = tmp_path / "requirements.txt"
    path.write_bytes("至少需要60篇参考文献。".encode("gb18030"))

    assert load_requirement_text(path) == "至少需要60篇参考文献。"


def test_loads_docx_paragraphs_without_external_word_dependency(tmp_path) -> None:
    path = tmp_path / "requirements.docx"
    document_xml = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>至少15000字</w:t></w:r></w:p>
    <w:p><w:r><w:t>参考文献不少于60篇</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    text = load_requirement_text(path)

    assert text == "至少15000字\n参考文献不少于60篇"


def test_old_doc_explains_word_dependency_when_converter_is_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "requirements.doc"
    path.write_bytes(b"legacy")
    monkeypatch.setitem(sys.modules, "pythoncom", None)

    with pytest.raises(
        UnsupportedRequirementFileError,
        match="Microsoft Word",
    ):
        load_requirement_text(path)


def test_loads_text_from_pdf_with_page_evidence(tmp_path) -> None:
    path = tmp_path / "requirements.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_reference}
            )
        }
    )
    content = DecodedStreamObject()
    content.set_data(
        b"BT /F1 12 Tf 72 720 Td (Minimum 15000 chars) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(content)
    with path.open("wb") as output:
        writer.write(output)

    text = load_requirement_text(path)

    assert "[PDF_PAGE_1]" in text
    assert "Minimum 15000 chars" in text


def test_scanned_pdf_uses_local_ocr(tmp_path, monkeypatch) -> None:
    path = tmp_path / "scanned.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as output:
        writer.write(output)
    monkeypatch.setattr(
        requirement_input,
        "_ocr_pdf_page",
        lambda path, index: OCRTextResult(
            text="课程综述15000字以上",
            average_confidence=0.94,
            line_count=1,
        ),
    )

    result = extract_requirement_text(path)

    assert result.method == "ocr"
    assert result.ocr_average_confidence == pytest.approx(0.94)
    assert "[OCR_PDF_PAGE_1]" in result.text


def test_blank_pdf_still_reports_no_usable_text(tmp_path, monkeypatch) -> None:
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as output:
        writer.write(output)
    monkeypatch.setattr(
        requirement_input,
        "_ocr_pdf_page",
        lambda path, index: None,
    )

    with pytest.raises(RequirementTextExtractionError, match="没有识别到"):
        extract_requirement_text(path)


def test_image_input_reports_ocr_method_and_low_confidence(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "phone-photo.png"
    Image.new("RGB", (200, 100), "white").save(path)
    monkeypatch.setattr(
        requirement_input,
        "extract_image_text",
        lambda image: OCRTextResult(
            text="参考文献不少于60篇",
            average_confidence=0.72,
            line_count=1,
        ),
    )

    result = extract_requirement_text(path)

    assert result.method == "ocr"
    assert result.text.endswith("参考文献不少于60篇")
    assert result.warnings
    assert "72.0%" in result.warnings[0]


def test_image_without_recognizable_text_has_clear_error(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "blank.jpg"
    Image.new("RGB", (100, 100), "white").save(path)

    def no_text(image):
        raise OCRNoTextError("OCR 没有识别到可用文本。")

    monkeypatch.setattr(requirement_input, "extract_image_text", no_text)

    with pytest.raises(RequirementTextExtractionError, match="OCR"):
        load_requirement_text(path)
