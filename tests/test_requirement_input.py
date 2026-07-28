import sys
from zipfile import ZipFile

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from veriwrite_agent.services.requirement_input import (
    RequirementTextExtractionError,
    UnsupportedRequirementFileError,
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


def test_scanned_pdf_reports_that_ocr_is_required(tmp_path) -> None:
    path = tmp_path / "scanned.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as output:
        writer.write(output)

    with pytest.raises(RequirementTextExtractionError, match="OCR"):
        load_requirement_text(path)
