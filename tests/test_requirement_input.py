from zipfile import ZipFile

import pytest

from veriwrite_agent.services.requirement_input import (
    UnsupportedRequirementFileError,
    load_requirement_text,
)


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


def test_old_doc_requires_conversion(tmp_path) -> None:
    path = tmp_path / "requirements.doc"
    path.write_bytes(b"legacy")

    with pytest.raises(UnsupportedRequirementFileError, match="转换为 .docx"):
        load_requirement_text(path)
