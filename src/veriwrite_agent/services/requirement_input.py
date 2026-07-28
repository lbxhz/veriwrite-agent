"""Read requirement text from supported V0.1 file formats."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


class UnsupportedRequirementFileError(ValueError):
    """Raised when the input must be converted before parsing."""


def load_requirement_text(path: Path) -> str:
    """Load UTF-8 text/Markdown or extract paragraph text from a DOCX."""

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".doc":
        raise UnsupportedRequirementFileError(
            "旧版 .doc 文件需要先用 Word 或 LibreOffice 转换为 .docx。"
        )
    raise UnsupportedRequirementFileError(
        f"不支持 {suffix or '无扩展名'} 文件；V0.1 支持 .txt、.md 和 .docx。"
    )


def _read_docx(path: Path) -> str:
    try:
        with ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
    except (BadZipFile, KeyError) as exc:
        raise UnsupportedRequirementFileError(
            f"{path} 不是有效的 .docx 文件。"
        ) from exc

    root = ElementTree.fromstring(document_xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        text = "".join(
            node.text or "" for node in paragraph.iter(f"{namespace}t")
        ).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)
