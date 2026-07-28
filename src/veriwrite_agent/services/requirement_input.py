"""Read requirement text from supported V0.1 file formats."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


class UnsupportedRequirementFileError(ValueError):
    """Raised when the input must be converted before parsing."""


class RequirementTextExtractionError(ValueError):
    """Raised when a supported file contains no usable machine-readable text."""


def load_requirement_text(path: Path) -> str:
    """Load text from TXT, Markdown, DOCX, legacy DOC, or text-based PDF."""

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _read_plain_text(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".doc":
        return _read_legacy_doc(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise UnsupportedRequirementFileError(
        f"不支持 {suffix or '无扩展名'} 文件；"
        "V0.1 支持 .txt、.md、.docx、.doc 和 .pdf。"
    )


def _read_plain_text(path: Path) -> str:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise RequirementTextExtractionError(
        f"无法识别 {path.name} 的文本编码；请转换为 UTF-8 或 GB18030。"
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


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise UnsupportedRequirementFileError(
            "读取 PDF 需要安装 UI 依赖：pip install -e \".[ui]\""
        ) from exc

    try:
        reader = PdfReader(path)
        pages = [
            f"[PDF_PAGE_{index}]\n{page.extract_text() or ''}"
            for index, page in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:
        raise RequirementTextExtractionError(
            f"无法读取 PDF：{path.name}"
        ) from exc

    text = "\n\n".join(pages).strip()
    visible_text = "\n".join(
        line for line in text.splitlines() if not line.startswith("[PDF_PAGE_")
    ).strip()
    if not visible_text:
        raise RequirementTextExtractionError(
            "PDF 没有可提取文本，可能是扫描件；需要先执行 OCR。"
        )
    return text


def _read_legacy_doc(path: Path) -> str:
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise UnsupportedRequirementFileError(
            "自动读取 .doc 需要 Windows、Microsoft Word 和 pywin32；"
            "也可以先手动转换为 .docx。"
        ) from exc

    pythoncom.CoInitialize()
    word = None
    document = None
    try:
        with TemporaryDirectory(prefix="veriwrite-doc-") as temp_dir:
            converted = Path(temp_dir) / f"{path.stem}.docx"
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0
            word.AutomationSecurity = 3
            document = word.Documents.Open(
                FileName=str(path.resolve()),
                ConfirmConversions=False,
                ReadOnly=True,
                AddToRecentFiles=False,
                OpenAndRepair=True,
                NoEncodingDialog=True,
            )
            document.SaveAs2(
                FileName=str(converted),
                FileFormat=16,
                AddToRecentFiles=False,
            )
            document.Close(False)
            document = None
            return _read_docx(converted)
    except Exception as exc:
        raise RequirementTextExtractionError(
            f"Word 无法转换旧版 DOC：{path.name}"
        ) from exc
    finally:
        if document is not None:
            document.Close(False)
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()
