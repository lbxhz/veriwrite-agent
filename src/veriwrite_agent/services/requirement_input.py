"""Read requirement text from supported V0.1 file formats."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Literal, Sequence
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from veriwrite_agent.services.ocr import (
    OCRNoTextError,
    OCRTextResult,
    OCRUnavailableError,
    extract_image_text,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MAX_OCR_PDF_PAGES = 30


class UnsupportedRequirementFileError(ValueError):
    """Raised when the input must be converted before parsing."""


class RequirementTextExtractionError(ValueError):
    """Raised when a supported file contains no usable machine-readable text."""


@dataclass(frozen=True)
class RequirementTextResult:
    """Text plus extraction provenance shown by the validation workbench."""

    text: str
    method: Literal["native", "ocr", "mixed"]
    ocr_average_confidence: float | None = None
    warnings: tuple[str, ...] = ()
    source_count: int = 1


def load_requirement_text(path: Path) -> str:
    """Compatibility wrapper returning only the extracted text."""

    return extract_requirement_text(path).text


def extract_requirement_text(path: Path) -> RequirementTextResult:
    """Load text from TXT, Markdown, DOCX, legacy DOC, or text-based PDF."""

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return RequirementTextResult(text=_read_plain_text(path), method="native")
    if suffix == ".docx":
        return RequirementTextResult(text=_read_docx(path), method="native")
    if suffix == ".doc":
        return RequirementTextResult(text=_read_legacy_doc(path), method="native")
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return _read_image(path)
    raise UnsupportedRequirementFileError(
        f"不支持 {suffix or '无扩展名'} 文件；V0.1 支持文本、Word、PDF 和常见图片格式。"
    )


def extract_requirement_texts(
    paths: Sequence[Path],
) -> RequirementTextResult:
    """Extract and merge ordered files, including overlapping phone screenshots."""

    if not paths:
        raise RequirementTextExtractionError("至少需要一个要求文件。")
    results = [extract_requirement_text(path) for path in paths]
    cleaned = [_clean_extracted_text(result.text) for result in results]
    merged = cleaned[0]
    overlap_count = 0
    for next_text in cleaned[1:]:
        merged, overlapped = _merge_overlapping_text(merged, next_text)
        overlap_count += int(overlapped)

    methods = {result.method for result in results}
    if methods == {"native"}:
        method: Literal["native", "ocr", "mixed"] = "native"
    elif methods == {"ocr"}:
        method = "ocr"
    else:
        method = "mixed"

    confidences = [
        (
            result.ocr_average_confidence,
            max(len(cleaned[index]), 1),
        )
        for index, result in enumerate(results)
        if result.ocr_average_confidence is not None
    ]
    confidence = None
    if confidences:
        total_weight = sum(weight for _, weight in confidences)
        confidence = sum(value * weight for value, weight in confidences) / total_weight

    warnings = list(dict.fromkeys(warning for result in results for warning in result.warnings))
    if len(paths) > 1:
        warnings.append(
            f"已按上传顺序合并 {len(paths)} 个文件；检测并去除了 {overlap_count} 处滚动截图重叠。"
        )
        warnings.append(
            "手机工具栏遮挡的文字无法由 OCR 恢复；请在“原文与 JSON”中核对合并文本并可校正后重跑。"
        )
    return RequirementTextResult(
        text=merged,
        method=method,
        ocr_average_confidence=confidence,
        warnings=tuple(warnings),
        source_count=len(paths),
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
        raise UnsupportedRequirementFileError(f"{path} 不是有效的 .docx 文件。") from exc

    root = ElementTree.fromstring(document_xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t")).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _read_pdf(path: Path) -> RequirementTextResult:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise UnsupportedRequirementFileError(
            '读取 PDF 需要安装 UI 依赖：pip install -e ".[ui]"'
        ) from exc

    try:
        reader = PdfReader(path)
        extracted_pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        raise RequirementTextExtractionError(f"无法读取 PDF：{path.name}") from exc

    missing_indexes = [index for index, text in enumerate(extracted_pages) if not text]
    if len(missing_indexes) > MAX_OCR_PDF_PAGES:
        raise RequirementTextExtractionError(
            f"PDF 有 {len(missing_indexes)} 个扫描页，超过单次 OCR 上限 {MAX_OCR_PDF_PAGES} 页。"
        )

    ocr_results: dict[int, OCRTextResult] = {}
    empty_page_indexes: list[int] = []
    for index in missing_indexes:
        result = _ocr_pdf_page(path, index)
        if result is None:
            empty_page_indexes.append(index)
        else:
            ocr_results[index] = result

    pages: list[str] = []
    for index, native_text in enumerate(extracted_pages):
        page_number = index + 1
        if native_text:
            pages.append(f"[PDF_PAGE_{page_number}]\n{native_text}")
        elif index in ocr_results:
            pages.append(f"[OCR_PDF_PAGE_{page_number}]\n{ocr_results[index].text}")
        else:
            pages.append(f"[EMPTY_PDF_PAGE_{page_number}]")

    if not any(extracted_pages) and not ocr_results:
        raise RequirementTextExtractionError("PDF 没有原生文本，OCR 也没有识别到可用文字。")

    has_native = any(extracted_pages)
    has_ocr = bool(ocr_results)
    method: Literal["native", "ocr", "mixed"]
    if has_native and has_ocr:
        method = "mixed"
    elif has_ocr:
        method = "ocr"
    else:
        method = "native"

    confidence = _weighted_ocr_confidence(ocr_results.values())
    warnings = _ocr_warnings(confidence)
    if method == "mixed":
        warnings = (
            "PDF 同时包含原生文本页和扫描页，仅对无文本页执行了 OCR。",
            *warnings,
        )
    if empty_page_indexes:
        page_numbers = "、".join(str(index + 1) for index in empty_page_indexes)
        warnings = (
            *warnings,
            f"PDF 第 {page_numbers} 页没有原生文本，OCR 也未识别到文字，已按空白页保留。",
        )
    return RequirementTextResult(
        text="\n\n".join(pages),
        method=method,
        ocr_average_confidence=confidence,
        warnings=warnings,
    )


def _read_image(path: Path) -> RequirementTextResult:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise UnsupportedRequirementFileError(
            '读取图片需要安装 OCR 依赖：pip install -e ".[ocr]"'
        ) from exc

    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            result = extract_image_text(image)
    except (OCRUnavailableError, OCRNoTextError) as exc:
        raise RequirementTextExtractionError(str(exc)) from exc
    except Exception as exc:
        raise RequirementTextExtractionError(f"无法读取或识别图片：{path.name}") from exc

    return RequirementTextResult(
        text=f"[OCR_IMAGE]\n{_clean_extracted_text(result.text)}",
        method="ocr",
        ocr_average_confidence=result.average_confidence,
        warnings=_ocr_warnings(result.average_confidence),
    )


def _clean_extracted_text(text: str) -> str:
    """Remove obvious mobile chrome while preserving document wording."""

    exact_noise = {
        "0",
        "目",
        "品",
        "编辑",
        "分页视图",
        "大纲",
        "AI",
        "操作",
        "148",
        "三",
        "C",
        "O",
        "[OCR_IMAGE]",
    }
    replacements = {
        "SARInSAR": "SAR/InSAR",
        "Al Tool Usage Declaration": "AI Tool Usage Declaration",
        "Al Tool Name": "AI Tool Name",
        "no Al tools": "no AI tools",
        "No Al tools": "No AI tools",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)

    lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = " ".join(raw_line.split())
        if not line or line in exact_noise:
            continue
        if re.fullmatch(r"\d{1,2}:\d{2}[0.…]*", line):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?\s*(?:KB/s)?", line, re.IGNORECASE):
            continue
        if line.casefold() == "kb/s":
            continue
        lines.append(line)
    return "\n".join(lines)


def _merge_overlapping_text(left: str, right: str) -> tuple[str, bool]:
    """Prefer the later screenshot when two scroll captures overlap."""

    left_norm, left_map = _normalised_with_map(left)
    right_norm, _ = _normalised_with_map(right)
    if not left_norm or not right_norm:
        return "\n\n".join(part for part in (left, right) if part), False

    left_start = max(0, len(left_norm) - 3000)
    left_tail = left_norm[left_start:]
    right_head = right_norm[:3000]
    match = SequenceMatcher(
        None,
        left_tail,
        right_head,
        autojunk=False,
    ).find_longest_match()
    if match.size < 35 or match.b > 160:
        return f"{left.rstrip()}\n\n{right.lstrip()}", False

    absolute_match = left_start + match.a
    cut_at = left_map[absolute_match]
    return f"{left[:cut_at].rstrip()}\n\n{right.lstrip()}", True


def _normalised_with_map(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            normalized.append(char.casefold())
            positions.append(index)
    return "".join(normalized), positions


def _ocr_pdf_page(path: Path, page_index: int) -> OCRTextResult | None:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise UnsupportedRequirementFileError(
            '扫描 PDF OCR 需要安装依赖：pip install -e ".[ocr]"'
        ) from exc

    document = None
    page = None
    bitmap = None
    try:
        document = pdfium.PdfDocument(str(path))
        page = document[page_index]
        bitmap = page.render(scale=2.5)
        image = bitmap.to_pil().convert("RGB")
        return extract_image_text(image)
    except OCRNoTextError:
        return None
    except OCRUnavailableError as exc:
        raise RequirementTextExtractionError(f"PDF 第 {page_index + 1} 页：{exc}") from exc
    except Exception as exc:
        raise RequirementTextExtractionError(f"无法 OCR PDF 第 {page_index + 1} 页。") from exc
    finally:
        if bitmap is not None:
            bitmap.close()
        if page is not None:
            page.close()
        if document is not None:
            document.close()


def _weighted_ocr_confidence(
    results: Iterable[OCRTextResult],
) -> float | None:
    items = list(results)
    total_lines = sum(result.line_count for result in items)
    if not total_lines:
        return None
    return sum(result.average_confidence * result.line_count for result in items) / total_lines


def _ocr_warnings(confidence: float | None) -> tuple[str, ...]:
    if confidence is None:
        return ()
    if confidence < 0.8:
        return (f"OCR 平均置信度为 {confidence:.1%}，建议先核对提取文本再执行双路解析。",)
    return ()


def _read_legacy_doc(path: Path) -> str:
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise UnsupportedRequirementFileError(
            "自动读取 .doc 需要 Windows、Microsoft Word 和 pywin32；也可以先手动转换为 .docx。"
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
        raise RequirementTextExtractionError(f"Word 无法转换旧版 DOC：{path.name}") from exc
    finally:
        if document is not None:
            document.Close(False)
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()
