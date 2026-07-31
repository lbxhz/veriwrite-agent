"""Page-preserving native text and optional OCR extraction for verified PDFs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader

from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    DocumentExtractionIssue,
    DocumentExtractionResult,
    DocumentPage,
)
from veriwrite_agent.services.ocr import (
    OCRNoTextError,
    OCRUnavailableError,
    extract_image_text,
)


class PdfPageExtractor:
    """Extract every PDF page without losing page identity or file identity."""

    def __init__(
        self,
        *,
        enable_ocr: bool = True,
        min_native_chars: int = 20,
    ) -> None:
        self.enable_ocr = enable_ocr
        self.min_native_chars = min_native_chars

    def extract(
        self,
        acquisition: DocumentAcquisition,
    ) -> DocumentExtractionResult:
        if acquisition.status != "available" or not acquisition.local_path:
            raise ValueError("PDF extraction requires an available document")
        path = Path(acquisition.local_path).expanduser().resolve()
        if not path.is_file():
            return self._failed(
                acquisition,
                "file_missing",
                f"PDF文件不存在：{path}",
            )

        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != acquisition.sha256:
            return self._failed(
                acquisition,
                "hash_mismatch",
                "PDF当前哈希与下载检查阶段记录的哈希不一致。",
            )

        try:
            reader = PdfReader(path, strict=False)
            page_count = len(reader.pages)
        except Exception as exc:
            return self._failed(
                acquisition,
                "pdf_unreadable",
                f"无法解析PDF：{exc}",
            )

        pages: list[DocumentPage] = []
        issues: list[DocumentExtractionIssue] = []
        ocr_document = None
        for page_number, page in enumerate(reader.pages, 1):
            try:
                native_text = page.extract_text() or ""
            except Exception:
                native_text = ""
            native_text = _normalize_page_text(native_text)
            if len(native_text) >= self.min_native_chars:
                pages.append(
                    DocumentPage(
                        doi=acquisition.doi,
                        document_sha256=actual_sha256,
                        page_number=page_number,
                        text=native_text,
                        extraction_method="native_text",
                    )
                )
                continue

            if self.enable_ocr:
                try:
                    if ocr_document is None:
                        import pypdfium2

                        ocr_document = pypdfium2.PdfDocument(str(path))
                    image = ocr_document[page_number - 1].render(scale=2).to_pil()
                    ocr = extract_image_text(image)
                except (ImportError, OCRUnavailableError) as exc:
                    issues.append(
                        DocumentExtractionIssue(
                            code="ocr_unavailable",
                            page_number=page_number,
                            severity="warning",
                            detail=f"第{page_number}页需要OCR，但运行时不可用：{exc}",
                        )
                    )
                    continue
                except OCRNoTextError:
                    pass
                except Exception as exc:
                    issues.append(
                        DocumentExtractionIssue(
                            code="page_text_missing",
                            page_number=page_number,
                            severity="warning",
                            detail=f"第{page_number}页OCR失败：{exc}",
                        )
                    )
                    continue
                else:
                    pages.append(
                        DocumentPage(
                            doi=acquisition.doi,
                            document_sha256=actual_sha256,
                            page_number=page_number,
                            text=ocr.text,
                            extraction_method=(
                                "hybrid" if native_text else "ocr"
                            ),
                            ocr_confidence=ocr.average_confidence,
                        )
                    )
                    continue

            issues.append(
                DocumentExtractionIssue(
                    code="page_text_missing",
                    page_number=page_number,
                    severity="warning",
                    detail=f"第{page_number}页没有足够的原生文本，需要OCR或人工检查。",
                )
            )

        status = "complete" if len(pages) == page_count else "needs_ocr"
        return DocumentExtractionResult(
            doi=acquisition.doi,
            document_sha256=actual_sha256,
            status=status,
            page_count=page_count,
            pages=pages,
            issues=issues,
        )

    @staticmethod
    def _failed(
        acquisition: DocumentAcquisition,
        code: str,
        detail: str,
    ) -> DocumentExtractionResult:
        return DocumentExtractionResult(
            doi=acquisition.doi,
            document_sha256=acquisition.sha256 or "0" * 64,
            status="failed",
            page_count=0,
            issues=[
                DocumentExtractionIssue(
                    code=code,
                    severity="blocking",
                    detail=detail,
                )
            ],
        )


def _normalize_page_text(value: str) -> str:
    return "\n".join(
        line for line in (" ".join(line.split()) for line in value.splitlines()) if line
    )
