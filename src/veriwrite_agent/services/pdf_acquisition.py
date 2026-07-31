"""Human-in-the-loop PDF acquisition and deterministic file inspection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from veriwrite_agent.models.evidence import (
    CorePaperExpectation,
    DocumentAcquisition,
    PdfIdentityBasis,
    PdfInspectionBatch,
    PdfInspectionIssue,
    PdfInspectionReport,
)

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class _PdfSnapshot:
    path: Path
    sha256: str | None
    file_size_bytes: int | None
    page_count: int | None
    extractable_page_count: int
    text: str
    metadata: str
    issues: tuple[PdfInspectionIssue, ...]


class PdfAcquisitionInspector:
    """Match downloaded PDFs to core papers and check identity and integrity."""

    def __init__(self, *, max_files: int = 100, text_page_limit: int = 8) -> None:
        self.max_files = max_files
        self.text_page_limit = text_page_limit

    def inspect_file(
        self,
        expectation: CorePaperExpectation,
        path: str | Path,
    ) -> PdfInspectionReport:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            return self._missing_report(expectation)
        snapshot = self._read_snapshot(candidate)
        return self._report(expectation, snapshot)

    def scan_download_directory(
        self,
        expectations: list[CorePaperExpectation],
        directory: str | Path,
    ) -> PdfInspectionBatch:
        download_directory = Path(directory).expanduser().resolve()
        if not download_directory.is_dir():
            return PdfInspectionBatch(
                download_directory=str(download_directory),
                inspected_file_count=0,
                reports=[
                    self._missing_report(
                        expectation,
                        detail=f"下载目录不存在：{download_directory}",
                    )
                    for expectation in expectations
                ],
            )

        paths = sorted(
            (
                path
                for path in download_directory.iterdir()
                if path.is_file()
                and path.suffix.casefold() in {".pdf", ".htm", ".html"}
            ),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[: self.max_files]
        snapshots = [self._read_snapshot(path) for path in paths]
        assignments: dict[str, tuple[_PdfSnapshot, float]] = {}
        used_paths: set[Path] = set()

        scored_pairs: list[
            tuple[float, CorePaperExpectation, _PdfSnapshot]
        ] = []
        for expectation in expectations:
            for snapshot in snapshots:
                score, _ = self._identity_score(expectation, snapshot)
                if score >= 0.8:
                    scored_pairs.append((score, expectation, snapshot))
        scored_pairs.sort(key=lambda item: item[0], reverse=True)

        for score, expectation, snapshot in scored_pairs:
            if expectation.doi in assignments or snapshot.path in used_paths:
                continue
            assignments[expectation.doi] = (snapshot, score)
            used_paths.add(snapshot.path)

        reports = []
        for expectation in expectations:
            assignment = assignments.get(expectation.doi)
            if assignment is None:
                reports.append(self._missing_report(expectation))
            else:
                reports.append(self._report(expectation, assignment[0]))

        return PdfInspectionBatch(
            download_directory=str(download_directory),
            inspected_file_count=len(snapshots),
            reports=reports,
            unmatched_files=[
                str(snapshot.path)
                for snapshot in snapshots
                if snapshot.path not in used_paths
            ],
        )

    def to_document_acquisitions(
        self,
        batch: PdfInspectionBatch,
    ) -> list[DocumentAcquisition]:
        acquisitions: list[DocumentAcquisition] = []
        for report in batch.reports:
            expectation = report.expectation
            if report.status == "verified":
                acquisitions.append(
                    DocumentAcquisition(
                        doi=expectation.doi,
                        status="available",
                        method="user_upload",
                        source_url=expectation.source_url,
                        local_path=report.local_path,
                        sha256=report.sha256,
                        media_type="application/pdf",
                        file_size_bytes=report.file_size_bytes,
                        attempts=1,
                        acquired_at=report.inspected_at,
                    )
                )
                continue
            reason_codes = [issue.code for issue in report.issues]
            acquisitions.append(
                DocumentAcquisition(
                    doi=expectation.doi,
                    status="upload_required",
                    method="none",
                    source_url=expectation.source_url,
                    attempts=1,
                    reason_codes=reason_codes or ["user_download_pending"],
                )
            )
        return acquisitions

    def _read_snapshot(self, path: Path) -> _PdfSnapshot:
        issues: list[PdfInspectionIssue] = []
        try:
            payload = path.read_bytes()
        except OSError as exc:
            return _PdfSnapshot(
                path=path,
                sha256=None,
                file_size_bytes=None,
                page_count=None,
                extractable_page_count=0,
                text="",
                metadata="",
                issues=(
                    PdfInspectionIssue(
                        code="pdf_unreadable",
                        severity="blocking",
                        detail=f"无法读取文件：{exc}",
                    ),
                ),
            )

        sha256 = hashlib.sha256(payload).hexdigest()
        if not payload.startswith(b"%PDF-"):
            return _PdfSnapshot(
                path=path,
                sha256=sha256,
                file_size_bytes=len(payload),
                page_count=None,
                extractable_page_count=0,
                text=payload.decode("utf-8", errors="ignore"),
                metadata="",
                issues=(
                    PdfInspectionIssue(
                        code="not_pdf",
                        severity="blocking",
                        detail=(
                            "下载结果不是 PDF；可能保存了出版社网页、"
                            "登录页或人机验证拦截页。"
                        ),
                    ),
                ),
            )
        if b"%%EOF" not in payload[-4096:]:
            issues.append(
                PdfInspectionIssue(
                    code="missing_eof_marker",
                    severity="warning",
                    detail="文件末尾未发现 PDF EOF 标记，可能下载不完整。",
                )
            )

        page_count: int | None = None
        extractable_page_count = 0
        text_parts: list[str] = []
        metadata_text = ""
        try:
            reader = PdfReader(path, strict=False)
            if reader.is_encrypted:
                try:
                    unlocked = reader.decrypt("")
                except Exception:
                    unlocked = 0
                if not unlocked:
                    issues.append(
                        PdfInspectionIssue(
                            code="pdf_encrypted",
                            severity="blocking",
                            detail="PDF 已加密，系统无法读取正文。",
                        )
                    )
            page_count = len(reader.pages)
            if page_count == 0:
                issues.append(
                    PdfInspectionIssue(
                        code="empty_pdf",
                        severity="blocking",
                        detail="PDF 不包含任何页面。",
                    )
                )
            metadata = reader.metadata or {}
            metadata_text = " ".join(
                str(value) for value in metadata.values() if value
            )
            for page in reader.pages[: self.text_page_limit]:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                if page_text.strip():
                    extractable_page_count += 1
                    text_parts.append(page_text)
        except Exception as exc:
            issues.append(
                PdfInspectionIssue(
                    code="pdf_unreadable",
                    severity="blocking",
                    detail=f"PDF 结构无法解析：{exc}",
                )
            )

        if page_count and extractable_page_count == 0:
            issues.append(
                PdfInspectionIssue(
                    code="text_not_extractable",
                    severity="warning",
                    detail="PDF 可打开，但前几页没有可提取文本，后续可能需要 OCR。",
                )
            )
        return _PdfSnapshot(
            path=path,
            sha256=sha256,
            file_size_bytes=len(payload),
            page_count=page_count,
            extractable_page_count=extractable_page_count,
            text="\n".join(text_parts),
            metadata=metadata_text,
            issues=tuple(issues),
        )

    def _report(
        self,
        expectation: CorePaperExpectation,
        snapshot: _PdfSnapshot,
    ) -> PdfInspectionReport:
        score, basis = self._identity_score(expectation, snapshot)
        issues = list(snapshot.issues)
        blocking = any(issue.severity == "blocking" for issue in issues)
        if score < 0.8:
            issues.append(
                PdfInspectionIssue(
                    code="identity_not_confirmed",
                    severity="blocking",
                    detail="PDF 中的 DOI/题名不足以确认它就是目标论文。",
                )
            )
            blocking = True

        if blocking:
            status = "invalid" if any(
                issue.code
                in {"not_pdf", "pdf_unreadable", "pdf_encrypted", "empty_pdf"}
                for issue in issues
            ) else "needs_review"
        elif any(issue.severity == "warning" for issue in issues):
            status = "needs_review"
        else:
            status = "verified"

        return PdfInspectionReport(
            expectation=expectation,
            status=status,
            local_path=str(snapshot.path),
            sha256=snapshot.sha256,
            file_size_bytes=snapshot.file_size_bytes,
            page_count=snapshot.page_count,
            extractable_page_count=snapshot.extractable_page_count,
            identity_score=score,
            identity_basis=basis,
            issues=issues,
        )

    def _identity_score(
        self,
        expectation: CorePaperExpectation,
        snapshot: _PdfSnapshot,
    ) -> tuple[float, list[PdfIdentityBasis]]:
        expected_doi = expectation.doi.casefold()
        text = snapshot.text.casefold()
        metadata = snapshot.metadata.casefold()
        filename = snapshot.path.stem.casefold()
        basis: list[PdfIdentityBasis] = []

        if expected_doi in text or expected_doi in _extract_dois(text):
            basis.append("doi_text")
        if expected_doi in metadata or expected_doi in _extract_dois(metadata):
            basis.append("doi_metadata")
        filename_doi = re.sub(r"[^a-z0-9]", "", expected_doi)
        if filename_doi and filename_doi in re.sub(r"[^a-z0-9]", "", filename):
            basis.append("filename")

        title_text_score = _title_coverage(expectation.title, snapshot.text)
        title_metadata_score = _title_coverage(
            expectation.title,
            snapshot.metadata,
        )
        if title_text_score >= 0.8:
            basis.append("title_text")
        if title_metadata_score >= 0.8:
            basis.append("title_metadata")

        if "doi_text" in basis or "doi_metadata" in basis:
            return 1.0, basis
        if "filename" in basis:
            return max(0.9, title_text_score, title_metadata_score), basis
        return max(title_text_score, title_metadata_score), basis

    @staticmethod
    def _missing_report(
        expectation: CorePaperExpectation,
        *,
        detail: str = "下载目录中尚未找到能够匹配该 DOI 或题名的 PDF。",
    ) -> PdfInspectionReport:
        return PdfInspectionReport(
            expectation=expectation,
            status="missing",
            issues=[
                PdfInspectionIssue(
                    code="file_missing",
                    severity="blocking",
                    detail=detail,
                )
            ],
        )


def _extract_dois(value: str) -> str:
    return " ".join(match.group(0).rstrip(".,;)") for match in DOI_PATTERN.finditer(value))


def _title_coverage(expected_title: str, candidate: str) -> float:
    expected_tokens = _meaningful_tokens(expected_title)
    if not expected_tokens:
        return 0
    candidate_tokens = set(_meaningful_tokens(candidate))
    return len(set(expected_tokens) & candidate_tokens) / len(set(expected_tokens))


def _meaningful_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", value.casefold())
        if len(token) >= 3 or any("\u4e00" <= char <= "\u9fff" for char in token)
    ]
