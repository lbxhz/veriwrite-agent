"""Human-in-the-loop PDF acquisition and deterministic file inspection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
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
        if not expectations:
            return PdfInspectionBatch(
                download_directory=str(download_directory),
                inspected_file_count=0,
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
        assignments: dict[str, tuple[_PdfSnapshot, float]] = {}
        used_paths: set[Path] = set()
        inspected_snapshots: list[_PdfSnapshot] = []
        fallback_pairs: list[
            tuple[float, int, CorePaperExpectation, _PdfSnapshot]
        ] = []

        # Downloads are already sorted newest first. Inspect incrementally and
        # stop as soon as every expected paper has a complete verified PDF.
        # Non-PDF/OCR/review candidates are retained as fallbacks, but do not
        # prevent the scanner from looking for a better copy further down.
        for recency_rank, path in enumerate(paths):
            snapshot = self._read_snapshot(path)
            inspected_snapshots.append(snapshot)
            scored = sorted(
                (
                    (self._identity_score(expectation, snapshot)[0], expectation)
                    for expectation in expectations
                    if expectation.doi not in assignments
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if not scored or scored[0][0] < 0.8:
                continue
            score, expectation = scored[0]
            report = self._report(expectation, snapshot)
            if report.status == "verified":
                assignments[expectation.doi] = (snapshot, score)
                used_paths.add(snapshot.path)
                if len(assignments) == len(expectations):
                    break
                continue
            fallback_pairs.append(
                (score, -recency_rank, expectation, snapshot)
            )

        # Preserve actionable invalid/OCR reports only when no verified copy
        # was found. Higher identity confidence wins, then the newer file.
        fallback_pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for score, _recency, expectation, snapshot in fallback_pairs:
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
            inspected_file_count=len(inspected_snapshots),
            reports=reports,
            unmatched_files=[
                str(snapshot.path)
                for snapshot in inspected_snapshots
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
            for page_index, page in enumerate(
                reader.pages[: self.text_page_limit]
            ):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                if page_text.strip():
                    extractable_page_count += 1
                    # DOI/title identity belongs on the article's first page. Searching
                    # several body/reference pages lets common title words or a cited DOI
                    # create a high-scoring false match to an unrelated PDF.
                    if page_index == 0:
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
        detected_dois = sorted(
            _extract_dois(snapshot.text) | _extract_dois(snapshot.metadata)
        )
        conflicting_dois = _conflicting_dois(
            expectation.doi,
            snapshot.text,
            snapshot.metadata,
        )
        if conflicting_dois:
            issues.append(
                PdfInspectionIssue(
                    code="doi_conflict",
                    severity="blocking",
                    detail=(
                        "PDF 首页或元数据中的 DOI 与目标文献不一致："
                        f"目标={expectation.doi}；检测到={', '.join(conflicting_dois)}。"
                        "不能用标题词相似度覆盖 DOI 身份冲突。"
                    ),
                )
            )
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
            detected_dois=detected_dois,
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

        if _conflicting_dois(expectation.doi, snapshot.text, snapshot.metadata):
            return 0.0, basis

        text_dois = _extract_dois(text)
        metadata_dois = _extract_dois(metadata)
        if expected_doi in _normalized_doi_text(text) or expected_doi in text_dois:
            basis.append("doi_text")
        if (
            expected_doi in _normalized_doi_text(metadata)
            or expected_doi in metadata_dois
        ):
            basis.append("doi_metadata")
        filename_doi = re.sub(r"[^a-z0-9]", "", expected_doi)
        if filename_doi and filename_doi in re.sub(r"[^a-z0-9]", "", filename):
            basis.append("filename")

        title_text_score = _title_similarity(expectation.title, snapshot.text)
        title_metadata_score = _title_similarity(
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


def _normalized_doi_text(value: str) -> str:
    """Normalize whitespace around DOI separators before identity checks."""

    normalized = value.casefold()
    normalized = re.sub(r"(10\.\d{4,9})\s*/\s*", r"\1/", normalized)
    # PDF glyph positioning can insert spaces inside a DOI, for example
    # ``10.1016/j.atmosenv.2008.07 .018``. Remove whitespace only when it
    # directly precedes DOI punctuation followed by an identifier character.
    return re.sub(r"(?<=[a-z0-9])\s+(?=[._;()/:][a-z0-9])", "", normalized)


def _extract_dois(value: str) -> set[str]:
    normalized = _normalized_doi_text(value)
    return {
        match.group(0).rstrip(".,;:)").casefold()
        for match in DOI_PATTERN.finditer(normalized)
    }


def _conflicting_dois(expected_doi: str, *values: str) -> list[str]:
    expected = expected_doi.casefold()
    detected = set().union(*(_extract_dois(value) for value in values))
    normalized_values = " ".join(_normalized_doi_text(value) for value in values)
    if expected in normalized_values:
        detected.add(expected)
    if detected and expected not in detected:
        return sorted(detected)
    return []


def evidence_document_identity_conflicts(library) -> dict[str, list[str]]:
    """Find full-text records whose first page declares a different DOI."""

    full_text_dois = {
        record.doi
        for record in getattr(library, "records", [])
        if getattr(record, "evidence_status", None) == "full_text_verified"
    }
    first_page_text: dict[str, list[str]] = {}
    for page in getattr(library, "pages", []):
        if page.page_number == 1 and page.doi in full_text_dois:
            first_page_text.setdefault(page.doi, []).append(page.text)
    conflicts: dict[str, list[str]] = {}
    for expected_doi, parts in first_page_text.items():
        detected = _conflicting_dois(expected_doi, "\n".join(parts))
        if detected:
            conflicts[expected_doi] = detected
    return conflicts


def _title_similarity(expected_title: str, candidate: str) -> float:
    """Measure an ordered, local title match instead of page-wide token overlap."""

    expected_tokens = _meaningful_tokens(expected_title)
    if not expected_tokens:
        return 0.0
    candidate_tokens = _meaningful_tokens(candidate)
    if not candidate_tokens:
        return 0.0
    expected_count = len(expected_tokens)
    minimum = max(1, expected_count - 2)
    maximum = min(len(candidate_tokens), expected_count + 3)
    best = 0.0
    for size in range(minimum, maximum + 1):
        for start in range(0, len(candidate_tokens) - size + 1):
            window = candidate_tokens[start : start + size]
            score = SequenceMatcher(None, expected_tokens, window).ratio()
            if score > best:
                best = score
    return best


def _meaningful_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", value.casefold())
        if len(token) >= 3 or any("\u4e00" <= char <= "\u9fff" for char in token)
    ]
