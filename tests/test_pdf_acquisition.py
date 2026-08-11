from pathlib import Path
from types import SimpleNamespace
import os

from pypdf import PdfWriter

from veriwrite_agent.models.evidence import CorePaperExpectation
from veriwrite_agent.services.pdf_acquisition import (
    PdfAcquisitionInspector,
    _PdfSnapshot,
    _extract_dois,
    evidence_document_identity_conflicts,
)
from veriwrite_agent.ui import evidence_console

DOI = "10.1000/core.1"
TITLE = "Atmospheric Remote Sensing with Multispectral Observations"


def test_project_pdf_directory_migrates_legacy_browser_downloads(monkeypatch) -> None:
    monkeypatch.setenv(
        evidence_console.EVIDENCE_VAULT_ENV,
        r"E:\AI-Agent-Projects\Evidence-Vault",
    )
    state = {
        evidence_console.PDF_DIRECTORY_KEY: str(Path.home() / "Downloads"),
    }

    assert evidence_console.project_pdf_directory(state) == (
        r"E:\AI-Agent-Projects\Evidence-Vault"
    )


def test_project_pdf_directory_preserves_an_explicit_custom_location(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        evidence_console.EVIDENCE_VAULT_ENV,
        r"E:\AI-Agent-Projects\Evidence-Vault",
    )
    state = {
        evidence_console.PDF_DIRECTORY_KEY: r"D:\Research\Current Paper",
    }

    assert evidence_console.project_pdf_directory(state) == (
        r"D:\Research\Current Paper"
    )


def expectation(
    *,
    doi: str = DOI,
    title: str = TITLE,
) -> CorePaperExpectation:
    return CorePaperExpectation(
        doi=doi,
        title=title,
        source_url=f"https://doi.org/{doi}",
        theme_id="remote_sensing",
    )


def write_pdf(path: Path, *, doi: str = DOI, title: str = TITLE) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata(
        {
            "/Title": title,
            "/Subject": f"Official article DOI: {doi}",
        }
    )
    with path.open("wb") as handle:
        writer.write(handle)


def test_inspects_a_valid_pdf_and_confirms_identity_from_metadata(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    write_pdf(pdf_path)

    report = PdfAcquisitionInspector().inspect_file(expectation(), pdf_path)

    assert report.status == "needs_review"
    assert report.identity_score == 1
    assert "doi_metadata" in report.identity_basis
    assert report.page_count == 1
    assert report.sha256
    assert report.issues[0].code == "text_not_extractable"


def test_scan_matches_one_download_and_keeps_other_paper_pending(
    tmp_path: Path,
) -> None:
    write_pdf(tmp_path / "downloaded.pdf")
    other = expectation(
        doi="10.1000/core.2",
        title="A Different Atmospheric Retrieval Paper",
    )

    batch = PdfAcquisitionInspector().scan_download_directory(
        [expectation(), other],
        tmp_path,
    )

    assert batch.inspected_file_count == 1
    assert batch.reports[0].status == "needs_review"
    assert batch.reports[1].status == "missing"
    assert batch.reports[1].issues[0].code == "file_missing"


def test_scan_does_not_assign_a_weak_partial_title_match(
    tmp_path: Path,
) -> None:
    write_pdf(
        tmp_path / "different.pdf",
        doi="10.1000/different",
        title="Atmospheric Remote Sensing for an Unrelated Experiment",
    )

    batch = PdfAcquisitionInspector().scan_download_directory(
        [expectation()],
        tmp_path,
    )

    assert batch.reports[0].status == "missing"
    assert batch.unmatched_files == [str(tmp_path / "different.pdf")]


def test_conflicting_doi_blocks_even_an_exact_title_match(tmp_path: Path) -> None:
    pdf_path = tmp_path / "wrong-paper.pdf"
    write_pdf(pdf_path, doi="10.1109/TGRS.2025.3593486", title=TITLE)

    report = PdfAcquisitionInspector().inspect_file(expectation(), pdf_path)

    assert report.status == "needs_review"
    assert report.identity_score == 0
    assert report.detected_dois == ["10.1109/tgrs.2025.3593486"]
    assert any(issue.code == "doi_conflict" for issue in report.issues)


def test_doi_scanner_repairs_pdf_glyph_whitespace() -> None:
    assert _extract_dois("doi:10.1016/j.atmosenv.2008.07 .018") == {
        "10.1016/j.atmosenv.2008.07.018"
    }


def test_scan_stops_after_the_newest_complete_match(
    tmp_path: Path,
    monkeypatch,
) -> None:
    older = tmp_path / "older-unrelated.pdf"
    newest = tmp_path / "newest-target.pdf"
    older.touch()
    newest.touch()
    os.utime(older, (1, 1))
    os.utime(newest, (2, 2))
    inspected: list[str] = []

    def snapshot(path: Path) -> _PdfSnapshot:
        inspected.append(path.name)
        return _PdfSnapshot(
            path=path,
            sha256="a" * 64,
            file_size_bytes=100,
            page_count=1,
            extractable_page_count=1,
            text=f"{TITLE}\nDOI: {DOI}",
            metadata="",
            issues=(),
        )

    inspector = PdfAcquisitionInspector()
    monkeypatch.setattr(inspector, "_read_snapshot", snapshot)

    batch = inspector.scan_download_directory([expectation()], tmp_path)

    assert inspected == ["newest-target.pdf"]
    assert batch.inspected_file_count == 1
    assert batch.reports[0].status == "verified"
    assert batch.unmatched_files == []


def test_delivery_time_identity_check_catches_a_stale_corrupted_library() -> None:
    library = SimpleNamespace(
        records=[
            SimpleNamespace(
                doi="10.1016/j.atmosenv.2008.07.018",
                evidence_status="full_text_verified",
            )
        ],
        pages=[
            SimpleNamespace(
                doi="10.1016/j.atmosenv.2008.07.018",
                page_number=1,
                text="Article DOI: 10.1109 /TGRS.2025.3593486",
            )
        ],
    )

    assert evidence_document_identity_conflicts(library) == {
        "10.1016/j.atmosenv.2008.07.018": ["10.1109/tgrs.2025.3593486"]
    }


def test_scan_identifies_an_html_download_as_an_invalid_pdf(
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "publisher-download.htm"
    html_path.write_text(
        (
            f'<meta property="og:title" content="{TITLE}">'
            f'<meta name="citation_doi" content="{DOI}">'
        ),
        encoding="utf-8",
    )

    batch = PdfAcquisitionInspector().scan_download_directory(
        [expectation()],
        tmp_path,
    )

    assert batch.reports[0].status == "invalid"
    assert batch.reports[0].local_path == str(html_path)
    assert batch.reports[0].identity_score == 1
    assert batch.reports[0].issues[0].code == "not_pdf"


def test_rejects_a_file_with_a_pdf_extension_but_wrong_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "10_1000_core_1.pdf"
    path.write_text("not a real PDF", encoding="utf-8")

    report = PdfAcquisitionInspector().inspect_file(expectation(), path)

    assert report.status == "invalid"
    assert any(issue.code == "not_pdf" for issue in report.issues)


def test_verified_report_converts_to_document_acquisition(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    write_pdf(pdf_path)
    inspector = PdfAcquisitionInspector()
    report = inspector.inspect_file(expectation(), pdf_path)
    verified = report.model_copy(
        update={
            "status": "verified",
            "issues": [],
        }
    )
    batch = inspector.scan_download_directory([], tmp_path).model_copy(
        update={"reports": [verified]}
    )

    acquisition = inspector.to_document_acquisitions(batch)[0]

    assert acquisition.status == "available"
    assert acquisition.method == "user_upload"
    assert acquisition.sha256 == report.sha256
