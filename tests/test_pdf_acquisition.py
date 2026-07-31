from pathlib import Path

from pypdf import PdfWriter

from veriwrite_agent.models.evidence import CorePaperExpectation
from veriwrite_agent.services.pdf_acquisition import PdfAcquisitionInspector

DOI = "10.1000/core.1"
TITLE = "Atmospheric Remote Sensing with Multispectral Observations"


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
