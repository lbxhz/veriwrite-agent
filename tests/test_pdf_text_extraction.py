import hashlib
from pathlib import Path

from veriwrite_agent.models.evidence import DocumentAcquisition
from veriwrite_agent.services.pdf_text_extraction import PdfPageExtractor

DOI = "10.1000/extract.1"


class FakePage:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class FakeReader:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages


def acquisition(path: Path) -> DocumentAcquisition:
    payload = path.read_bytes()
    return DocumentAcquisition(
        doi=DOI,
        status="available",
        method="user_upload",
        source_url=f"https://doi.org/{DOI}",
        local_path=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
        media_type="application/pdf",
        file_size_bytes=len(payload),
        attempts=1,
    )


def test_extracts_native_text_with_page_and_hash_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-test")
    monkeypatch.setattr(
        "veriwrite_agent.services.pdf_text_extraction.PdfReader",
        lambda *_args, **_kwargs: FakeReader(
            [
                FakePage("First page contains enough native article text."),
                FakePage("Second page contains enough native method text."),
            ]
        ),
    )

    result = PdfPageExtractor(enable_ocr=False).extract(acquisition(path))

    assert result.status == "complete"
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert all(page.extraction_method == "native_text" for page in result.pages)


def test_marks_a_textless_page_as_needing_ocr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-scan")
    monkeypatch.setattr(
        "veriwrite_agent.services.pdf_text_extraction.PdfReader",
        lambda *_args, **_kwargs: FakeReader([FakePage("")]),
    )

    result = PdfPageExtractor(enable_ocr=False).extract(acquisition(path))

    assert result.status == "needs_ocr"
    assert result.pages == []
    assert result.issues[0].code == "page_text_missing"


def test_rejects_a_pdf_changed_after_download_inspection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "changed.pdf"
    path.write_bytes(b"%PDF-original")
    document = acquisition(path)
    path.write_bytes(b"%PDF-modified")

    result = PdfPageExtractor(enable_ocr=False).extract(document)

    assert result.status == "failed"
    assert result.issues[0].code == "hash_mismatch"
