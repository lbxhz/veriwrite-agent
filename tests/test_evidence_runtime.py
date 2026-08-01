from pathlib import Path

from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    DocumentExtractionResult,
    DocumentPage,
)
from veriwrite_agent.services.evidence_runtime import (
    EvidencePageRetriever,
    EvidenceRuntimeCache,
)


DOI = "10.1000/runtime.1"
SHA = "a" * 64


def pages() -> list[DocumentPage]:
    return [
        DocumentPage(
            doi=DOI,
            document_sha256=SHA,
            page_number=1,
            text="Abstract and introduction to atmospheric observation.",
            extraction_method="native_text",
        ),
        DocumentPage(
            doi=DOI,
            document_sha256=SHA,
            page_number=2,
            text="Unrelated administrative material.",
            extraction_method="native_text",
        ),
        DocumentPage(
            doi=DOI,
            document_sha256=SHA,
            page_number=3,
            text="Methane satellite retrieval method and result discussion.",
            extraction_method="native_text",
        ),
    ]


def acquisition() -> DocumentAcquisition:
    return DocumentAcquisition(
        doi=DOI,
        status="available",
        method="user_upload",
        source_url=f"https://doi.org/{DOI}",
        local_path="paper.pdf",
        sha256=SHA,
        media_type="application/pdf",
        file_size_bytes=2048,
        attempts=1,
    )


def test_retrieval_is_auditable_and_extraction_cache_round_trips(tmp_path: Path) -> None:
    selection, selected_pages = EvidencePageRetriever(max_pages=2).select(
        doi=DOI,
        theme_id="methane",
        query_text="satellite methane retrieval",
        pages=pages(),
    )

    assert selection.selected_page_numbers == [1, 3]
    assert [page.page_number for page in selected_pages] == [1, 3]
    extraction = DocumentExtractionResult(
        doi=DOI,
        document_sha256=SHA,
        status="complete",
        page_count=3,
        pages=pages(),
    )
    cache = EvidenceRuntimeCache(tmp_path, policy_fingerprint="b" * 64)
    cache.save_extraction(extraction)

    assert cache.load_extraction(acquisition()) == extraction
