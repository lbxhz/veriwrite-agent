"""Pure, injectable assembly of the mixed-tier V0.3 evidence library and handoff.

These functions are shared by the V0.3 PDF-acquisition console and the V0.4
post-draft "download deferred PDFs -> rebuild -> merge -> enhance" chain.  They
take every input explicitly so they can be driven from either UI surface and
tested without ``streamlit`` session state.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.evidence import (
    EvidenceLibrary,
    LiteratureLibraryRecord,
    PdfInspectionBatch,
)
from veriwrite_agent.models.executable_policy import ExecutableRequirementPolicy
from veriwrite_agent.models.literature_selection import BalancedLiteratureSelection
from veriwrite_agent.models.literature_verification import LiteratureVerificationBatch
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.services.evidence_card_extraction import LLMEvidenceCardExtractor
from veriwrite_agent.services.evidence_library import (
    EvidenceLibraryBuilder,
    EvidenceLibraryConfirmationService,
)
from veriwrite_agent.services.evidence_runtime import (
    EvidencePageRetriever,
    EvidenceRuntimeCache,
)
from veriwrite_agent.services.pdf_acquisition import PdfAcquisitionInspector
from veriwrite_agent.services.pdf_text_extraction import PdfPageExtractor
from veriwrite_agent.services.writing_evidence_recovery import merge_recovery_handoffs
from veriwrite_agent.services.writing_handoff import (
    WritingHandoffService,
    WritingOutlineBuilder,
)


def build_evidence_library(
    selection: BalancedLiteratureSelection,
    batch: PdfInspectionBatch,
    *,
    confirmed_requirement: ConfirmedRequirementSpec,
    verifications: LiteratureVerificationBatch,
    policy: ExecutableRequirementPolicy,
    cache_root: Path,
    card_extractor: LLMEvidenceCardExtractor | None = None,
) -> EvidenceLibrary:
    """Build the mixed-tier evidence library from a scanned PDF batch.

    ``selection`` supplies the admitted records and their metadata; ``batch``
    supplies the identity/availability outcome of the PDF scan.  Every DOI that
    resolved to an available document becomes ``A_core``/``full_text_verified``,
    while the remaining core expectations stay ``B_supporting`` and everything
    else stays ``C_background``.
    """

    cache = EvidenceRuntimeCache(
        cache_root,
        policy_fingerprint=policy.requirement_fingerprint,
    )
    inspector = PdfAcquisitionInspector()
    documents = inspector.to_document_acquisitions(batch)
    available = {document.doi: document for document in documents if document.status == "available"}
    core_dois = {report.expectation.doi for report in batch.reports}
    verification_by_doi = {
        result.candidate.doi: result for result in verifications.verified_records
    }

    def authority_fields(doi: str) -> tuple[list[str], str | None, str | None, str]:
        verification = verification_by_doi.get(doi)
        if verification is None:
            return [], None, None, f"https://doi.org/{quote(doi, safe='/')}"
        metadata = verification.authority.metadata if verification.authority is not None else None
        return (
            metadata.authors if metadata is not None else verification.candidate.authors,
            (
                metadata.journal_title
                if metadata is not None
                else verification.candidate.journal_title
            ),
            verification.candidate.abstract,
            (
                verification.resolution.final_url
                if verification.resolution is not None and verification.resolution.final_url
                else f"https://doi.org/{quote(doi, safe='/')}"
            ),
        )

    authority_by_doi = {item.doi: authority_fields(item.doi) for item in selection.selected}
    records = [
        LiteratureLibraryRecord(
            doi=item.doi,
            title=item.title,
            authors=item.authors or authority_by_doi[item.doi][0],
            year=item.year,
            journal=item.journal or authority_by_doi[item.doi][1],
            publisher=item.publisher,
            language=item.language,
            source_type=item.source_type,
            is_foreign=item.is_foreign,
            abstract=authority_by_doi[item.doi][2],
            source_url=authority_by_doi[item.doi][3],
            theme_ids=[item.theme_id],
            admission_status="admitted",
            centrality=item.centrality,
            supported_claim=item.supported_claim,
            suitable_section_id=item.suitable_section_id,
            use_boundary=item.use_boundary,
            evidence_tier=(
                "A_core"
                if item.doi in available
                else ("B_supporting" if item.doi in core_dois else "C_background")
            ),
            evidence_status=(
                "full_text_verified" if item.doi in available else "metadata_verified"
            ),
            permitted_use=(
                "detailed_claims"
                if item.doi in available
                else ("section_support" if item.doi in core_dois else "background_only")
            ),
        )
        for item in selection.selected
    ]
    unresolved = [
        f"core_pdf_{report.status}:{report.expectation.doi}"
        for report in batch.reports
        if report.status != "verified"
    ]

    pages = []
    cards = []
    extractions = []
    page_selections = []
    active_extractor = card_extractor or LLMEvidenceCardExtractor(
        DeepSeekClient(LLMSettings().for_structured_output())
    )
    themes = {theme.theme_id: theme for theme in selection.blueprint.themes}
    record_by_doi = {record.doi: record for record in records}
    for document in available.values():
        extraction = cache.load_extraction(document)
        if extraction is None:
            extraction = PdfPageExtractor(enable_ocr=True).extract(document)
            cache.save_extraction(extraction)
        extractions.append(extraction)
        pages.extend(extraction.pages)
        if extraction.status != "complete":
            unresolved.append(f"pdf_extraction_{extraction.status}:{document.doi}")
        if not extraction.pages:
            continue
        record = record_by_doi[document.doi]
        theme_id = record.theme_ids[0]
        theme = themes[theme_id]
        query_text = " ".join(
            [
                record.title,
                theme.section_title,
                theme.section_purpose,
                *theme.research_questions,
                *theme.primary_keywords,
                *theme.related_keywords,
            ]
        )
        selection_audit, selected_pages = EvidencePageRetriever().select(
            doi=document.doi,
            theme_id=theme_id,
            query_text=query_text,
            pages=extraction.pages,
        )
        page_selections.append(selection_audit)
        try:
            document_cards = cache.load_cards(
                document,
                title=record.title,
                selection=selection_audit,
            )
            if document_cards is None:
                document_cards = active_extractor.extract(
                    doi=document.doi,
                    title=record.title,
                    theme_id=theme_id,
                    section_purpose=theme.section_purpose,
                    pages=selected_pages,
                )
                cache.save_cards(
                    document,
                    title=record.title,
                    selection=selection_audit,
                    cards=document_cards,
                )
            cards.extend(document_cards)
        except Exception as exc:
            unresolved.append(f"evidence_extraction_failed:{document.doi}:{exc}")

    return EvidenceLibraryBuilder().build(
        records=records,
        documents=documents,
        extractions=extractions,
        page_selections=page_selections,
        pages=pages,
        evidence_cards=cards,
        unresolved_issues=unresolved,
        requirement_policy_fingerprint=policy.requirement_fingerprint,
    )


def build_deferred_enhancement_handoff(
    previous_handoff: V04WritingHandoff,
    *,
    selection: BalancedLiteratureSelection,
    batch: PdfInspectionBatch,
    affected_section_ids: set[str],
    confirmed_requirement: ConfirmedRequirementSpec,
    verifications: LiteratureVerificationBatch,
    policy: ExecutableRequirementPolicy,
    cache_root: Path,
    smoke_test: bool = False,
    card_extractor: LLMEvidenceCardExtractor | None = None,
) -> V04WritingHandoff:
    """Rebuild the evidence library after deferred PDFs arrive and merge it in.

    The recovered library must be fully resolved (no unresolved issues) before it
    is confirmed and merged; otherwise the existing confirmed handoff is left
    untouched and the caller reports the still-missing DOIs.
    """

    library = build_evidence_library(
        selection,
        batch,
        confirmed_requirement=confirmed_requirement,
        verifications=verifications,
        policy=policy,
        cache_root=cache_root,
        card_extractor=card_extractor,
    )
    if library.unresolved_issues:
        raise ValueError(
            "仍有全文未就绪，无法合并增强："
            + "；".join(library.unresolved_issues)
        )
    confirmed_library = EvidenceLibraryConfirmationService().confirm(
        library,
        confirmed_by=confirmed_requirement.confirmed_by,
    )
    outline = WritingOutlineBuilder().build(
        selection.blueprint,
        confirmed_library,
        policy=policy,
        smoke_test=smoke_test,
    )
    confirmed_outline = WritingHandoffService().confirm_outline(
        outline,
        confirmed_by=confirmed_requirement.confirmed_by,
    )
    current = WritingHandoffService().create(
        requirement=confirmed_requirement,
        outline=confirmed_outline,
        evidence_library=confirmed_library,
        policy=policy,
    )
    return merge_recovery_handoffs(
        previous_handoff,
        current,
        affected_section_ids=affected_section_ids,
    )
