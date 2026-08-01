import hashlib
import json
from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject

from veriwrite_agent.literature.cug_catalog import CugJournalRankingProvider
from veriwrite_agent.literature.fake import (
    FakeAuthoritativeMetadataProvider,
    FakeDoiResolver,
    FakeLiteratureSearchProvider,
)
from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    LiteratureLibraryRecord,
)
from veriwrite_agent.models.literature_discovery import LiteratureCandidate
from veriwrite_agent.models.literature_selection import (
    LiteratureRelevanceAssessmentBatch,
)
from veriwrite_agent.models.literature_verification import (
    AuthoritativeMetadataEvidence,
    DoiResolutionEvidence,
    RisBibliographicMetadata,
)
from veriwrite_agent.models.requirement_workflow import RequirementConfirmation
from veriwrite_agent.models.writing import (
    DraftParagraphProposal,
    SectionDraftProposal,
)
from veriwrite_agent.models.final_delivery import FinalMatterProposal
from veriwrite_agent.services.evidence_card_extraction import (
    LLMEvidenceCardExtractor,
)
from veriwrite_agent.services.evidence_library import (
    EvidenceLibraryBuilder,
    EvidenceLibraryConfirmationService,
)
from veriwrite_agent.services.evidence_runtime import EvidencePageRetriever
from veriwrite_agent.services.final_delivery import (
    FinalPaperAssembler,
    FinalPaperDocxExporter,
)
from veriwrite_agent.services.grounded_writing import (
    GroundedSectionDraftService,
    SectionEvidencePacketBuilder,
    WritingProjectService,
)
from veriwrite_agent.services.literature_blueprint_confirmation import (
    LiteratureBlueprintConfirmationService,
)
from veriwrite_agent.services.literature_blueprint_planner import (
    LiteratureBlueprintPlanner,
)
from veriwrite_agent.services.literature_blueprint_search import (
    LiteratureBlueprintSearchExpander,
)
from veriwrite_agent.services.literature_discovery import LiteratureDiscoveryService
from veriwrite_agent.services.literature_identity_verification import (
    LiteratureIdentityVerificationService,
)
from veriwrite_agent.services.literature_relevance_scorer import (
    LLMLiteratureRelevanceScorer,
)
from veriwrite_agent.services.llm_requirement_parser import LLMRequirementParser
from veriwrite_agent.services.pdf_text_extraction import PdfPageExtractor
from veriwrite_agent.services.requirement_confirmation import (
    RequirementConfirmationService,
)
from veriwrite_agent.services.requirement_parser import RuleBasedRequirementParser
from veriwrite_agent.services.requirement_pipeline import RequirementReviewPipeline
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.services.writing_handoff import (
    WritingHandoffService,
    WritingOutlineBuilder,
)
from veriwrite_agent.ui.literature_workbench import LiteratureWorkbench


DOIS = ["10.1000/gold.1", "10.1000/gold.2"]


def test_realistic_gold_path_reaches_confirmed_markdown_and_docx(tmp_path: Path) -> None:
    requirement_text = (
        Path(__file__).parent / "fixtures" / "mvp_golden_requirement.txt"
    ).read_text(encoding="utf-8")
    rule_parser = RuleBasedRequirementParser()
    rule_spec = rule_parser.parse(requirement_text)
    review = RequirementReviewPipeline(
        rule_parser,
        llm_parser=LLMRequirementParser(FakeLLMClient(rule_spec.model_dump_json())),
    ).prepare(requirement_text)
    confirmed = RequirementConfirmationService().confirm(
        review,
        RequirementConfirmation(
            confirmed_by="gold-tester",
            field_updates={
                "output_language": "English",
                "references.recent_year_rule_strength": "hard",
                "references.in_text_style": "author_year",
                "structure.required_or_recommended_sections": [
                    "Abstract",
                    "Keywords",
                    "Research background",
                    "Method comparison",
                    "Conclusion",
                    "References",
                ],
                "formatting.paper_size": "A4",
                "formatting.body_font": "Times New Roman",
                "formatting.body_font_size": "11 pt",
                "formatting.line_spacing": 1.5,
            },
            acknowledged_issue_ids=[issue.issue_id for issue in review.completeness.issues],
        ),
    )
    policy = RequirementPolicyCompiler(current_year=2026).compile(confirmed)
    assert policy.length.minimum_units == 600
    assert policy.references.minimum_total == 2

    blueprint_json = json.dumps(
        {
            "topic": "ignored LLM topic",
            "discipline": "测绘科学与技术",
            "writing_through_line": "From background to method comparison.",
            "target_total": 2,
            "themes": [
                {
                    "theme_id": "background",
                    "section_title": "Research background",
                    "section_purpose": "Explain the recent research background.",
                    "research_questions": ["Why are retrieval methods needed?"],
                    "primary_keywords": ["atmospheric remote sensing"],
                    "search_queries": ["atmospheric remote sensing retrieval"],
                    "target_count": 1,
                },
                {
                    "theme_id": "methods",
                    "section_title": "Method comparison",
                    "section_purpose": "Compare recent retrieval methods.",
                    "research_questions": ["How do recent methods differ?"],
                    "primary_keywords": ["satellite retrieval method"],
                    "search_queries": ["satellite atmospheric retrieval method"],
                    "target_count": 1,
                },
            ],
        },
        ensure_ascii=False,
    )
    ranking = CugJournalRankingProvider.from_default_catalog()
    blueprint = LiteratureBlueprintPlanner(
        FakeLLMClient(blueprint_json),
        ranking.available_disciplines,
        current_year=2026,
    ).plan(confirmed)
    confirmed_blueprint = LiteratureBlueprintConfirmationService().confirm(
        blueprint,
        confirmed_by="gold-tester",
        expected_policy=blueprint.requirement_policy,
    )

    candidates = [
        _candidate(DOIS[0], "Recent atmospheric retrieval background", 2025),
        _candidate(DOIS[1], "Comparison of satellite retrieval methods", 2024),
    ]
    relevance = LiteratureRelevanceAssessmentBatch.model_validate(
        {
            "assessments": [
                _assessment(DOIS[0], 0.95, 0.40, "background"),
                _assessment(DOIS[1], 0.45, 0.96, "methods"),
            ]
        }
    )
    workbench = LiteratureWorkbench(
        planner=None,
        search_expander=LiteratureBlueprintSearchExpander(pool_multiplier=2),
        discovery_service=LiteratureDiscoveryService(
            FakeLiteratureSearchProvider(candidates),
            ranking,
        ),
        verification_service=_verification_service(candidates),
        relevance_scorer=LLMLiteratureRelevanceScorer(FakeLLMClient(relevance.model_dump_json())),
    )
    literature = workbench.run(
        confirmed_blueprint,
        cache_root=tmp_path / "literature",
    )
    assert literature.selection.target_reached is True
    assert len(literature.selection.selected) == 2

    records = []
    documents = []
    extractions = []
    page_selections = []
    pages = []
    cards = []
    themes = {theme.theme_id: theme for theme in blueprint.themes}
    for selected in literature.selection.selected:
        pdf_path = tmp_path / f"{selected.doi.rsplit('/', 1)[-1]}.pdf"
        _write_text_pdf(
            pdf_path,
            (
                "The full text reports atmospheric remote sensing retrieval methods. "
                "The verified results support a reproducible comparison and identify "
                "method limitations in recent satellite observations."
            ),
        )
        payload = pdf_path.read_bytes()
        document = DocumentAcquisition(
            doi=selected.doi,
            status="available",
            method="user_upload",
            source_url=f"https://doi.org/{selected.doi}",
            local_path=str(pdf_path),
            sha256=hashlib.sha256(payload).hexdigest(),
            media_type="application/pdf",
            file_size_bytes=len(payload),
            attempts=1,
        )
        extraction = PdfPageExtractor(enable_ocr=False).extract(document)
        assert extraction.status == "complete"
        theme = themes[selected.theme_id]
        page_selection, selected_pages = EvidencePageRetriever(max_pages=2).select(
            doi=selected.doi,
            theme_id=selected.theme_id,
            query_text=f"{selected.title} {theme.section_purpose}",
            pages=extraction.pages,
        )
        paper_cards = LLMEvidenceCardExtractor(
            FakeLLMClient(
                json.dumps(
                    {
                        "selections": [
                            {
                                "evidence_type": "result",
                                "normalized_claim": (
                                    "The verified paper supports comparison of retrieval methods."
                                ),
                                "passage_ids": ["page_1_passage_1"],
                                "support_strength": "direct",
                            }
                        ]
                    }
                )
            ),
            page_batch_size=1,
        ).extract(
            doi=selected.doi,
            title=selected.title,
            theme_id=selected.theme_id,
            section_purpose=theme.section_purpose,
            pages=selected_pages,
        )
        records.append(
            LiteratureLibraryRecord(
                doi=selected.doi,
                title=selected.title,
                authors=selected.authors,
                year=selected.year,
                journal=selected.journal,
                publisher=selected.publisher,
                language=selected.language,
                source_type=selected.source_type,
                is_foreign=selected.is_foreign,
                source_url=f"https://doi.org/{selected.doi}",
                theme_ids=[selected.theme_id],
                evidence_tier="A_core",
                evidence_status="full_text_verified",
                permitted_use="detailed_claims",
            )
        )
        documents.append(document)
        extractions.append(extraction)
        page_selections.append(page_selection)
        pages.extend(extraction.pages)
        cards.extend(paper_cards)

    library = EvidenceLibraryConfirmationService().confirm(
        EvidenceLibraryBuilder().build(
            records=records,
            documents=documents,
            extractions=extractions,
            page_selections=page_selections,
            pages=pages,
            evidence_cards=cards,
            requirement_policy_fingerprint=policy.requirement_fingerprint,
        ),
        confirmed_by="gold-tester",
    )
    assert len(library.pages) == 2
    assert all(item.status == "complete" for item in library.extractions)

    outline = WritingOutlineBuilder().build(blueprint, library, policy=policy)
    handoff_service = WritingHandoffService()
    handoff = handoff_service.create(
        requirement=confirmed,
        outline=handoff_service.confirm_outline(
            outline,
            confirmed_by="gold-tester",
        ),
        evidence_library=library,
    )
    writing = WritingProjectService()
    project = writing.start(handoff)
    paragraph_text = (
        "The verified full text evidence establishes a reproducible comparison of "
        "atmospheric retrieval methods and reported outcomes. " * 20
    ).strip()
    for section in outline.sections:
        packet = SectionEvidencePacketBuilder().build(handoff, section.section_id)
        draft = GroundedSectionDraftService().create(
            packet,
            SectionDraftProposal(
                section_id=section.section_id,
                paragraphs=[
                    DraftParagraphProposal(
                        role="detailed_evidence",
                        text=paragraph_text,
                        evidence_card_ids=[section.evidence_card_ids[0]],
                    )
                ],
            ),
        )
        project = writing.save_draft(project, draft)
        project = writing.confirm_section(
            project,
            section.section_id,
            confirmed_by="gold-tester",
        )
    body = writing.assemble_body(project)

    final_matter = FinalMatterProposal(
        title="Recent Atmospheric Remote Sensing Retrieval Methods",
        abstract=(
            "This review synthesizes the confirmed body evidence on recent atmospheric "
            "remote sensing retrieval methods, their reported outcomes, reproducibility, "
            "and limitations across satellite observation settings."
        ),
        keywords=["remote sensing", "atmosphere", "retrieval methods"],
        conclusion=(
            "The confirmed evidence shows that recent retrieval research emphasizes "
            "reproducible comparison, transparent reporting of outcomes, and explicit "
            "discussion of methodological limitations without introducing new sources."
        ),
    )
    assembler = FinalPaperAssembler()
    package = assembler.assemble(
        handoff=handoff,
        body=body,
        final_matter=final_matter,
    )
    assert package.status == "ready_for_confirmation"
    assert package.audit.blocking_count == 0
    assert package.audit.reference_count == 2
    assert package.audit.foreign_reference_count == 2
    assert package.audit.deferred_checks == ["claim_entailment"]
    confirmed_package = assembler.confirm(package, confirmed_by="gold-tester")
    docx_bytes = FinalPaperDocxExporter().export(confirmed_package)

    assert confirmed_package.markdown.startswith(
        "# Recent Atmospheric Remote Sensing Retrieval Methods"
    )
    assert "## References" in confirmed_package.markdown
    assert docx_bytes.startswith(b"PK")
    rendered_doc = Document(BytesIO(docx_bytes))
    assert rendered_doc.paragraphs[0].text == confirmed_package.title


def _candidate(doi: str, title: str, year: int) -> LiteratureCandidate:
    return LiteratureCandidate(
        doi=doi,
        title=title,
        authors=["Taylor Smith"],
        year=year,
        journal_title="Remote Sensing of Environment",
        issns=["0034-4257"],
        publisher="Elsevier BV",
        language="en",
        source_provider="crossref",
        source_url=f"https://doi.org/{doi}",
    )


def _assessment(
    doi: str,
    background_score: float,
    methods_score: float,
    best_theme_id: str,
) -> dict[str, object]:
    return {
        "doi": doi,
        "theme_scores": [
            {
                "theme_id": "background",
                "score": background_score,
                "rationale": "gold semantic fit",
            },
            {
                "theme_id": "methods",
                "score": methods_score,
                "rationale": "gold semantic fit",
            },
        ],
        "best_theme_id": best_theme_id,
    }


def _verification_service(
    candidates: list[LiteratureCandidate],
) -> LiteratureIdentityVerificationService:
    resolutions = {}
    authorities = {}
    for candidate in candidates:
        resolutions[candidate.doi] = DoiResolutionEvidence(
            doi=candidate.doi,
            status="resolved",
            resolver_url=f"https://doi.org/{candidate.doi}",
            final_url=f"https://publisher.example/{candidate.doi}",
            http_status=200,
            attempts=1,
            reason="resolved",
        )
        authorities[candidate.doi] = AuthoritativeMetadataEvidence(
            doi=candidate.doi,
            status="available",
            source_url=f"https://doi.org/{candidate.doi}",
            metadata=RisBibliographicMetadata(
                doi=candidate.doi,
                title=candidate.title,
                authors=candidate.authors,
                year=candidate.year,
                journal_title=candidate.journal_title,
                publisher=candidate.publisher,
                ris_type="JOUR",
            ),
            raw_ris=(f"TY  - JOUR\nDO  - {candidate.doi}\nTI  - {candidate.title}\nER  -"),
            attempts=1,
            reason="available",
        )
    return LiteratureIdentityVerificationService(
        FakeDoiResolver(resolutions),
        FakeAuthoritativeMetadataProvider(authorities),
    )


def _write_text_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
    )
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = StreamObject()
    content.set_data(f"BT /F1 11 Tf 54 720 Td ({safe_text}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(content)
    with path.open("wb") as output:
        writer.write(output)
