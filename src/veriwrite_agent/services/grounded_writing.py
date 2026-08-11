"""V0.4 evidence packets, constrained LLM writing, and deterministic citations."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.evidence import EvidenceCard, LiteratureLibraryRecord
from veriwrite_agent.models.writing import (
    BodyDraftPackage,
    CitationBinding,
    DraftParagraphContent,
    DraftParagraphProposal,
    ParagraphSupportBinding,
    SectionDraft,
    SectionDraftIssue,
    SectionDraftProposal,
    SectionEvidenceItem,
    SectionEvidencePacket,
    SectionSourceRecord,
    SectionSupportBindingBatch,
    UnboundSectionDraftProposal,
    V04WritingProject,
    WritingSectionState,
)
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.services.requirement_policy import (
    RequirementPolicyCompiler,
    ai_generation_prohibitions,
)
from veriwrite_agent.services.topic_admission import audit_topic_admission
from veriwrite_agent.services.writing_quality import (
    language_mismatch_detail,
    repeated_sentence_pairs,
    workflow_instruction_leak_detail,
)


class GroundedWritingError(ValueError):
    """Raised when a writing step violates a confirmed contract."""


class SectionEvidencePacketBuilder:
    """Resolve one outline section to its permitted V0.3 evidence context."""

    def build(
        self,
        handoff: V04WritingHandoff,
        section_id: str,
        *,
        include_policy_required_routes: bool = True,
    ) -> SectionEvidencePacket:
        section = next(
            (
                item
                for item in handoff.outline.outline.sections
                if item.section_id == section_id
            ),
            None,
        )
        if section is None:
            raise GroundedWritingError(
                f"section_id is not in the confirmed outline: {section_id}"
            )

        library = handoff.evidence_library
        policy = handoff.requirement_policy or RequirementPolicyCompiler().compile(
            handoff.requirement
        )
        records = {record.doi: record for record in library.records}
        cards = {card.evidence_id: card for card in library.evidence_cards}
        evidence_items: list[SectionEvidenceItem] = []
        evidence_dois: list[str] = []
        for evidence_id in section.evidence_card_ids:
            card = cards.get(evidence_id)
            if card is None:
                raise GroundedWritingError(
                    f"confirmed outline references unknown evidence: {evidence_id}"
                )
            if card.review_status != "confirmed":
                raise GroundedWritingError(
                    f"section evidence is not confirmed: {evidence_id}"
                )
            evidence_items.append(_evidence_item(card))
            if card.doi not in evidence_dois:
                evidence_dois.append(card.doi)

        source_dois = list(
            dict.fromkeys(
                [
                    *section.core_dois,
                    *section.supporting_dois,
                    *evidence_dois,
                    *(
                        record.doi
                        for record in library.records
                        if (
                            include_policy_required_routes
                            and policy.references.all_bibliography_items_must_be_cited_and_discussed
                            and record.admission_status == "admitted"
                            and record.suitable_section_id == section.section_id
                        )
                    ),
                ]
            )
        )
        missing_records = [doi for doi in source_dois if doi not in records]
        if missing_records:
            raise GroundedWritingError(
                "section references DOI values missing from the evidence library: "
                + ", ".join(missing_records)
            )
        sources = [_source_record(records[doi]) for doi in source_dois]
        if not evidence_items:
            raise GroundedWritingError(
                "section has no confirmed full-text evidence cards"
            )
        admission = audit_topic_admission(
            library,
            policy,
            valid_section_ids=(
                item.section_id for item in handoff.outline.outline.sections
            ),
        )
        if not admission.passed:
            raise GroundedWritingError(
                "section writing is blocked until literature topic admission is "
                f"revalidated: {admission.detail}"
            )
        policy_reasons = _ai_generation_prohibitions(handoff)
        return SectionEvidencePacket(
            section_id=section.section_id,
            title=section.title,
            purpose=section.purpose,
            target_words=section.target_words,
            counting_policy=section.counting_policy,
            output_language=policy.output_language,
            research_questions=section.research_questions,
            max_sources_per_paragraph=(
                policy.references.max_references_per_citation_cluster or 3
            ),
            evidence_items=evidence_items,
            sources=sources,
            ai_writing_mode=(
                "generation_blocked"
                if policy_reasons
                else "generation_allowed"
            ),
            ai_policy_reasons=policy_reasons,
        )


class LLMGroundedSectionWriter:
    """Let an LLM organize prose while code retains citation authority."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def draft(self, packet: SectionEvidencePacket) -> SectionDraft:
        if packet.ai_writing_mode == "generation_blocked":
            raise GroundedWritingError(
                "AI prose generation is prohibited by the confirmed requirement: "
                + "; ".join(packet.ai_policy_reasons)
            )
        schema = json.dumps(
            SectionDraftProposal.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        raw = self._client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You write one scholarly body section using only the supplied "
                        "evidence packet. Return JSON only. Do not invent papers, DOI "
                        "values, results, numbers, methods, or citations. Do not write "
                        "citation markers in paragraph text; the application adds them. "
                        "Every paragraph must declare the exact evidence_card_ids and/or "
                        "source_dois it uses. detailed_evidence paragraphs require "
                        "at least one full-text evidence card; a DOI alone is not enough. "
                        "Never leave both support arrays empty, including for synthesis. "
                        "Use detailed_evidence only when the paragraph is grounded in "
                        "the attached evidence cards. Do not state specific numbers, "
                        "results, or methods from metadata-only sources. "
                        "Metadata-only B sources may support general section claims; "
                        "C sources are background only. Respect every source's "
                        "supported_claim, suitable_section_id, and use_boundary; a "
                        "supporting source must not become the paragraph's main subject. Group "
                        "literature by problem, method, trend, limitation, or comparison "
                        "instead of listing one paper per paragraph. Use a restrained "
                        "academic style and stay close to the target word count. "
                        f"The response must satisfy this JSON Schema: {schema}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        packet.model_dump(mode="json"),
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        try:
            proposal = SectionDraftProposal.model_validate_json(raw)
        except ValidationError as exc:
            if not _only_missing_support_errors(exc):
                raise GroundedWritingError(
                    "LLM section output violates the V0.4 data contract: "
                    f"{_validation_error_summary(exc)}"
                ) from exc
            proposal = self._repair_support_bindings(packet, raw, exc)
        if proposal.section_id != packet.section_id:
            raise GroundedWritingError(
                "LLM changed the confirmed section_id"
            )
        return GroundedSectionDraftService().create(packet, proposal)

    def _repair_support_bindings(
        self,
        packet: SectionEvidencePacket,
        raw: str,
        validation_error: ValidationError,
    ) -> SectionDraftProposal:
        try:
            unbound = UnboundSectionDraftProposal.model_validate_json(raw)
        except ValidationError as exc:
            raise GroundedWritingError(
                "LLM prose could not be preserved for support repair: "
                f"{_validation_error_summary(exc)}"
            ) from exc
        if unbound.section_id != packet.section_id:
            raise GroundedWritingError("LLM changed the confirmed section_id")

        repair_numbers = _support_error_paragraph_numbers(validation_error)

        schema = json.dumps(
            SectionSupportBindingBatch.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        repair_payload = {
            "section_id": unbound.section_id,
            "paragraphs": [
                {
                    "paragraph_number": number,
                    "role": paragraph.role,
                    "text": paragraph.text,
                    "current_evidence_card_ids": paragraph.evidence_card_ids,
                    "current_source_dois": paragraph.source_dois,
                }
                for number, paragraph in enumerate(unbound.paragraphs, 1)
                if number in repair_numbers
            ],
            "allowed_evidence_items": [
                item.model_dump(mode="json") for item in packet.evidence_items
            ],
            "allowed_sources": [
                source.model_dump(mode="json") for source in packet.sources
            ],
        }
        repaired_raw = self._client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You repair evidence bindings for an immutable scholarly draft. "
                        "Return JSON only and do not return or rewrite paragraph text. "
                        "Return exactly one binding for every supplied paragraph_number. "
                        "Existing valid paragraph bindings are code-owned and immutable. Use only "
                        "evidence_card_ids and source_dois from the supplied allowlists. "
                        "Every paragraph needs at least one declared support identifier. "
                        "A detailed_evidence paragraph must include at least one evidence "
                        "card whose normalized claim or exact quote supports that paragraph. "
                        "General synthesis may use source_dois, but must still declare its "
                        "sources. Do not guess when the packet lacks support. "
                        f"The response must satisfy this JSON Schema: {schema}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(repair_payload, ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
        )
        try:
            repaired = SectionSupportBindingBatch.model_validate_json(repaired_raw)
        except ValidationError as exc:
            raise GroundedWritingError(
                "LLM support repair still violates the V0.4 data contract: "
                f"{_validation_error_summary(exc)}"
            ) from exc
        if repaired.section_id != packet.section_id:
            raise GroundedWritingError("LLM support repair changed the confirmed section_id")

        bindings = {binding.paragraph_number: binding for binding in repaired.bindings}
        if set(bindings) != repair_numbers:
            raise GroundedWritingError(
                "LLM support repair did not bind every paragraph exactly once"
            )

        try:
            paragraphs = [
                (
                    _bind_paragraph_support(paragraph, bindings[number])
                    if number in bindings
                    else DraftParagraphProposal.model_validate(
                        paragraph.model_dump(mode="python")
                    )
                )
                for number, paragraph in enumerate(unbound.paragraphs, 1)
            ]
            return SectionDraftProposal(
                section_id=unbound.section_id,
                paragraphs=paragraphs,
            )
        except ValidationError as exc:
            raise GroundedWritingError(
                "LLM support repair left invalid paragraph bindings: "
                f"{_validation_error_summary(exc)}"
            ) from exc


def _only_missing_support_errors(exc: ValidationError) -> bool:
    allowed_messages = {
        "Value error, every paragraph requires declared source support",
        "Value error, detailed_evidence paragraphs require evidence cards",
    }
    errors = exc.errors(include_url=False)
    return bool(errors) and all(
        error.get("msg") in allowed_messages
        and tuple(error.get("loc", ()))[:1] == ("paragraphs",)
        for error in errors
    )


def _validation_error_summary(exc: ValidationError) -> list[dict[str, object]]:
    """Keep UI errors useful without echoing entire generated paragraphs."""

    return [
        {
            "location": ".".join(str(item) for item in error.get("loc", ())),
            "message": error.get("msg", "validation failed"),
        }
        for error in exc.errors(include_url=False)[:8]
    ]


def _support_error_paragraph_numbers(exc: ValidationError) -> set[int]:
    return {
        int(error["loc"][1]) + 1
        for error in exc.errors(include_url=False)
        if len(error.get("loc", ())) >= 2
        and error["loc"][0] == "paragraphs"
        and isinstance(error["loc"][1], int)
    }


def _bind_paragraph_support(
    paragraph: DraftParagraphContent,
    binding: ParagraphSupportBinding,
) -> DraftParagraphProposal:
    return DraftParagraphProposal(
        role=paragraph.role,
        text=paragraph.text,
        evidence_card_ids=binding.evidence_card_ids,
        source_dois=binding.source_dois,
    )


class GroundedSectionDraftService:
    """Audit support declarations and render citations without LLM authority."""

    def create(
        self,
        packet: SectionEvidencePacket,
        proposal: SectionDraftProposal,
    ) -> SectionDraft:
        if proposal.section_id != packet.section_id:
            raise GroundedWritingError("proposal section_id does not match packet")
        evidence = {item.evidence_id: item for item in packet.evidence_items}
        sources = {source.doi: source for source in packet.sources}
        issues: list[SectionDraftIssue] = []
        citations: list[CitationBinding] = []
        rendered_paragraphs: list[str] = []

        for paragraph_number, paragraph in enumerate(proposal.paragraphs, 1):
            paragraph_issues, paragraph_citations = _audit_paragraph(
                packet.section_id,
                paragraph_number,
                paragraph,
                evidence,
                sources,
                packet.output_language,
            )
            issues.extend(paragraph_issues)
            citations.extend(paragraph_citations)
            rendered_paragraphs.append(
                _render_paragraph(paragraph.text, paragraph_citations)
            )

        for first, second in repeated_sentence_pairs(
            [paragraph.text for paragraph in proposal.paragraphs]
        ):
            issues.append(
                SectionDraftIssue(
                    code="paragraph_repetition",
                    severity="warning",
                    paragraph_number=second,
                    detail=(
                        f"paragraph {second} substantially repeats paragraph {first}; "
                        "merge or advance the argument before confirmation"
                    ),
                )
            )

        markdown = f"## {packet.title}\n\n" + "\n\n".join(
            rendered_paragraphs
        )
        counted_words = sum(
            count_writing_units(
                paragraph.text,
                counting_policy=packet.counting_policy,
            )
            for paragraph in proposal.paragraphs
        )
        if counted_words < int(packet.target_words * 0.6):
            issues.append(
                SectionDraftIssue(
                    code="word_count_low",
                    severity="warning",
                    detail=(
                        f"section has {counted_words} counted units; "
                        f"target is {packet.target_words}"
                    ),
                )
            )
        if counted_words > int(packet.target_words * 1.4):
            issues.append(
                SectionDraftIssue(
                    code="word_count_high",
                    severity="warning",
                    detail=(
                        f"section has {counted_words} counted units; "
                        f"target is {packet.target_words}"
                    ),
                )
            )
        blocking = any(issue.severity == "blocking" for issue in issues)
        return SectionDraft(
            section_id=packet.section_id,
            title=packet.title,
            status="needs_review" if blocking else "draft",
            target_words=packet.target_words,
            counted_words=counted_words,
            paragraphs=proposal.paragraphs,
            markdown=markdown,
            citations=citations,
            issues=issues,
        )

    def confirm(
        self,
        draft: SectionDraft,
        *,
        confirmed_by: str,
    ) -> SectionDraft:
        if draft.status == "needs_review":
            raise GroundedWritingError(
                "section has blocking citation or evidence issues"
            )
        if draft.quality_review_status != "passed":
            raise GroundedWritingError(
                "section cannot be confirmed until the independent quality review passes; "
                f"current status={draft.quality_review_status}"
            )
        name = confirmed_by.strip()
        if not name:
            raise GroundedWritingError("confirmed_by cannot be blank")
        return draft.model_copy(
            update={
                "status": "confirmed",
                "confirmed_by": name,
                "confirmed_at": datetime.now(timezone.utc),
            }
        )


class WritingProjectService:
    """Persist section checkpoints and assemble only confirmed body text."""

    def start(self, handoff: V04WritingHandoff) -> V04WritingProject:
        return V04WritingProject(
            handoff=handoff,
            sections=[
                WritingSectionState(section_id=section.section_id)
                for section in handoff.outline.outline.sections
            ],
        )

    def save_draft(
        self,
        project: V04WritingProject,
        draft: SectionDraft,
    ) -> V04WritingProject:
        if draft.section_id not in {
            state.section_id for state in project.sections
        }:
            raise GroundedWritingError(
                "draft section is not in the confirmed outline"
            )
        states = [
            (
                WritingSectionState(
                    section_id=state.section_id,
                    status=draft.status,
                    draft=draft,
                )
                if state.section_id == draft.section_id
                else state
            )
            for state in project.sections
        ]
        return project.model_copy(
            update={
                "status": (
                    "body_complete"
                    if all(state.status == "confirmed" for state in states)
                    else "drafting"
                ),
                "sections": states,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def confirm_section(
        self,
        project: V04WritingProject,
        section_id: str,
        *,
        confirmed_by: str,
    ) -> V04WritingProject:
        state = next(
            (
                item
                for item in project.sections
                if item.section_id == section_id
            ),
            None,
        )
        if state is None or state.draft is None:
            raise GroundedWritingError("section has no draft to confirm")
        confirmed = GroundedSectionDraftService().confirm(
            state.draft,
            confirmed_by=confirmed_by,
        )
        return self.save_draft(project, confirmed)

    def assemble_body(self, project: V04WritingProject) -> BodyDraftPackage:
        if project.status != "body_complete":
            raise GroundedWritingError(
                "all body sections must be confirmed before assembly"
            )
        drafts = [
            state.draft for state in project.sections if state.draft is not None
        ]
        incomplete_reviews = [
            draft.section_id
            for draft in drafts
            if draft.quality_review_status != "passed"
        ]
        if incomplete_reviews:
            raise GroundedWritingError(
                "body assembly requires every section quality review to pass: "
                + ", ".join(incomplete_reviews)
            )
        markdown = f"# {project.handoff.outline.outline.topic}\n\n" + (
            "\n\n".join(draft.markdown for draft in drafts)
        )
        citations = [
            citation for draft in drafts for citation in draft.citations
        ]
        source_dois = list(
            dict.fromkeys(citation.doi for citation in citations)
        )
        return BodyDraftPackage(
            topic=project.handoff.outline.outline.topic,
            markdown=markdown,
            counted_words=sum(draft.counted_words for draft in drafts),
            citations=citations,
            source_dois=source_dois,
        )


def _evidence_item(card: EvidenceCard) -> SectionEvidenceItem:
    return SectionEvidenceItem(
        evidence_id=card.evidence_id,
        doi=card.doi,
        normalized_claim=card.normalized_claim,
        evidence_type=card.evidence_type,
        support_strength=card.support_strength,
        supporting_quotes=card.supporting_quotes,
    )


def _source_record(record: LiteratureLibraryRecord) -> SectionSourceRecord:
    return SectionSourceRecord(
        doi=record.doi,
        citation_key=_citation_key(record),
        title=record.title,
        authors=record.authors,
        year=record.year,
        journal=record.journal,
        abstract=record.abstract,
        evidence_tier=record.evidence_tier,
        permitted_use=record.permitted_use,
        admission_status=record.admission_status,
        centrality=record.centrality,
        supported_claim=record.supported_claim,
        suitable_section_id=record.suitable_section_id,
        use_boundary=record.use_boundary,
    )


def _citation_key(record: LiteratureLibraryRecord) -> str:
    author = record.authors[0] if record.authors else "anonymous"
    surname = author.split(",", 1)[0] if "," in author else author.split()[-1]
    surname_slug = re.sub(r"[^a-z0-9]+", "", surname.casefold()) or "author"
    suffix = re.sub(r"[^a-z0-9]+", "", record.doi.rsplit("/", 1)[-1].casefold())
    return f"{surname_slug}{record.year}_{suffix[-16:]}"[:80]


def _audit_paragraph(
    section_id: str,
    paragraph_number: int,
    paragraph: DraftParagraphProposal,
    evidence: dict[str, SectionEvidenceItem],
    sources: dict[str, SectionSourceRecord],
    output_language: str,
) -> tuple[list[SectionDraftIssue], list[CitationBinding]]:
    issues: list[SectionDraftIssue] = []
    valid_cards: list[SectionEvidenceItem] = []
    for evidence_id in paragraph.evidence_card_ids:
        card = evidence.get(evidence_id)
        if card is None:
            issues.append(
                SectionDraftIssue(
                    code="unknown_evidence_card",
                    severity="blocking",
                    paragraph_number=paragraph_number,
                    detail=f"unknown section evidence card: {evidence_id}",
                )
            )
            continue
        valid_cards.append(card)
        if card.support_strength == "partial":
            issues.append(
                SectionDraftIssue(
                    code="partial_support",
                    severity="warning",
                    paragraph_number=paragraph_number,
                    detail=f"{evidence_id} only partially supports the claim",
                )
            )

    referenced_dois = list(
        dict.fromkeys(
            [
                *(card.doi for card in valid_cards),
                *paragraph.source_dois,
            ]
        )
    )
    valid_sources: list[SectionSourceRecord] = []
    for doi in referenced_dois:
        source = sources.get(doi)
        if source is None:
            issues.append(
                SectionDraftIssue(
                    code="unknown_source_doi",
                    severity="blocking",
                    paragraph_number=paragraph_number,
                    detail=f"DOI is outside this section evidence packet: {doi}",
                )
            )
            continue
        valid_sources.append(source)
        if not _permission_allows(source.permitted_use, paragraph.role):
            issues.append(
                SectionDraftIssue(
                    code="source_permission_exceeded",
                    severity="blocking",
                    paragraph_number=paragraph_number,
                    detail=(
                        f"{doi} permits {source.permitted_use}, not "
                        f"{paragraph.role}"
                    ),
                )
            )

    if re.search(r"\[@|https?://(?:dx\.)?doi\.org/|doi\s*:", paragraph.text, re.I):
        issues.append(
            SectionDraftIssue(
                code="llm_authored_citation",
                severity="blocking",
                paragraph_number=paragraph_number,
                detail="LLM paragraph text attempted to author its own citation",
            )
        )

    workflow_leak = workflow_instruction_leak_detail(paragraph.text)
    if workflow_leak:
        issues.append(
            SectionDraftIssue(
                code="workflow_instruction_leak",
                severity="blocking",
                paragraph_number=paragraph_number,
                detail=workflow_leak,
            )
        )

    language_detail = language_mismatch_detail(
        paragraph.text,
        output_language=output_language,
    )
    if language_detail:
        issues.append(
            SectionDraftIssue(
                code="language_mismatch",
                severity="blocking",
                paragraph_number=paragraph_number,
                detail=language_detail,
            )
        )

    citations = []
    for source in valid_sources:
        source_cards = [card for card in valid_cards if card.doi == source.doi]
        pages = sorted(
            {
                quote.page_number
                for card in source_cards
                for quote in card.supporting_quotes
            }
        )
        citations.append(
            CitationBinding(
                section_id=section_id,
                paragraph_number=paragraph_number,
                citation_key=source.citation_key,
                doi=source.doi,
                evidence_card_ids=[card.evidence_id for card in source_cards],
                page_numbers=pages,
            )
        )
    return issues, citations


def _permission_allows(
    permitted_use: str,
    role: str,
) -> bool:
    if role == "detailed_evidence":
        return permitted_use == "detailed_claims"
    if role == "section_support":
        return permitted_use in {"detailed_claims", "section_support"}
    return permitted_use in {
        "detailed_claims",
        "section_support",
        "background_only",
    }


def _render_paragraph(
    text: str,
    citations: list[CitationBinding],
) -> str:
    if not citations:
        return text.strip()
    # Page numbers remain available in CitationBinding and the audit export.
    # Generated paragraphs are paraphrases, not direct quotations, so the
    # submission text uses ordinary in-text citations without page locators.
    markers = [f"@{citation.citation_key}" for citation in citations]
    return f"{text.strip()} [{'; '.join(markers)}]"


def _page_locator(page_numbers: list[int]) -> str:
    if not page_numbers:
        return ""
    label = "p." if len(page_numbers) == 1 else "pp."
    return f"{label} {', '.join(str(page) for page in page_numbers)}"


def count_writing_units(
    value: str,
    *,
    counting_policy: str = "chinese_chars_and_english_words",
) -> int:
    """Count Chinese characters plus English-like words for stable UI guidance."""

    if counting_policy == "words":
        return len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", value, re.UNICODE))
    if counting_policy != "chinese_chars_and_english_words":
        raise ValueError(f"unsupported counting policy: {counting_policy}")
    chinese = re.findall(r"[\u4e00-\u9fff]", value)
    without_chinese = re.sub(r"[\u4e00-\u9fff]", " ", value)
    words = re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", without_chinese, re.UNICODE)
    return len(chinese) + len(words)


def _ai_generation_prohibitions(
    handoff: V04WritingHandoff,
) -> list[str]:
    policy = (
        handoff.requirement_policy
        or RequirementPolicyCompiler().compile(handoff.requirement)
    )
    return ai_generation_prohibitions(policy)
