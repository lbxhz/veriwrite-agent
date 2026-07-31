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
    DraftParagraphProposal,
    SectionDraft,
    SectionDraftIssue,
    SectionDraftProposal,
    SectionEvidenceItem,
    SectionEvidencePacket,
    SectionSourceRecord,
    V04WritingProject,
    WritingSectionState,
)
from veriwrite_agent.models.writing_handoff import V04WritingHandoff


class GroundedWritingError(ValueError):
    """Raised when a writing step violates a confirmed contract."""


class SectionEvidencePacketBuilder:
    """Resolve one outline section to its permitted V0.3 evidence context."""

    def build(
        self,
        handoff: V04WritingHandoff,
        section_id: str,
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
        policy_reasons = _ai_generation_prohibitions(handoff)
        return SectionEvidencePacket(
            section_id=section.section_id,
            title=section.title,
            purpose=section.purpose,
            target_words=section.target_words,
            research_questions=section.research_questions,
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
                        "full-text evidence cards. Metadata-only B sources may support "
                        "general section claims; C sources are background only. Group "
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
            raise GroundedWritingError(
                "LLM section output violates the V0.4 data contract: "
                f"{exc.errors(include_url=False)[:8]}"
            ) from exc
        if proposal.section_id != packet.section_id:
            raise GroundedWritingError(
                "LLM changed the confirmed section_id"
            )
        return GroundedSectionDraftService().create(packet, proposal)


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
            )
            issues.extend(paragraph_issues)
            citations.extend(paragraph_citations)
            rendered_paragraphs.append(
                _render_paragraph(paragraph.text, paragraph_citations)
            )

        markdown = f"## {packet.title}\n\n" + "\n\n".join(
            rendered_paragraphs
        )
        counted_words = sum(
            count_writing_units(paragraph.text)
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
    markers = []
    for citation in citations:
        locator = _page_locator(citation.page_numbers)
        markers.append(
            f"@{citation.citation_key}{', ' + locator if locator else ''}"
        )
    return f"{text.strip()} [{'; '.join(markers)}]"


def _page_locator(page_numbers: list[int]) -> str:
    if not page_numbers:
        return ""
    label = "p." if len(page_numbers) == 1 else "pp."
    return f"{label} {', '.join(str(page) for page in page_numbers)}"


def count_writing_units(value: str) -> int:
    """Count Chinese characters plus English-like words for stable UI guidance."""

    chinese = re.findall(r"[\u4e00-\u9fff]", value)
    without_chinese = re.sub(r"[\u4e00-\u9fff]", " ", value)
    words = re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", without_chinese, re.UNICODE)
    return len(chinese) + len(words)


def _ai_generation_prohibitions(
    handoff: V04WritingHandoff,
) -> list[str]:
    requirement = handoff.requirement.requirement
    statements = list(requirement.ai_policy.prohibited_uses)
    statements.extend(
        rule.description
        for rule in requirement.policy_rules
        if rule.category in {"ai_usage", "academic_integrity"}
    )
    generation_pattern = re.compile(
        r"AI.*(?:生成|撰写|写作|代写|句子|段落|正文|洗稿)"
        r"|(?:禁止|不允许)[^。；]{0,40}(?:AI|人工智能)"
        r"[^。；]{0,40}(?:生成|撰写|写作|代写|句子|段落|正文|洗稿)"
        r"|(?:generate|draft|write|compose).*(?:AI|artificial intelligence)"
        r"|(?:AI|artificial intelligence).*(?:generate|draft|write|compose)",
        re.IGNORECASE,
    )
    return [
        statement.strip()
        for statement in statements
        if statement.strip() and generation_pattern.search(statement)
    ]
