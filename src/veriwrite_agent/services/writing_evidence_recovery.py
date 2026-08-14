"""Route V0.4 evidence shortages to bounded recovery instead of blind rewriting."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

from veriwrite_agent.models.evidence import EvidenceCard, EvidenceLibrary
from veriwrite_agent.models.literature_selection import (
    ConfirmedLiteratureSearchBlueprint,
    LiteratureSearchBlueprint,
)
from veriwrite_agent.models.requirements import StrictModel
from veriwrite_agent.models.writing import SectionEvidencePacket
from veriwrite_agent.models.writing_plan import (
    GroundedWritingPlan,
    WritingParagraphPlan,
    WritingSectionPlan,
)
from veriwrite_agent.models.writing_handoff import V04WritingHandoff


def merge_recovery_handoffs(
    previous: V04WritingHandoff,
    current: V04WritingHandoff,
    *,
    affected_section_ids: set[str],
) -> V04WritingHandoff:
    """Add recovered evidence without invalidating accepted, unaffected chapters."""

    if previous.requirement != current.requirement:
        raise ValueError("evidence recovery cannot change the confirmed requirement")
    if previous.requirement_policy != current.requirement_policy:
        raise ValueError("evidence recovery cannot change the executable requirement policy")
    previous_sections = {
        section.section_id: section for section in previous.outline.outline.sections
    }
    merged_outline_sections = [
        (
            previous_sections.get(section.section_id, section)
            if section.section_id not in affected_section_ids
            else section
        )
        for section in current.outline.outline.sections
    ]
    merged_library = _merge_evidence_libraries(
        previous.evidence_library,
        current.evidence_library,
    )
    outline = current.outline.model_copy(
        update={
            "outline": current.outline.outline.model_copy(
                update={"sections": merged_outline_sections}
            )
        }
    )
    return V04WritingHandoff.model_validate(
        current.model_copy(
            update={
                "outline": outline,
                "evidence_library": merged_library,
            }
        ).model_dump(mode="json")
    )


def _merge_evidence_libraries(
    previous: EvidenceLibrary,
    current: EvidenceLibrary,
) -> EvidenceLibrary:
    if (
        previous.requirement_policy_fingerprint
        != current.requirement_policy_fingerprint
    ):
        raise ValueError("evidence recovery libraries use different requirement policies")

    old_records = {record.doi: record for record in previous.records}
    new_records = {record.doi: record for record in current.records}
    old_documents = {document.doi: document for document in previous.documents}
    new_documents = {document.doi: document for document in current.documents}
    ordered_dois = list(dict.fromkeys([*new_records, *old_records]))
    origins: dict[str, str] = {}
    records = []
    documents = []
    for doi in ordered_dois:
        old_record = old_records.get(doi)
        new_record = new_records.get(doi)
        if old_record is None:
            origin = "current"
        elif new_record is None:
            origin = "previous"
        elif (
            old_record.evidence_status == "full_text_verified"
            and new_record.evidence_status != "full_text_verified"
        ):
            origin = "previous"
        else:
            origin = "current"
        old_document = old_documents.get(doi)
        new_document = new_documents.get(doi)
        if (
            old_document is not None
            and new_document is not None
            and old_document.status == "available"
            and new_document.status == "available"
            and old_document.sha256 == new_document.sha256
        ):
            origin = "both"
        origins[doi] = origin
        records.append(
            (new_record if origin in {"current", "both"} else old_record)
        )
        chosen_document = (
            new_document if origin in {"current", "both"} else old_document
        )
        if chosen_document is not None:
            documents.append(chosen_document)

    def chosen_items(field: str) -> list[object]:
        old_items = getattr(previous, field)
        new_items = getattr(current, field)
        selected: list[object] = []
        for doi in ordered_dois:
            origin = origins[doi]
            source = new_items if origin in {"current", "both"} else old_items
            matching = [item for item in source if getattr(item, "doi", None) == doi]
            if not matching and origin == "both":
                matching = [
                    item for item in old_items if getattr(item, "doi", None) == doi
                ]
            selected.extend(matching)
        return selected

    cards = chosen_items("evidence_cards")
    # When the recovered PDF is byte-identical, retain old confirmed cards that
    # accepted prose still cites and add genuinely new cards by stable evidence ID.
    card_ids = {card.evidence_id for card in cards}
    for card in previous.evidence_cards:
        if origins.get(card.doi) == "both" and card.evidence_id not in card_ids:
            cards.append(card)
            card_ids.add(card.evidence_id)

    payload = current.model_copy(
        update={
            "records": records,
            "documents": documents,
            "extractions": chosen_items("extractions"),
            "page_selections": chosen_items("page_selections"),
            "pages": chosen_items("pages"),
            "evidence_cards": cards,
            "literature_matrix": chosen_items("literature_matrix"),
            "unresolved_issues": list(
                dict.fromkeys(
                    [
                        *previous.unresolved_issues,
                        *current.unresolved_issues,
                    ]
                )
            ),
        }
    )
    return EvidenceLibrary.model_validate(payload.model_dump(mode="json"))


class ParagraphEvidenceGap(StrictModel):
    """One paragraph whose requested argument is stronger than its evidence tier."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    section_title: str = Field(min_length=1)
    paragraph_number: int = Field(ge=1)
    reason: Literal[
        "comparison_requires_full_text",
        "detailed_claim_requires_full_text",
    ]
    claim_focus: str = Field(min_length=1)
    central_question: str = Field(min_length=1)
    missing_full_text_dois: list[str] = Field(default_factory=list)
    available_direct_evidence_dois: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(min_length=1, max_length=4)
    detail: str = Field(min_length=1)

    @field_validator(
        "missing_full_text_dois",
        "available_direct_evidence_dois",
        "search_queries",
        mode="after",
    )
    @classmethod
    def values_must_be_unique(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class WritingEvidenceRecoveryRequest(StrictModel):
    """Durable cross-stage instruction produced by a V0.4 evidence gap."""

    schema_version: Literal["0.4-evidence-recovery.0"] = "0.4-evidence-recovery.0"
    status: Literal[
        "pending_full_text",
        "pending_search",
        "ready_to_resume",
        "resolved",
        "blocked",
    ]
    source_plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    affected_section_ids: list[str] = Field(min_length=1)
    gaps: list[ParagraphEvidenceGap] = Field(default_factory=list)
    requested_core_dois: list[str] = Field(default_factory=list)
    unavailable_full_text_dois: list[str] = Field(default_factory=list)
    search_queries_by_section: dict[str, list[str]] = Field(default_factory=dict)
    repair_feedback_by_section: dict[str, list[str]] = Field(default_factory=dict)
    recovery_round: int = Field(default=1, ge=1)
    max_recovery_rounds: int = Field(default=4, ge=1, le=5)
    planning_repair_round: int = Field(default=0, ge=0)
    max_planning_repair_rounds: int = Field(default=2, ge=0, le=3)
    blocked_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "affected_section_ids",
        "requested_core_dois",
        "unavailable_full_text_dois",
        mode="after",
    )
    @classmethod
    def identifiers_must_be_unique(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def recovery_must_have_actionable_input(self) -> WritingEvidenceRecoveryRequest:
        if not self.gaps and not self.repair_feedback_by_section:
            raise ValueError("evidence recovery requires gaps or reviewer plan feedback")
        return self


class WritingEvidenceRecoveryService:
    """Detect evidence-tier contradictions and prepare a resumable fallback."""

    _COMPARISON_MOVES = frozenset({"compare_studies", "analyze_difference"})
    _DETAIL_TERMS = re.compile(
        r"(?:accuracy|performance|architecture|metric|benchmark|compare|comparison|"
        r"精度|性能|架构|结构|指标|比较|优于|误差)",
        re.IGNORECASE,
    )

    def audit_section(
        self,
        section_plan: WritingSectionPlan,
        packet: SectionEvidencePacket,
        *,
        paragraph_numbers: set[int] | None = None,
    ) -> tuple[ParagraphEvidenceGap, ...]:
        if section_plan.section_id != packet.section_id:
            raise ValueError("evidence recovery audit section does not match packet")
        evidence_by_id = {item.evidence_id: item for item in packet.evidence_items}
        sources_by_doi = {source.doi: source for source in packet.sources}
        gaps: list[ParagraphEvidenceGap] = []
        for paragraph in section_plan.paragraphs:
            if (
                paragraph_numbers is not None
                and paragraph.paragraph_number not in paragraph_numbers
            ):
                continue
            direct_dois = list(
                dict.fromkeys(
                    item.doi
                    for evidence_id in paragraph.evidence_card_ids
                    if (item := evidence_by_id.get(evidence_id)) is not None
                    and item.support_strength == "direct"
                )
            )
            metadata_only = [
                doi
                for doi in paragraph.source_dois
                if (source := sources_by_doi.get(doi)) is not None
                and source.permitted_use == "background_only"
                and doi not in direct_dois
            ]
            comparison_gap = (
                paragraph.argument_move in self._COMPARISON_MOVES
                and len(direct_dois) < 2
            )
            detail_gap = (
                bool(metadata_only)
                and not direct_dois
                and paragraph.argument_move != "author_judgment"
                and self._DETAIL_TERMS.search(
                    " ".join(
                        part
                        for part in (
                            paragraph.purpose,
                            paragraph.claim_focus,
                            paragraph.central_question,
                            paragraph.comparison_axis or "",
                        )
                        if part
                    )
                )
                is not None
            )
            if not comparison_gap and not detail_gap:
                continue
            reason = (
                "comparison_requires_full_text"
                if comparison_gap
                else "detailed_claim_requires_full_text"
            )
            gaps.append(
                ParagraphEvidenceGap(
                    section_id=section_plan.section_id,
                    section_title=section_plan.title,
                    paragraph_number=paragraph.paragraph_number,
                    reason=reason,
                    claim_focus=paragraph.claim_focus,
                    central_question=paragraph.central_question,
                    missing_full_text_dois=metadata_only,
                    available_direct_evidence_dois=direct_dois,
                    search_queries=_recovery_queries(section_plan, paragraph),
                    detail=_gap_detail(paragraph, metadata_only, direct_dois),
                )
            )
        return tuple(gaps)

    def request(
        self,
        *,
        plan_fingerprint: str,
        gaps: tuple[ParagraphEvidenceGap, ...],
    ) -> WritingEvidenceRecoveryRequest:
        if not gaps:
            raise ValueError("evidence recovery requires at least one gap")
        requested_dois = list(
            dict.fromkeys(
                doi
                for gap in gaps
                for doi in gap.missing_full_text_dois[
                    : 2 if gap.reason == "comparison_requires_full_text" else 1
                ]
            )
        )
        queries_by_section: dict[str, list[str]] = {}
        for gap in gaps:
            queries_by_section.setdefault(gap.section_id, [])
            queries_by_section[gap.section_id].extend(gap.search_queries)
        queries_by_section = {
            section_id: list(dict.fromkeys(queries))[:4]
            for section_id, queries in queries_by_section.items()
        }
        return WritingEvidenceRecoveryRequest(
            status="pending_full_text" if requested_dois else "pending_search",
            source_plan_fingerprint=plan_fingerprint,
            affected_section_ids=list(dict.fromkeys(gap.section_id for gap in gaps)),
            gaps=list(gaps),
            requested_core_dois=requested_dois,
            search_queries_by_section=queries_by_section,
        )

    def validate_resolution(
        self,
        plan: GroundedWritingPlan,
        packets: list[SectionEvidencePacket],
        *,
        affected_section_ids: list[str],
    ) -> tuple[str, ...]:
        """Verify that a recovered plan is executable before closing recovery.

        A recovery request is not resolved merely because search or downgrade code ran.
        Every surviving paragraph in an affected section must still have known,
        permission-compatible support, and the evidence-tier audit must be clean.
        """

        packet_by_section = {packet.section_id: packet for packet in packets}
        section_by_id = {section.section_id: section for section in plan.sections}
        errors: list[str] = []
        for section_id in affected_section_ids:
            section = section_by_id.get(section_id)
            packet = packet_by_section.get(section_id)
            if section is None or packet is None:
                errors.append(f"{section_id}: recovered section or evidence packet is missing")
                continue
            evidence_by_id = {
                item.evidence_id: item for item in packet.evidence_items
            }
            source_by_doi = {source.doi: source for source in packet.sources}
            for paragraph in section.paragraphs:
                support_dois = set(paragraph.source_dois)
                for evidence_id in paragraph.evidence_card_ids:
                    evidence = evidence_by_id.get(evidence_id)
                    if evidence is None:
                        errors.append(
                            f"{section_id} paragraph {paragraph.paragraph_number}: "
                            f"unknown evidence card {evidence_id}"
                        )
                        continue
                    support_dois.add(evidence.doi)
                if not support_dois:
                    errors.append(
                        f"{section_id} paragraph {paragraph.paragraph_number}: "
                        "no surviving support"
                    )
                    continue
                for doi in sorted(support_dois):
                    source = source_by_doi.get(doi)
                    if source is None:
                        errors.append(
                            f"{section_id} paragraph {paragraph.paragraph_number}: "
                            f"unknown source {doi}"
                        )
                    elif not _permission_allows(source.permitted_use, paragraph.role):
                        errors.append(
                            f"{section_id} paragraph {paragraph.paragraph_number}: "
                            f"source {doi} is not permitted for {paragraph.role}"
                        )
            for gap in self.audit_section(section, packet):
                errors.append(
                    f"{section_id} paragraph {gap.paragraph_number}: {gap.reason}"
                )
        return tuple(dict.fromkeys(errors))

    def enrich_search_blueprint(
        self,
        confirmed: ConfirmedLiteratureSearchBlueprint,
        request: WritingEvidenceRecoveryRequest,
    ) -> ConfirmedLiteratureSearchBlueprint:
        """Prioritize targeted replacement queries without weakening any boundary."""

        payload = confirmed.blueprint.model_dump(mode="json")
        matched: set[str] = set()
        for theme in payload["themes"]:
            theme_id = str(theme["theme_id"])
            recovery_queries = request.search_queries_by_section.get(theme_id, [])
            if not recovery_queries:
                continue
            matched.add(theme_id)
            theme["search_queries"] = list(
                dict.fromkeys([*recovery_queries, *theme["search_queries"]])
            )[:4]
            theme["priority"] = min(10, int(theme.get("priority", 1)) + 2)
        missing_sections = set(request.affected_section_ids) - matched
        if missing_sections:
            raise ValueError(
                "evidence recovery sections are absent from the V0.2 blueprint: "
                + ", ".join(sorted(missing_sections))
            )
        payload["max_candidates"] = min(
            1000,
            max(int(payload["max_candidates"]), int(payload["target_total"]) * 8),
        )
        blueprint = LiteratureSearchBlueprint.model_validate(payload)
        return confirmed.model_copy(
            update={
                "blueprint": blueprint,
                "confirmed_at": datetime.now(timezone.utc),
                "confirmation_note": (
                    "V0.4 detected a full-text evidence gap and automatically added "
                    "boundary-preserving replacement queries."
                ),
            }
        )


def downgrade_unresolved_evidence_claims(
    plan: GroundedWritingPlan,
    gaps: list[ParagraphEvidenceGap],
) -> GroundedWritingPlan:
    """Replace impossible detailed claims with explicit metadata-bounded background."""

    gap_by_paragraph = {
        (gap.section_id, gap.paragraph_number): gap for gap in gaps
    }
    affected = set(gap_by_paragraph)
    sections: list[WritingSectionPlan] = []
    for section in plan.sections:
        paragraphs: list[WritingParagraphPlan] = []
        for paragraph in section.paragraphs:
            if (section.section_id, paragraph.paragraph_number) not in affected:
                paragraphs.append(paragraph)
                continue
            gap = gap_by_paragraph[(section.section_id, paragraph.paragraph_number)]
            source_dois = list(
                dict.fromkeys(
                    [
                        *paragraph.source_dois,
                        *gap.missing_full_text_dois,
                        *gap.available_direct_evidence_dois,
                    ]
                )
            )
            # Keep any already-confirmed direct evidence. The downgrade only removes
            # claims that exceed the current permission boundary; it must not discard
            # valid support that can still ground a cautious background statement.
            evidence_card_ids = list(paragraph.evidence_card_ids)
            paragraphs.append(
                paragraph.model_copy(
                    update={
                        "role": "background",
                        "purpose": (
                            "State a bounded background overview and the limits of the "
                            "available records."
                        ),
                        "claim_focus": (
                            f"{section.title} includes a research direction represented "
                            "by the admitted sources, but the available records permit "
                            "only a general description of its scope."
                        ),
                        "central_question": (
                            "What is the general scope of this direction, and which "
                            "conclusions remain outside the available evidence?"
                        ),
                        "argument_move": "frame_problem",
                        "comparison_axis": None,
                        "deferred_argument": paragraph.argument_move,
                        "deferred_comparison_axis": paragraph.comparison_axis,
                        "deferred_purpose": paragraph.purpose,
                        "deferred_claim_focus": paragraph.claim_focus,
                        "deferred_central_question": paragraph.central_question,
                        "deferred_recovery_dois": list(
                            dict.fromkeys(gap.missing_full_text_dois)
                        ),
                        "evidence_card_ids": evidence_card_ids,
                        "source_dois": source_dois,
                    }
                )
            )
        sections.append(section.model_copy(update={"paragraphs": paragraphs}))
    canonical = json.dumps(
        {
            "fallback": "metadata-bounded-claim-downgrade-v1",
            "source_plan_fingerprint": plan.plan_fingerprint,
            "affected": sorted(affected),
            "sections": [section.model_dump(mode="json") for section in sections],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return GroundedWritingPlan.model_validate(
        plan.model_copy(
            update={"sections": sections, "plan_fingerprint": fingerprint}
        ).model_dump(mode="json")
    )


_COMPARISON_ARGUMENT_MOVES = frozenset({"compare_studies", "analyze_difference"})


def upgrade_deferred_evidence_claims(
    plan: GroundedWritingPlan,
    library: EvidenceLibrary,
) -> GroundedWritingPlan:
    """Restore downgraded paragraphs once their deferred PDFs are full-text verified.

    ``downgrade_unresolved_evidence_claims`` saved each gap paragraph's original
    argument intent in ``deferred_argument``/``deferred_comparison_axis``/
    ``deferred_recovery_dois`` so a later PDF-backed pass can upgrade it back from
    background to detailed evidence without losing the planned claim. This pure
    function only flips a paragraph when the library actually contains full-text
    records and confirmed direct evidence cards for enough of its source DOIs;
    paragraphs whose PDFs are still missing remain background as a safe fallback.
    """

    full_text_dois = {
        record.doi
        for record in library.records
        if record.evidence_status == "full_text_verified"
        and record.permitted_use == "detailed_claims"
    }
    direct_cards_by_doi: dict[str, list[EvidenceCard]] = {}
    for card in library.evidence_cards:
        if card.review_status == "confirmed" and card.support_strength == "direct":
            direct_cards_by_doi.setdefault(card.doi, []).append(card)

    sections: list[WritingSectionPlan] = []
    upgraded_any = False
    for section in plan.sections:
        paragraphs: list[WritingParagraphPlan] = []
        for paragraph in section.paragraphs:
            upgraded = _upgrade_paragraph_if_ready(
                paragraph,
                full_text_dois,
                direct_cards_by_doi,
            )
            if upgraded is None:
                paragraphs.append(paragraph)
            else:
                paragraphs.append(upgraded)
                upgraded_any = True
        sections.append(section.model_copy(update={"paragraphs": paragraphs}))

    if not upgraded_any:
        return plan

    canonical = json.dumps(
        {
            "enhancement": "deferred-evidence-upgrade-v1",
            "source_plan_fingerprint": plan.plan_fingerprint,
            "sections": [section.model_dump(mode="json") for section in sections],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return GroundedWritingPlan.model_validate(
        plan.model_copy(
            update={"sections": sections, "plan_fingerprint": fingerprint}
        ).model_dump(mode="json")
    )


def _upgrade_paragraph_if_ready(
    paragraph: WritingParagraphPlan,
    full_text_dois: set[str],
    direct_cards_by_doi: dict[str, list[EvidenceCard]],
) -> WritingParagraphPlan | None:
    if paragraph.deferred_argument is None:
        return None
    recovered = [
        doi
        for doi in paragraph.source_dois
        if doi in full_text_dois and doi in direct_cards_by_doi
    ]
    required = (
        2 if paragraph.deferred_argument in _COMPARISON_ARGUMENT_MOVES else 1
    )
    if len(recovered) < required:
        return None
    evidence_card_ids = [
        direct_cards_by_doi[doi][0].evidence_id for doi in recovered
    ][:5]
    axis = paragraph.deferred_comparison_axis
    if paragraph.deferred_argument in _COMPARISON_ARGUMENT_MOVES:
        fallback_purpose = (
            "Present the recovered full-text comparison and its evidence-backed "
            "conclusion."
        )
        fallback_claim_focus = (
            f"Compare the recovered studies along {axis}."
            if axis
            else "Compare the recovered studies on their verified findings."
        )
        fallback_central_question = (
            f"What does the full-text evidence show about {axis}?"
            if axis
            else "What does the full-text evidence show about the comparison?"
        )
    else:
        fallback_purpose = (
            "Present the recovered full-text finding and its evidence-backed "
            "conclusion."
        )
        fallback_claim_focus = (
            "State the detailed finding supported by the recovered full-text evidence."
        )
        fallback_central_question = (
            "What does the recovered full-text evidence establish?"
        )
    return paragraph.model_copy(
        update={
            "role": "detailed_evidence",
            "purpose": paragraph.deferred_purpose or fallback_purpose,
            "claim_focus": paragraph.deferred_claim_focus or fallback_claim_focus,
            "central_question": (
                paragraph.deferred_central_question or fallback_central_question
            ),
            "argument_move": paragraph.deferred_argument,
            "comparison_axis": axis,
            "evidence_card_ids": evidence_card_ids,
            "source_dois": list(dict.fromkeys([*paragraph.source_dois, *recovered])),
            "deferred_argument": None,
            "deferred_comparison_axis": None,
            "deferred_purpose": None,
            "deferred_claim_focus": None,
            "deferred_central_question": None,
            "deferred_recovery_dois": [],
        }
    )


def deferred_recovery_dois(plan: GroundedWritingPlan) -> list[str]:
    """Aggregate the DOIs whose full text is still needed to enhance downgraded paragraphs.

    This is the batch-download list surfaced to the user after a conservative draft
    completes: every deferred paragraph contributed the DOIs it downgraded for, and a
    later PDF-backed pass upgrades those paragraphs once these DOIs are full-text
    verified in the evidence library.
    """

    return list(
        dict.fromkeys(
            doi
            for section in plan.sections
            for paragraph in section.paragraphs
            for doi in paragraph.deferred_recovery_dois
        )
    )


def deferred_enhancement_targets(
    plan: GroundedWritingPlan,
    library: EvidenceLibrary,
) -> set[tuple[str, int]]:
    """Return (section_id, paragraph_number) that a PDF-backed pass would upgrade."""

    upgraded = upgrade_deferred_evidence_claims(plan, library)
    previous_by_section = {
        section.section_id: section for section in plan.sections
    }
    targets: set[tuple[str, int]] = set()
    for section in upgraded.sections:
        previous = previous_by_section[section.section_id]
        for old, current in zip(
            previous.paragraphs,
            section.paragraphs,
            strict=True,
        ):
            if old.deferred_argument is not None and current.deferred_argument is None:
                targets.add((section.section_id, current.paragraph_number))
    return targets


def deferred_section_ids(plan: GroundedWritingPlan) -> set[str]:
    """Return section IDs that contain at least one deferred (downgraded) paragraph."""

    return {
        section.section_id
        for section in plan.sections
        if any(paragraph.deferred_argument is not None for paragraph in section.paragraphs)
    }


def preserve_unresolved_deferred_sections(
    previous: GroundedWritingPlan,
    candidate: GroundedWritingPlan,
    library: EvidenceLibrary,
) -> GroundedWritingPlan:
    """Prevent replanning from silently restoring claims that still need PDFs.

    A semantic replanner may redesign an affected chapter, but it does not have
    authority to erase an evidence-boundary marker. Only the explicit full-text
    upgrade pass may do that. Sections whose deferred evidence has become available
    are first upgraded normally; all other deferred sections retain their conservative
    paragraph plan while unrelated candidate sections remain untouched.
    """

    previous_ids = [section.section_id for section in previous.sections]
    candidate_ids = [section.section_id for section in candidate.sections]
    if previous_ids != candidate_ids:
        raise ValueError("replanned section order changed while evidence was deferred")
    guarded_previous = upgrade_deferred_evidence_claims(previous, library)
    unresolved = deferred_section_ids(guarded_previous)
    if not unresolved:
        return candidate
    previous_by_id = {
        section.section_id: section for section in guarded_previous.sections
    }
    sections = [
        previous_by_id[section.section_id]
        if section.section_id in unresolved
        else section
        for section in candidate.sections
    ]
    canonical = json.dumps(
        {
            "guard": "preserve-unresolved-deferred-sections-v1",
            "previous_plan_fingerprint": previous.plan_fingerprint,
            "candidate_plan_fingerprint": candidate.plan_fingerprint,
            "sections": [section.model_dump(mode="json") for section in sections],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return GroundedWritingPlan.model_validate(
        candidate.model_copy(
            update={
                "sections": sections,
                "plan_fingerprint": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest(),
            }
        ).model_dump(mode="json")
    )


def _permission_allows(permitted_use: str, role: str) -> bool:
    if role == "detailed_evidence":
        return permitted_use == "detailed_claims"
    if role == "section_support":
        return permitted_use in {"detailed_claims", "section_support"}
    return permitted_use in {
        "detailed_claims",
        "section_support",
        "background_only",
    }


def _recovery_queries(
    section: WritingSectionPlan,
    paragraph: WritingParagraphPlan,
) -> list[str]:
    candidates = (
        f"{paragraph.claim_focus} {paragraph.comparison_axis or ''}",
        f"{paragraph.central_question} {section.title}",
        f"{section.purpose} {paragraph.claim_focus}",
    )
    queries: list[str] = []
    for candidate in candidates:
        clean = " ".join(candidate.split()).strip(" ?？。.;；:")
        clean = re.sub(r"\b(?:AND|OR|NOT)\b", " ", clean)
        clean = " ".join(clean.split())[:240].strip()
        if clean and clean.casefold() not in {item.casefold() for item in queries}:
            queries.append(clean)
    return queries[:3]


def _gap_detail(
    paragraph: WritingParagraphPlan,
    metadata_only: list[str],
    direct_dois: list[str],
) -> str:
    missing = "、".join(metadata_only) or "当前比较对象"
    return (
        f"第 {paragraph.paragraph_number} 段要求执行 {paragraph.argument_move}，"
        f"但只有 {len(direct_dois)} 篇来源具有直接全文证据；"
        f"{missing} 目前仅允许作背景信息。"
    )
