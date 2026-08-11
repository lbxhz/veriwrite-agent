"""Route V0.4 evidence shortages to bounded recovery instead of blind rewriting."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

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
            evidence_card_ids = [] if source_dois else paragraph.evidence_card_ids
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
