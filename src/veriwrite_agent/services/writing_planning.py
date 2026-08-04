"""Evidence-first V0.4 planning, paragraph writing, and resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.writing import (
    DraftParagraphProposal,
    SectionDraft,
    SectionDraftProposal,
    SectionEvidenceItem,
    SectionEvidencePacket,
    SectionSourceRecord,
)
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.models.writing_plan import (
    GroundedWritingPlan,
    ParagraphEvidencePacket,
    ParagraphPlanProposal,
    ParagraphTextProposal,
    SectionPlanProposal,
    WritingParagraphPlan,
    WritingSectionPlan,
)
from veriwrite_agent.services.grounded_writing import (
    GroundedSectionDraftService,
    GroundedWritingError,
    SectionEvidencePacketBuilder,
    count_writing_units,
)
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler


class WritingPlanError(ValueError):
    """Raised when semantic planning cannot compile to real evidence authority."""


class ParagraphLengthError(ValueError):
    """Raised when one model paragraph greatly exceeds its locked word budget."""


class ParagraphCitationError(ValueError):
    """Raised when paragraph prose attempts to create its own citation."""


@dataclass(frozen=True)
class WritingPlanCoverageRepair:
    """A source-coverage plan update plus the paragraphs it actually changed."""

    plan: GroundedWritingPlan
    changed_paragraph_numbers: dict[str, tuple[int, ...]]


class GroundedWritingPlanner:
    """Plan paragraph purposes and evidence before any body prose is generated."""

    def __init__(
        self,
        client: LLMClient,
        *,
        cache: WritingPlanRuntimeCache | None = None,
        reuse_cache: bool = True,
    ) -> None:
        self._client = client
        self._cache = cache
        self._reuse_cache = reuse_cache

    def plan(self, handoff: V04WritingHandoff) -> GroundedWritingPlan:
        section_plans: list[WritingSectionPlan] = []
        for outline_section in handoff.outline.outline.sections:
            packet = SectionEvidencePacketBuilder().build(
                handoff,
                outline_section.section_id,
            )
            cached = (
                self._cache.load_section(packet)
                if self._cache and self._reuse_cache
                else None
            )
            if cached is None:
                cached = self._plan_section(packet)
                if self._cache:
                    self._cache.save_section(packet, cached)
            section_plans.append(cached)

        required_source_dois = _required_source_dois(handoff)
        section_plans = _apply_required_source_coverage(
            handoff,
            section_plans,
            required_source_dois=required_source_dois,
        )
        fingerprint = _writing_plan_fingerprint(
            handoff.outline.outline.topic,
            section_plans,
            required_source_dois=required_source_dois,
        )
        return GroundedWritingPlan(
            topic=handoff.outline.outline.topic,
            plan_fingerprint=fingerprint,
            required_source_dois=required_source_dois,
            sections=section_plans,
        )

    def _plan_section(self, packet: SectionEvidencePacket) -> WritingSectionPlan:
        paragraph_count = _paragraph_count(packet.target_words)
        sources_by_doi = {source.doi: source for source in packet.sources}
        evidence_aliases = {
            f"E{index:03d}": item
            for index, item in enumerate(packet.evidence_items, 1)
        }
        source_aliases = {
            f"S{index:03d}": source
            for index, source in enumerate(packet.sources, 1)
        }
        payload = {
            "section": {
                "section_id": packet.section_id,
                "title": packet.title,
                "purpose": packet.purpose,
                "target_words": packet.target_words,
                "research_questions": packet.research_questions,
                "required_paragraph_count": paragraph_count,
            },
            "evidence_catalog": [
                {
                    "ref": alias,
                    "doi": item.doi,
                    "evidence_type": item.evidence_type,
                    "support_strength": item.support_strength,
                    "normalized_claim": item.normalized_claim,
                    "allowed_roles": _allowed_roles(
                        sources_by_doi[item.doi].permitted_use
                    ),
                    "page_numbers": [
                        quote.page_number for quote in item.supporting_quotes
                    ],
                    "source_excerpt": item.supporting_quotes[0].exact_text[:240],
                }
                for alias, item in evidence_aliases.items()
            ],
            "source_catalog": [
                {
                    "ref": alias,
                    "doi": source.doi,
                    "title": source.title,
                    "year": source.year,
                    "evidence_tier": source.evidence_tier,
                    "permitted_use": source.permitted_use,
                    "allowed_roles": _allowed_roles(source.permitted_use),
                    "abstract": (source.abstract or "")[:400],
                }
                for alias, source in source_aliases.items()
            ],
            "allowed_support_refs_by_role": {
                role: {
                    "evidence_refs": [
                        alias
                        for alias, item in evidence_aliases.items()
                        if _permission_allows(
                            sources_by_doi[item.doi].permitted_use,
                            role,
                        )
                    ],
                    "source_refs": [
                        alias
                        for alias, source in source_aliases.items()
                        if _permission_allows(source.permitted_use, role)
                    ],
                }
                for role in (
                    "detailed_evidence",
                    "section_support",
                    "background",
                    "synthesis",
                )
            },
        }
        schema = json.dumps(
            SectionPlanProposal.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You plan one scholarly section before prose is written. Return JSON "
                    "only and do not write body paragraphs. Return exactly the requested "
                    "number of paragraph plans in a coherent order. Use only the short "
                    "E### and S### aliases supplied by the application. For each paragraph "
                    "role, use refs only from allowed_support_refs_by_role[role]. "
                    "detailed_evidence requires one to five evidence_refs. A source whose "
                    "permitted_use is background_only may support only background or "
                    "synthesis, never section_support or detailed_evidence. Every paragraph "
                    "needs at least one permitted evidence_ref or source_ref. "
                    "Keep claim_focus narrow enough for one paragraph. relative_weight is "
                    "an integer from 1 to 10; code assigns exact word budgets. "
                    f"The response must satisfy this JSON Schema: {schema}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(2):
            raw = self._client.complete(
                messages,
                response_format={"type": "json_object"},
            )
            try:
                proposal = SectionPlanProposal.model_validate_json(raw)
                return _compile_section_plan(
                    packet,
                    proposal,
                    evidence_aliases=evidence_aliases,
                    source_aliases=source_aliases,
                    expected_paragraph_count=paragraph_count,
                )
            except (ValidationError, WritingPlanError) as exc:
                last_error = exc
                if attempt == 0:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Repair only the plan JSON. Do not write prose. The previous "
                                f"plan failed deterministic validation: {_short_error(exc)}. "
                                "For every paragraph, choose refs only from "
                                "allowed_support_refs_by_role for its role."
                            ),
                        },
                    ]
        raise WritingPlanError(
            f"section planning failed after one repair: {_short_error(last_error)}"
        ) from last_error


def repair_writing_plan_source_coverage(
    handoff: V04WritingHandoff,
    plan: GroundedWritingPlan,
) -> WritingPlanCoverageRepair:
    """Add only missing required sources and report the affected paragraphs."""

    required_source_dois = _required_source_dois(handoff)
    repaired_sections = _apply_required_source_coverage(
        handoff,
        plan.sections,
        required_source_dois=required_source_dois,
    )
    previous_sections = {section.section_id: section for section in plan.sections}
    changed: dict[str, tuple[int, ...]] = {}
    for section in repaired_sections:
        previous = previous_sections.get(section.section_id)
        previous_paragraphs = previous.paragraphs if previous is not None else []
        changed_numbers = tuple(
            paragraph.paragraph_number
            for index, paragraph in enumerate(section.paragraphs)
            if index >= len(previous_paragraphs)
            or paragraph != previous_paragraphs[index]
        )
        if changed_numbers:
            changed[section.section_id] = changed_numbers
    fingerprint = _writing_plan_fingerprint(
        plan.topic,
        repaired_sections,
        required_source_dois=required_source_dois,
    )
    repaired_plan = GroundedWritingPlan(
        status=plan.status,
        topic=plan.topic,
        plan_fingerprint=fingerprint,
        required_source_dois=required_source_dois,
        sections=repaired_sections,
        confirmed_by=plan.confirmed_by,
        confirmed_at=plan.confirmed_at,
    )
    return WritingPlanCoverageRepair(
        plan=repaired_plan,
        changed_paragraph_numbers=changed,
    )


class ParagraphEvidencePacketBuilder:
    """Reduce a section packet to the authority locked for one paragraph."""

    def build(
        self,
        section_packet: SectionEvidencePacket,
        paragraph: WritingParagraphPlan,
    ) -> ParagraphEvidencePacket:
        if paragraph.section_id != section_packet.section_id:
            raise WritingPlanError("paragraph plan belongs to a different section")
        evidence = {item.evidence_id: item for item in section_packet.evidence_items}
        sources = {source.doi: source for source in section_packet.sources}
        missing_cards = [
            evidence_id
            for evidence_id in paragraph.evidence_card_ids
            if evidence_id not in evidence
        ]
        missing_sources = [doi for doi in paragraph.source_dois if doi not in sources]
        if missing_cards or missing_sources:
            raise WritingPlanError(
                "paragraph plan references authority outside its section packet"
            )
        return ParagraphEvidencePacket(
            section_id=section_packet.section_id,
            section_title=section_packet.title,
            paragraph=paragraph,
            counting_policy=section_packet.counting_policy,
            evidence_items=[evidence[item] for item in paragraph.evidence_card_ids],
            sources=[sources[doi] for doi in paragraph.source_dois],
        )


class LLMGroundedParagraphWriter:
    """Write prose from one locked paragraph packet without selecting citations."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def write(self, packet: ParagraphEvidencePacket) -> ParagraphTextProposal:
        schema = json.dumps(
            ParagraphTextProposal.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        maximum_units = _paragraph_maximum_units(packet)
        messages = [
            {
                "role": "system",
                "content": (
                    "Write exactly one scholarly paragraph from the locked evidence "
                    "packet. Return JSON only. The application already selected and "
                    "locked all evidence and sources; do not output IDs, DOI values, "
                    "references, or citation markers. Do not introduce claims, numbers, "
                    "methods, or papers outside the packet. Follow purpose and claim_focus "
                    f"and stay close to target_words={packet.paragraph.target_words}; the "
                    f"paragraph must not exceed {maximum_units} counted units under "
                    f"counting_policy={packet.counting_policy}. Metadata-only sources permit "
                    "only general statements. Encode line breaks and control characters "
                    "legally inside the JSON string. Substantively discuss every locked "
                    "source in the packet at the level allowed by its permitted_use; do not "
                    "merely list titles or authors. "
                    f"The response must satisfy this JSON Schema: {schema}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(packet.model_dump(mode="json"), ensure_ascii=False),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            raw = self._client.complete(
                messages,
                response_format={"type": "json_object"},
            )
            try:
                proposal = _parse_paragraph_text(raw)
                _ensure_paragraph_has_no_authored_citation(proposal)
                _ensure_paragraph_not_too_long(packet, proposal)
                return proposal
            except (
                ValidationError,
                ParagraphLengthError,
                ParagraphCitationError,
            ) as exc:
                last_error = exc
                if attempt == 0:
                    repair_instruction = _paragraph_repair_instruction(
                        exc,
                        maximum_units=maximum_units,
                    )
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                f"{repair_instruction} Do not add IDs, DOI values, references, "
                                "or citation markers. The previous output failed validation: "
                                f"{_short_error(exc)}"
                            ),
                        },
                    ]
                    continue
                if attempt == 1 and isinstance(exc, ParagraphLengthError):
                    safe_maximum = max(
                        packet.paragraph.target_words,
                        int(maximum_units * 0.82),
                    )
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "The previous compression is still too long. Rewrite the "
                                "same evidence-bound paragraph to no more than "
                                f"{safe_maximum} counted units, leaving safety margin below "
                                f"the hard limit of {maximum_units}. Preserve the central "
                                "comparison and every locked source at its permitted level. "
                                "Return exactly one text field without citations or IDs."
                            ),
                        },
                    ]
                    continue
                if attempt == 2 and isinstance(exc, ParagraphLengthError):
                    compacted = _compact_paragraph_to_limit(
                        proposal,
                        maximum_units=maximum_units,
                        counting_policy=packet.counting_policy,
                    )
                    _ensure_paragraph_has_no_authored_citation(compacted)
                    _ensure_paragraph_not_too_long(packet, compacted)
                    return compacted
                break
        raise GroundedWritingError(
            "LLM paragraph output violates the data contract after adaptive repair: "
            f"{_short_error(last_error)}"
        ) from last_error


class PlannedSectionDraftService:
    """Write a section paragraph by paragraph and reuse completed checkpoints."""

    def draft(
        self,
        section_packet: SectionEvidencePacket,
        section_plan: WritingSectionPlan,
        writer: LLMGroundedParagraphWriter,
        *,
        cache: ParagraphWritingRuntimeCache | None = None,
        existing_draft: SectionDraft | None = None,
        force: bool = False,
        force_paragraph_numbers: set[int] | None = None,
    ) -> SectionDraft:
        if section_plan.section_id != section_packet.section_id:
            raise WritingPlanError("section plan does not match the evidence packet")
        paragraph_proposals: list[DraftParagraphProposal] = []
        packet_builder = ParagraphEvidencePacketBuilder()
        forced_numbers = force_paragraph_numbers or set()
        for paragraph_plan in section_plan.paragraphs:
            paragraph_packet = packet_builder.build(section_packet, paragraph_plan)
            should_force = force or paragraph_plan.paragraph_number in forced_numbers
            text_proposal = None
            if not should_force and existing_draft is not None:
                text_proposal = _reuse_existing_paragraph(
                    existing_draft,
                    paragraph_packet,
                )
            if text_proposal is None and not should_force and cache is not None:
                text_proposal = cache.load(paragraph_packet)
            if text_proposal is None:
                text_proposal = writer.write(paragraph_packet)
                if cache:
                    cache.save(paragraph_packet, text_proposal)
            paragraph_proposals.append(
                DraftParagraphProposal(
                    role=paragraph_plan.role,
                    text=text_proposal.text,
                    evidence_card_ids=paragraph_plan.evidence_card_ids,
                    source_dois=paragraph_plan.source_dois,
                )
            )
        return GroundedSectionDraftService().create(
            section_packet,
            SectionDraftProposal(
                section_id=section_packet.section_id,
                paragraphs=paragraph_proposals,
            ),
        )


def _reuse_existing_paragraph(
    draft: SectionDraft,
    packet: ParagraphEvidencePacket,
) -> ParagraphTextProposal | None:
    """Reuse confirmed prose only when its locked support still matches the plan."""

    index = packet.paragraph.paragraph_number - 1
    if index < 0 or index >= len(draft.paragraphs):
        return None
    existing = draft.paragraphs[index]
    planned = packet.paragraph
    if (
        existing.role != planned.role
        or existing.evidence_card_ids != planned.evidence_card_ids
        or existing.source_dois != planned.source_dois
    ):
        return None
    proposal = ParagraphTextProposal(text=existing.text)
    try:
        _ensure_paragraph_has_no_authored_citation(proposal)
        _ensure_paragraph_not_too_long(packet, proposal)
    except (ParagraphCitationError, ParagraphLengthError):
        return None
    return proposal


class WritingPlanRuntimeCache:
    """Persist successful section plans so one failure does not discard earlier work."""

    def __init__(self, root: Path, *, handoff: V04WritingHandoff) -> None:
        self._root = root / _handoff_fingerprint(handoff)[:16]

    def load_section(self, packet: SectionEvidencePacket) -> WritingSectionPlan | None:
        path = self._root / f"{packet.section_id}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != _section_plan_signature(packet):
                return None
            return WritingSectionPlan.model_validate(payload["plan"])
        except Exception:
            return None

    def save_section(
        self,
        packet: SectionEvidencePacket,
        plan: WritingSectionPlan,
    ) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self._root / f"{packet.section_id}.json",
            json.dumps(
                {
                    "schema_version": "0.4-plan-cache.0",
                    "signature": _section_plan_signature(packet),
                    "plan": plan.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )


class ParagraphWritingRuntimeCache:
    """Persist each generated paragraph under its confirmed plan fingerprint."""

    def __init__(self, root: Path, *, plan_fingerprint: str) -> None:
        self._root = root / plan_fingerprint[:16]

    def load(self, packet: ParagraphEvidencePacket) -> ParagraphTextProposal | None:
        path = self._path(packet)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != _paragraph_signature(packet):
                return None
            proposal = ParagraphTextProposal.model_validate(payload["proposal"])
            _ensure_paragraph_not_too_long(packet, proposal)
            _ensure_paragraph_has_no_authored_citation(proposal)
            return proposal
        except Exception:
            return None

    def save(
        self,
        packet: ParagraphEvidencePacket,
        proposal: ParagraphTextProposal,
    ) -> None:
        _ensure_paragraph_not_too_long(packet, proposal)
        _ensure_paragraph_has_no_authored_citation(proposal)
        path = self._path(packet)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            path,
            json.dumps(
                {
                    "schema_version": "0.4-paragraph-cache.0",
                    "signature": _paragraph_signature(packet),
                    "proposal": proposal.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    def _path(self, packet: ParagraphEvidencePacket) -> Path:
        return (
            self._root
            / packet.section_id
            / f"{packet.paragraph.paragraph_id}.json"
        )


def _compile_section_plan(
    packet: SectionEvidencePacket,
    proposal: SectionPlanProposal,
    *,
    evidence_aliases: dict[str, SectionEvidenceItem],
    source_aliases: dict[str, SectionSourceRecord],
    expected_paragraph_count: int,
) -> WritingSectionPlan:
    if proposal.section_id != packet.section_id:
        raise WritingPlanError("planner changed the confirmed section_id")
    if len(proposal.paragraphs) != expected_paragraph_count:
        raise WritingPlanError(
            f"planner returned {len(proposal.paragraphs)} paragraphs; "
            f"expected {expected_paragraph_count}"
        )
    targets = _allocate_targets(
        [paragraph.relative_weight for paragraph in proposal.paragraphs],
        packet.target_words,
    )
    paragraphs = [
        _compile_paragraph(
            packet,
            proposal_item,
            number=number,
            target_words=targets[number - 1],
            evidence_aliases=evidence_aliases,
            source_aliases=source_aliases,
        )
        for number, proposal_item in enumerate(proposal.paragraphs, 1)
    ]
    return WritingSectionPlan(
        section_id=packet.section_id,
        title=packet.title,
        purpose=packet.purpose,
        target_words=packet.target_words,
        counting_policy=packet.counting_policy,
        paragraphs=paragraphs,
    )


def _compile_paragraph(
    packet: SectionEvidencePacket,
    proposal: ParagraphPlanProposal,
    *,
    number: int,
    target_words: int,
    evidence_aliases: dict[str, SectionEvidenceItem],
    source_aliases: dict[str, SectionSourceRecord],
) -> WritingParagraphPlan:
    unknown_evidence = [ref for ref in proposal.evidence_refs if ref not in evidence_aliases]
    unknown_sources = [ref for ref in proposal.source_refs if ref not in source_aliases]
    if unknown_evidence or unknown_sources:
        raise WritingPlanError(
            f"paragraph {number} used unknown short evidence/source aliases"
        )

    evidence_items = [evidence_aliases[ref] for ref in proposal.evidence_refs]
    selected_sources = [source_aliases[ref] for ref in proposal.source_refs]
    evidence_ids = [item.evidence_id for item in evidence_items]
    sources_by_doi = {source.doi: source for source in packet.sources}
    invalid_evidence_sources = [
        item.doi
        for item in evidence_items
        if not _permission_allows(
            sources_by_doi[item.doi].permitted_use,
            proposal.role,
        )
    ]
    if invalid_evidence_sources:
        raise WritingPlanError(
            f"paragraph {number} used evidence outside the permission for "
            f"{proposal.role}: {', '.join(invalid_evidence_sources)}"
        )

    permitted_sources = [
        source
        for source in selected_sources
        if _permission_allows(source.permitted_use, proposal.role)
    ]
    rejected_source_refs = [
        ref
        for ref, source in zip(proposal.source_refs, selected_sources, strict=True)
        if not _permission_allows(source.permitted_use, proposal.role)
    ]
    source_dois = list(
        dict.fromkeys(
            [
                *(item.doi for item in evidence_items),
                *(source.doi for source in permitted_sources),
            ]
        )
    )
    if not evidence_ids and not source_dois:
        rejected = ", ".join(rejected_source_refs)
        detail = (
            f" after removing disallowed refs {rejected}"
            if rejected_source_refs
            else ""
        )
        raise WritingPlanError(
            f"paragraph {number} has no permitted support for {proposal.role}{detail}"
        )
    if proposal.role == "detailed_evidence" and not evidence_ids:
        raise WritingPlanError(
            f"paragraph {number} is detailed_evidence but has no evidence card"
        )

    return WritingParagraphPlan(
        paragraph_id=f"{packet.section_id}_p{number:02d}",
        section_id=packet.section_id,
        paragraph_number=number,
        role=proposal.role,
        purpose=proposal.purpose,
        claim_focus=proposal.claim_focus,
        target_words=target_words,
        evidence_card_ids=evidence_ids,
        source_dois=source_dois,
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


def _allowed_roles(permitted_use: str) -> list[str]:
    return [
        role
        for role in (
            "detailed_evidence",
            "section_support",
            "background",
            "synthesis",
        )
        if _permission_allows(permitted_use, role)
    ]


def _paragraph_count(target_words: int) -> int:
    return max(3, min(10, round(target_words / 400)))


def _allocate_targets(weights: list[int], target_words: int) -> list[int]:
    minimum = 80
    if target_words < minimum * len(weights):
        raise WritingPlanError("section target is too small for planned paragraphs")
    distributable = target_words - minimum * len(weights)
    total_weight = sum(weights)
    raw = [distributable * weight / total_weight for weight in weights]
    targets = [minimum + int(value) for value in raw]
    remainder = target_words - sum(targets)
    order = sorted(
        range(len(weights)),
        key=lambda index: raw[index] - int(raw[index]),
        reverse=True,
    )
    for index in order[:remainder]:
        targets[index] += 1
    return targets


def _handoff_fingerprint(handoff: V04WritingHandoff) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-writing-plan-v1",
            "outline": handoff.outline.model_dump(mode="json"),
            "library_policy": handoff.evidence_library.requirement_policy_fingerprint,
            "evidence_cards": [
                card.model_dump(mode="json")
                for card in handoff.evidence_library.evidence_cards
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _section_plan_signature(packet: SectionEvidencePacket) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-writing-plan-v1",
            "packet": packet.model_dump(mode="json"),
            "paragraph_count": _paragraph_count(packet.target_words),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _writing_plan_fingerprint(
    topic: str,
    sections: list[WritingSectionPlan],
    *,
    required_source_dois: list[str],
) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-writing-plan-v1",
            "topic": topic,
            "required_source_dois": required_source_dois,
            "sections": [section.model_dump(mode="json") for section in sections],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _required_source_dois(handoff: V04WritingHandoff) -> list[str]:
    policy = handoff.requirement_policy or RequirementPolicyCompiler().compile(
        handoff.requirement
    )
    records = handoff.evidence_library.records
    required_count = (
        len(records)
        if policy.references.all_bibliography_items_must_be_cited_and_discussed
        else (
            policy.references.minimum_total
            if policy.references.target_is_approximate
            else policy.references.target_total
        )
    )
    if len(records) < required_count:
        raise WritingPlanError(
            "confirmed evidence library contains fewer sources than the required "
            f"writing coverage: required={required_count}; available={len(records)}"
        )
    return [record.doi for record in records[:required_count]]


def _apply_required_source_coverage(
    handoff: V04WritingHandoff,
    sections: list[WritingSectionPlan],
    *,
    required_source_dois: list[str],
) -> list[WritingSectionPlan]:
    if not required_source_dois:
        return sections
    policy = handoff.requirement_policy or RequirementPolicyCompiler().compile(
        handoff.requirement
    )
    cluster_limit = min(
        policy.references.max_references_per_citation_cluster or 4,
        8,
    )
    records = {record.doi: record for record in handoff.evidence_library.records}
    section_payloads = {
        section.section_id: section.model_dump(mode="json") for section in sections
    }
    section_order = [section.section_id for section in sections]
    covered = {
        doi
        for section in sections
        for paragraph in section.paragraphs
        for doi in paragraph.source_dois
    }
    missing_required = [doi for doi in required_source_dois if doi not in covered]
    if not missing_required:
        return sections
    original_paragraph_counts = {
        section.section_id: len(section.paragraphs) for section in sections
    }

    for doi in missing_required:
        record = records[doi]
        candidate_section_ids = [
            section_id
            for section_id in section_order
            if section_id in record.theme_ids
        ] or section_order
        placement = _place_source_in_existing_paragraph(
            section_payloads,
            candidate_section_ids,
            doi=doi,
            permitted_use=record.permitted_use,
            cluster_limit=cluster_limit,
        )
        if placement is None and record.permitted_use == "background_only":
            placement = _convert_support_paragraph_and_place(
                section_payloads,
                candidate_section_ids,
                doi=doi,
                cluster_limit=cluster_limit,
            )
        if placement is None:
            placement = _add_coverage_paragraph(
                section_payloads,
                candidate_section_ids[0],
                doi=doi,
                cluster_limit=cluster_limit,
            )
        covered.add(doi)

    rebuilt: list[WritingSectionPlan] = []
    for section_id in section_order:
        payload = section_payloads[section_id]
        paragraphs = payload["paragraphs"]
        targets = (
            _allocate_targets(
                [max(1, int(paragraph["target_words"])) for paragraph in paragraphs],
                int(payload["target_words"]),
            )
            if len(paragraphs) != original_paragraph_counts[section_id]
            else [int(paragraph["target_words"]) for paragraph in paragraphs]
        )
        for number, (paragraph, target) in enumerate(
            zip(paragraphs, targets, strict=True),
            1,
        ):
            paragraph["paragraph_id"] = f"{section_id}_p{number:02d}"
            paragraph["paragraph_number"] = number
            paragraph["target_words"] = target
        rebuilt.append(WritingSectionPlan.model_validate(payload))
    return rebuilt


def _place_source_in_existing_paragraph(
    section_payloads: dict[str, dict[str, object]],
    section_ids: list[str],
    *,
    doi: str,
    permitted_use: str,
    cluster_limit: int,
) -> tuple[str, int] | None:
    candidates: list[tuple[int, int, str, int]] = []
    for section_id in section_ids:
        paragraphs = section_payloads[section_id]["paragraphs"]
        for index, paragraph in enumerate(paragraphs):
            source_dois = paragraph["source_dois"]
            if (
                len(source_dois) < cluster_limit
                and _permission_allows(permitted_use, paragraph["role"])
            ):
                candidates.append((len(source_dois), index, section_id, index))
    if not candidates:
        return None
    _, _, section_id, index = min(candidates)
    paragraph = section_payloads[section_id]["paragraphs"][index]
    paragraph["source_dois"].append(doi)
    _mark_coverage_purpose(paragraph)
    return section_id, index


def _convert_support_paragraph_and_place(
    section_payloads: dict[str, dict[str, object]],
    section_ids: list[str],
    *,
    doi: str,
    cluster_limit: int,
) -> tuple[str, int] | None:
    candidates: list[tuple[int, int, str, int]] = []
    for section_id in section_ids:
        paragraphs = section_payloads[section_id]["paragraphs"]
        for index, paragraph in enumerate(paragraphs):
            if (
                paragraph["role"] == "section_support"
                and len(paragraph["source_dois"]) < cluster_limit
            ):
                candidates.append(
                    (len(paragraph["source_dois"]), index, section_id, index)
                )
    if not candidates:
        return None
    _, _, section_id, index = min(candidates)
    paragraph = section_payloads[section_id]["paragraphs"][index]
    paragraph["role"] = "background"
    paragraph["source_dois"].append(doi)
    _mark_coverage_purpose(paragraph)
    return section_id, index


def _add_coverage_paragraph(
    section_payloads: dict[str, dict[str, object]],
    section_id: str,
    *,
    doi: str,
    cluster_limit: int,
) -> tuple[str, int]:
    paragraphs = section_payloads[section_id]["paragraphs"]
    if len(paragraphs) >= 12:
        raise WritingPlanError(
            f"section {section_id} has no paragraph capacity for required source {doi}"
        )
    insertion = next(
        (
            index
            for index in range(len(paragraphs) - 1, -1, -1)
            if paragraphs[index]["role"] == "synthesis"
        ),
        len(paragraphs),
    )
    paragraphs.insert(
        insertion,
        {
            "paragraph_id": f"{section_id}_p99",
            "section_id": section_id,
            "paragraph_number": 99,
            "role": "background",
            "purpose": (
                "Map the scope of additional verified literature required by the "
                "bibliography coverage policy."
            ),
            "claim_focus": (
                "Compare the research scope of the assigned metadata-supported sources "
                "without attributing unverified detailed results."
            ),
            "target_words": 80,
            "evidence_card_ids": [],
            "source_dois": [doi][:cluster_limit],
        },
    )
    return section_id, insertion


def _mark_coverage_purpose(paragraph: dict[str, object]) -> None:
    marker = "Discuss and compare every additional locked source"
    purpose = str(paragraph["purpose"])
    if marker not in purpose:
        paragraph["purpose"] = (
            purpose.rstrip()
            + " Discuss and compare every additional locked source assigned by the "
            "required bibliography coverage policy."
        )


def _paragraph_signature(packet: ParagraphEvidencePacket) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-paragraph-writer-v1",
            "packet": packet.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_paragraph_text(raw: str) -> ParagraphTextProposal:
    try:
        return ParagraphTextProposal.model_validate_json(raw)
    except ValidationError as strict_error:
        try:
            payload = json.loads(raw, strict=False)
        except json.JSONDecodeError:
            raise strict_error
        return ParagraphTextProposal.model_validate(payload)


def _paragraph_maximum_units(packet: ParagraphEvidencePacket) -> int:
    return max(packet.paragraph.target_words, int(packet.paragraph.target_words * 1.6))


def _ensure_paragraph_not_too_long(
    packet: ParagraphEvidencePacket,
    proposal: ParagraphTextProposal,
) -> None:
    counted_units = count_writing_units(
        proposal.text,
        counting_policy=packet.counting_policy,
    )
    maximum_units = _paragraph_maximum_units(packet)
    if counted_units > maximum_units:
        raise ParagraphLengthError(
            f"paragraph has {counted_units} counted units; maximum is {maximum_units}"
        )


def _ensure_paragraph_has_no_authored_citation(
    proposal: ParagraphTextProposal,
) -> None:
    if re.search(
        r"\[@|https?://(?:dx\.)?doi\.org/|doi\s*:",
        proposal.text,
        re.I,
    ):
        raise ParagraphCitationError(
            "paragraph text attempted to author a citation or DOI marker"
        )


def _paragraph_repair_instruction(
    exc: Exception,
    *,
    maximum_units: int,
) -> str:
    if isinstance(exc, ParagraphLengthError):
        return (
            "Rewrite only the paragraph text and shorten it to no more than "
            f"{maximum_units} counted units while preserving its evidence-bound meaning. "
            "Return exactly one text field."
        )
    if isinstance(exc, ParagraphCitationError):
        return (
            "Remove every citation marker, DOI value, DOI URL, and reference label from "
            "the paragraph text while preserving the evidence-bound prose. Return exactly "
            "one text field."
        )
    return (
        "Repair only the JSON encoding and schema. Preserve the paragraph meaning, return "
        "exactly one text field, and escape all line breaks or control characters."
    )


def _compact_paragraph_to_limit(
    proposal: ParagraphTextProposal,
    *,
    maximum_units: int,
    counting_policy: str,
) -> ParagraphTextProposal:
    """Keep complete leading sentences, then safely clip only as a last resort."""

    sentences = re.findall(
        r".+?(?:[。！？]|[.!?](?=\s|$)|$)",
        proposal.text,
    )
    accepted: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*accepted, sentence.strip()]).strip()
        if count_writing_units(candidate, counting_policy=counting_policy) > maximum_units:
            break
        accepted.append(sentence.strip())
    if accepted:
        compacted_text = " ".join(accepted).strip()
    else:
        compacted_text = _maximum_prefix_within_limit(
            proposal.text,
            maximum_units=maximum_units,
            counting_policy=counting_policy,
        )
    if not compacted_text:
        raise ParagraphLengthError(
            "paragraph could not be compacted without becoming blank"
        )
    return ParagraphTextProposal(text=compacted_text)


def _maximum_prefix_within_limit(
    text: str,
    *,
    maximum_units: int,
    counting_policy: str,
) -> str:
    low = 0
    high = len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        if (
            count_writing_units(
                text[:midpoint],
                counting_policy=counting_policy,
            )
            <= maximum_units
        ):
            low = midpoint
        else:
            high = midpoint - 1
    prefix = text[:low].rstrip()
    if (
        prefix
        and re.fullmatch(r"[A-Za-z0-9_]", prefix[-1])
        and low < len(text)
        and re.fullmatch(r"[A-Za-z0-9_]", text[low])
    ):
        prefix = re.sub(r"[A-Za-z0-9_'-]+$", "", prefix).rstrip()
    prefix = prefix.rstrip("，,;:；：")
    if not prefix or prefix.endswith((".", "!", "?", "。", "！", "？")):
        return prefix
    ending = "。" if re.search(r"[一-鿿]", prefix) else "."
    return prefix + ending


def _short_error(exc: Exception | None) -> str:
    if exc is None:
        return "unknown planning error"
    if isinstance(exc, ValidationError):
        return str(
            [
                {
                    "location": ".".join(str(item) for item in error.get("loc", ())),
                    "message": error.get("msg", "validation failed"),
                }
                for error in exc.errors(include_url=False)[:6]
            ]
        )
    return str(exc)[:1200]


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
