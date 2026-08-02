"""Evidence-first V0.4 planning, paragraph writing, and resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
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
)


class WritingPlanError(ValueError):
    """Raised when semantic planning cannot compile to real evidence authority."""


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

        fingerprint = _writing_plan_fingerprint(
            handoff.outline.outline.topic,
            section_plans,
        )
        return GroundedWritingPlan(
            topic=handoff.outline.outline.topic,
            plan_fingerprint=fingerprint,
            sections=section_plans,
        )

    def _plan_section(self, packet: SectionEvidencePacket) -> WritingSectionPlan:
        paragraph_count = _paragraph_count(packet.target_words)
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
                    "abstract": (source.abstract or "")[:400],
                }
                for alias, source in source_aliases.items()
            ],
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
                    "E### and S### aliases supplied by the application. detailed_evidence "
                    "requires one to five evidence_refs. Metadata-only or background-only "
                    "sources may support only general section_support, background, or "
                    "synthesis claims; never use them for specific results, methods, or "
                    "numbers. Every paragraph needs at least one evidence_ref or source_ref. "
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
                                f"plan failed deterministic validation: {_short_error(exc)}"
                            ),
                        },
                    ]
        raise WritingPlanError(
            f"section planning failed after one repair: {_short_error(last_error)}"
        ) from last_error


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
        raw = self._client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Write exactly one scholarly paragraph from the locked evidence "
                        "packet. Return JSON only. The application already selected and "
                        "locked all evidence and sources; do not output IDs, DOI values, "
                        "references, or citation markers. Do not introduce claims, numbers, "
                        "methods, or papers outside the packet. Follow purpose and claim_focus "
                        "and stay close to target_words. Metadata-only sources permit only "
                        "general statements. "
                        f"The response must satisfy this JSON Schema: {schema}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(packet.model_dump(mode="json"), ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
        )
        try:
            return ParagraphTextProposal.model_validate_json(raw)
        except ValidationError as exc:
            raise GroundedWritingError(
                f"LLM paragraph output violates the data contract: {_short_error(exc)}"
            ) from exc


class PlannedSectionDraftService:
    """Write a section paragraph by paragraph and reuse completed checkpoints."""

    def draft(
        self,
        section_packet: SectionEvidencePacket,
        section_plan: WritingSectionPlan,
        writer: LLMGroundedParagraphWriter,
        *,
        cache: ParagraphWritingRuntimeCache | None = None,
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
            text_proposal = (
                None
                if should_force or cache is None
                else cache.load(paragraph_packet)
            )
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
            return ParagraphTextProposal.model_validate(payload["proposal"])
        except Exception:
            return None

    def save(
        self,
        packet: ParagraphEvidencePacket,
        proposal: ParagraphTextProposal,
    ) -> None:
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
    source_dois = list(
        dict.fromkeys(
            [
                *(item.doi for item in evidence_items),
                *(source.doi for source in selected_sources),
            ]
        )
    )
    if not evidence_ids and not source_dois:
        raise WritingPlanError(f"paragraph {number} has no planned support")
    if proposal.role == "detailed_evidence" and not evidence_ids:
        raise WritingPlanError(
            f"paragraph {number} is detailed_evidence but has no evidence card"
        )

    sources_by_doi = {source.doi: source for source in packet.sources}
    for doi in source_dois:
        source = sources_by_doi[doi]
        if not _permission_allows(source.permitted_use, proposal.role):
            raise WritingPlanError(
                f"paragraph {number} exceeds source permission for {doi}"
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
) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-writing-plan-v1",
            "topic": topic,
            "sections": [section.model_dump(mode="json") for section in sections],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
