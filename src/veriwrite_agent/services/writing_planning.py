"""Evidence-first V0.4 planning, paragraph writing, and resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient, LLMResponseError
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
from veriwrite_agent.services.writing_quality import (
    false_self_attribution_detail,
    language_mismatch_detail,
    output_language_instruction,
)


class WritingPlanError(ValueError):
    """Raised when semantic planning cannot compile to real evidence authority."""


class WritingPlanBudgetExceeded(WritingPlanError):
    """Raised before another model call would exceed the bounded planning budget."""


class WritingPlanDependencyError(WritingPlanError):
    """Raised when deterministic evidence permissions make retrying prose useless."""


class WritingPausedError(RuntimeError):
    """Raised between model calls after a durable pause request."""


class ParagraphLengthError(ValueError):
    """Raised when one model paragraph greatly exceeds its locked word budget."""


class ParagraphTooShortError(ParagraphLengthError):
    """Raised when a normal body paragraph materially misses its planned budget."""


class ParagraphCitationError(ValueError):
    """Raised when paragraph prose attempts to create its own citation."""


class ParagraphLanguageError(ValueError):
    """Raised when prose violates the confirmed output language."""


class ParagraphGenreAttributionError(ValueError):
    """Raised when review prose claims a cited study as the current paper's work."""


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
        repair_feedback_by_section: dict[str, list[str]] | None = None,
        max_elapsed_seconds: float = 300.0,
        max_model_calls: int = 6,
    ) -> None:
        if max_elapsed_seconds <= 0:
            raise ValueError("max_elapsed_seconds must be positive")
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        self._client = client
        self._cache = cache
        self._reuse_cache = reuse_cache
        self._repair_feedback_by_section = repair_feedback_by_section or {}
        self._max_elapsed_seconds = max_elapsed_seconds
        self._max_model_calls = max_model_calls

    def plan(self, handoff: V04WritingHandoff) -> GroundedWritingPlan:
        started = perf_counter()
        model_calls = 0
        policy = handoff.requirement_policy or RequirementPolicyCompiler().compile(
            handoff.requirement
        )
        required_source_dois = _required_source_dois(handoff)
        section_plans: list[WritingSectionPlan] = []
        for outline_section in handoff.outline.outline.sections:
            packet = SectionEvidencePacketBuilder().build(
                handoff,
                outline_section.section_id,
                include_policy_required_routes=False,
            )
            packet_source_dois = {source.doi for source in packet.sources}
            packet = packet.model_copy(
                update={
                    "required_source_dois": [
                        doi
                        for doi in required_source_dois
                        if doi in packet_source_dois
                    ]
                }
            )
            cached = (
                self._cache.load_section(packet)
                if self._cache and self._reuse_cache
                else None
            )
            if cached is None:
                if (
                    model_calls >= self._max_model_calls
                    or perf_counter() - started >= self._max_elapsed_seconds
                ):
                    raise WritingPlanBudgetExceeded(
                        "writing planning stopped before exceeding its model-call or "
                        "wall-clock budget"
                    )
                model_calls += 1
                cached = self._plan_section(packet)
                if self._cache:
                    self._cache.save_section(packet, cached)
            section_plans.append(cached)

        section_plans = _apply_required_source_coverage(
            handoff,
            section_plans,
            required_source_dois=required_source_dois,
        )
        fingerprint = _writing_plan_fingerprint(
            handoff.outline.outline.topic,
            section_plans,
            required_source_dois=required_source_dois,
            output_language=policy.output_language,
        )
        return GroundedWritingPlan(
            topic=handoff.outline.outline.topic,
            output_language=policy.output_language,
            plan_fingerprint=fingerprint,
            required_source_dois=required_source_dois,
            sections=section_plans,
        )

    def replan_section(
        self,
        packet: SectionEvidencePacket,
        *,
        paragraph_count: int,
    ) -> WritingSectionPlan:
        """Semantically replan one structurally edited section under the same contracts."""

        if not 2 <= paragraph_count <= 12:
            raise WritingPlanError("replanned paragraph count must be between 2 and 12")
        return self._plan_section(
            packet,
            paragraph_count_override=paragraph_count,
        )

    def _plan_section(
        self,
        packet: SectionEvidencePacket,
        *,
        paragraph_count_override: int | None = None,
    ) -> WritingSectionPlan:
        paragraph_count = paragraph_count_override or _paragraph_count(
            packet.target_words
        )
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
                "max_sources_per_paragraph": packet.max_sources_per_paragraph,
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
                    "centrality": source.centrality,
                    "supported_claim": source.supported_claim,
                    "suitable_section_id": source.suitable_section_id,
                    "use_boundary": source.use_boundary,
                    "required_for_reference_policy": (
                        source.doi in packet.required_source_dois
                    ),
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
            "repair_feedback": self._repair_feedback_by_section.get(
                packet.section_id,
                [],
            ),
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
                    "detailed_evidence requires one to five evidence_refs. Never attach "
                    "more than five evidence_refs or more than eight source_refs to any "
                    "single paragraph. A source whose "
                    "permitted_use is background_only may support only background or "
                    "synthesis, never section_support or detailed_evidence. Every paragraph "
                    "needs at least one permitted evidence_ref or source_ref. "
                    "Select only the sources necessary for the paragraph's central claim, "
                    "never exceeding section.max_sources_per_paragraph. "
                    "Do not attach an evidence_ref unless its normalized_claim directly "
                    "supports that paragraph's claim_focus. Background and synthesis "
                    "paragraphs may rely on source_refs alone. For a source whose "
                    "permitted_use is background_only, claim_focus may paraphrase only its "
                    "supported_claim: do not plan instrument specifications, performance "
                    "numbers, methods, datasets, comparisons, or conclusions that are not "
                    "stated there. Unused bibliography items are handled separately by "
                    "the admission and planning stages. Every source marked "
                    "required_for_reference_policy must appear in at least one paragraph "
                    "where it supports that paragraph's central claim. Compare sources by "
                    "problem, method, evidence, difference, or limitation; never create a "
                    "bibliography-coverage paragraph or mention internal coverage policy. "
                    "Respect each source's supported_claim and use_boundary; never expand "
                    "a contextual or supporting source into the paragraph's main subject. "
                    "Keep claim_focus narrow enough for one paragraph. Every paragraph "
                    "must start from a research problem and a central judgment, not from "
                    "a paper. Set central_question and argument_move. Use compare_studies, "
                    "synthesize_consensus, or analyze_difference for multi-study synthesis "
                    "and state the comparison_axis. A single-paper paragraph is allowed "
                    "only when one representative study is genuinely needed as detailed "
                    "evidence; do not create a sequence of paper summaries. relative_weight is "
                    "an integer from 1 to 10; code assigns exact word budgets. "
                    "When repair_feedback is non-empty, treat it as a mandatory diagnosis "
                    "from an independent reviewer. Replace the defective paragraph intent: "
                    "narrow topic-drifted claims to this section's purpose, and weaken or "
                    "remove unsupported claims so they state only what the supplied evidence "
                    "and source use boundaries permit. Rebuild the complete paragraph-role "
                    "map so every surviving paragraph has one distinct question and no two "
                    "paragraphs summarize the same studies or conclusion. Assign sources by "
                    "semantic support, not merely to fill capacity. Do not merely paraphrase "
                    "the rejected claim. The opening paragraph establishes only the section "
                    "scope and organizing problem; the closing paragraph states limitations "
                    "and a section-level judgment. Neither may repeat the middle paragraphs' "
                    "study details. "
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
            try:
                raw = self._client.complete(
                    messages,
                    response_format={"type": "json_object"},
                )
            except LLMResponseError as exc:
                # A transient provider failure (empty or truncated output) must not
                # abort planning. Retry once with a tightening instruction before
                # reporting a planning contract failure.
                last_error = exc
                if attempt == 0:
                    messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                "The provider returned empty or truncated output. Return "
                                "the complete plan JSON only, using the short E### and "
                                "S### aliases, compressing rather than truncating so it "
                                "fits within the configured output limit."
                            ),
                        },
                    ]
                    continue
                break
            try:
                proposal = SectionPlanProposal.model_validate_json(raw)
                if attempt > 0 and len(proposal.paragraphs) > paragraph_count:
                    proposal = _trim_excess_paragraph_plans(
                        proposal,
                        expected_count=paragraph_count,
                    )
                proposal = _assign_required_sources_to_problem_paragraphs(
                    packet,
                    proposal,
                    evidence_aliases=evidence_aliases,
                    source_aliases=source_aliases,
                    repair_invalid_permissions=True,
                )
                proposal = _drop_semantically_misaligned_optional_evidence(
                    proposal,
                    evidence_aliases=evidence_aliases,
                    source_aliases=source_aliases,
                )
                # Removing an unrelated optional evidence card can also remove the
                # only occurrence of that source from the coverage table. Re-run the
                # deterministic allocator so the DOI is placed as a bounded source
                # reference in a semantically closer paragraph instead of restoring
                # the misleading evidence card.
                proposal = _assign_required_sources_to_problem_paragraphs(
                    packet,
                    proposal,
                    evidence_aliases=evidence_aliases,
                    source_aliases=source_aliases,
                    repair_invalid_permissions=True,
                )
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
                                "Correct exactly the reported problem and re-emit the full "
                                "plan, still satisfying every constraint in the system "
                                "prompt (aliases, role permissions, ref-count limits, and "
                                "metadata-only paraphrasing)."
                            ),
                        },
                    ]
        raise WritingPlanError(
            f"section planning failed after one repair: {_short_error(last_error)}"
        ) from last_error


def _trim_excess_paragraph_plans(
    proposal: SectionPlanProposal,
    *,
    expected_count: int,
) -> SectionPlanProposal:
    """Honor a structural paragraph count after the model repeats an over-count.

    The first mismatch is returned to the semantic planner. If its repaired JSON still
    contains too many paragraphs, code removes the lowest-value interior plan. Required
    source coverage is deliberately not migrated here: the ordinary deterministic
    allocator that runs immediately afterwards redistributes every hard-required DOI
    under the final roles and citation capacities.
    """

    paragraphs = list(proposal.paragraphs)
    while len(paragraphs) > expected_count:
        interior = range(1, len(paragraphs) - 1)
        candidates = list(interior) or list(range(len(paragraphs)))
        chosen = min(
            candidates,
            key=lambda index: (
                bool(paragraphs[index].evidence_refs),
                len(paragraphs[index].evidence_refs)
                + len(paragraphs[index].source_refs),
                paragraphs[index].relative_weight,
                -index,
            ),
        )
        paragraphs.pop(chosen)
    return proposal.model_copy(update={"paragraphs": paragraphs})


def repair_writing_plan_source_coverage(
    handoff: V04WritingHandoff,
    plan: GroundedWritingPlan,
    *,
    required_source_dois: list[str] | None = None,
) -> WritingPlanCoverageRepair:
    """Validate coverage without manufacturing bibliography-policy prose."""

    required_source_dois = (
        list(dict.fromkeys(required_source_dois))
        if required_source_dois is not None
        else _required_source_dois(handoff)
    )
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
            or _paragraph_requires_rewrite(previous_paragraphs[index], paragraph)
        )
        if changed_numbers:
            changed[section.section_id] = changed_numbers
    fingerprint = _writing_plan_fingerprint(
        plan.topic,
        repaired_sections,
        required_source_dois=required_source_dois,
        output_language=plan.output_language,
    )
    repaired_plan = GroundedWritingPlan(
        status=plan.status,
        topic=plan.topic,
        output_language=plan.output_language,
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


def rebase_writing_plan_authority(
    handoff: V04WritingHandoff,
    plan: GroundedWritingPlan,
) -> WritingPlanCoverageRepair:
    """Rebind a cached plan to the exact authority in a newer V0.3 handoff.

    Evidence recovery can replace the admitted literature set while a compatible
    writing plan and accepted chapters remain cached.  A plan must never reach the
    paragraph writer with DOI values or evidence cards that the current handoff no
    longer grants.  Metadata-only sources that have left the library may be removed
    when the paragraph still has other support; missing evidence cards or a paragraph
    left with no authority are dependency failures and require replanning instead.
    """

    packet_builder = SectionEvidencePacketBuilder()
    previous_sections = {section.section_id: section for section in plan.sections}
    rebased_sections: list[WritingSectionPlan] = []
    for section in plan.sections:
        packet = packet_builder.build(handoff, section.section_id)
        available_evidence = {
            item.evidence_id: item for item in packet.evidence_items
        }
        available_sources = {source.doi for source in packet.sources}
        paragraphs: list[WritingParagraphPlan] = []
        for paragraph in section.paragraphs:
            missing_evidence = [
                evidence_id
                for evidence_id in paragraph.evidence_card_ids
                if evidence_id not in available_evidence
            ]
            if missing_evidence:
                raise WritingPlanDependencyError(
                    f"{section.section_id} paragraph {paragraph.paragraph_number} "
                    "references evidence cards no longer available in the current "
                    f"handoff: {', '.join(missing_evidence)}"
                )
            source_dois = [
                doi for doi in paragraph.source_dois if doi in available_sources
            ]
            evidence_dois = [
                available_evidence[evidence_id].doi
                for evidence_id in paragraph.evidence_card_ids
            ]
            source_dois = list(dict.fromkeys([*evidence_dois, *source_dois]))
            if not source_dois:
                raise WritingPlanDependencyError(
                    f"{section.section_id} paragraph {paragraph.paragraph_number} "
                    "has no surviving authority after the V0.3 handoff changed"
                )
            update: dict[str, object] = {"source_dois": source_dois}
            if (
                paragraph.argument_move
                in {"compare_studies", "synthesize_consensus", "analyze_difference"}
                and len(source_dois) < 2
            ):
                update.update(
                    {
                        "argument_move": (
                            "evaluate_limitation"
                            if paragraph.role
                            in {"detailed_evidence", "section_support"}
                            else "frame_problem"
                        ),
                        "comparison_axis": None,
                    }
                )
            paragraphs.append(paragraph.model_copy(update=update))
        rebased_sections.append(section.model_copy(update={"paragraphs": paragraphs}))

    required_source_dois = _required_source_dois(handoff)
    rebased_sections = _apply_required_source_coverage(
        handoff,
        rebased_sections,
        required_source_dois=required_source_dois,
    )
    changed: dict[str, tuple[int, ...]] = {}
    for section in rebased_sections:
        previous = previous_sections[section.section_id]
        changed_numbers = tuple(
            current.paragraph_number
            for old, current in zip(
                previous.paragraphs,
                section.paragraphs,
                strict=True,
            )
            if _paragraph_requires_rewrite(old, current)
        )
        if changed_numbers:
            changed[section.section_id] = changed_numbers
    fingerprint = _writing_plan_fingerprint(
        plan.topic,
        rebased_sections,
        required_source_dois=required_source_dois,
        output_language=plan.output_language,
    )
    rebased_plan = GroundedWritingPlan.model_validate(
        plan.model_copy(
            update={
                "plan_fingerprint": fingerprint,
                "required_source_dois": required_source_dois,
                "sections": rebased_sections,
            }
        ).model_dump(mode="json")
    )
    return WritingPlanCoverageRepair(
        plan=rebased_plan,
        changed_paragraph_numbers=changed,
    )


def align_writing_plan_language(
    handoff: V04WritingHandoff,
    plan: GroundedWritingPlan,
) -> GroundedWritingPlan:
    """Migrate cached plans to language and dedicated-coverage contracts."""

    policy = handoff.requirement_policy or RequirementPolicyCompiler().compile(
        handoff.requirement
    )
    section_payloads = [section.model_dump(mode="json") for section in plan.sections]
    coverage_changed = False
    for section in section_payloads:
        for paragraph in section["paragraphs"]:
            purpose = str(paragraph["purpose"]).casefold()
            claim_focus = str(paragraph["claim_focus"]).casefold()
            legacy_coverage = (
                purpose.startswith(
                    "map the scope of additional verified literature required by"
                )
                or "assigned metadata-supported sources" in claim_focus
            )
            if paragraph.get("coverage_only", False) != legacy_coverage:
                paragraph["coverage_only"] = legacy_coverage
                coverage_changed = True
    if plan.output_language == policy.output_language and not coverage_changed:
        return plan
    sections = [WritingSectionPlan.model_validate(section) for section in section_payloads]
    fingerprint = _writing_plan_fingerprint(
        plan.topic,
        sections,
        required_source_dois=plan.required_source_dois,
        output_language=policy.output_language,
    )
    payload = plan.model_dump(mode="json")
    payload["output_language"] = policy.output_language
    payload["sections"] = [section.model_dump(mode="json") for section in sections]
    payload["plan_fingerprint"] = fingerprint
    return GroundedWritingPlan.model_validate(payload)


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
            output_language=section_packet.output_language,
            evidence_items=[evidence[item] for item in paragraph.evidence_card_ids],
            sources=[sources[doi] for doi in paragraph.source_dois],
        )


class LLMGroundedParagraphWriter:
    """Write prose from one locked paragraph packet without selecting citations."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def write(
        self,
        packet: ParagraphEvidencePacket,
        *,
        revision_instruction: str | None = None,
        editorial_context: list[dict[str, object]] | None = None,
    ) -> ParagraphTextProposal:
        if packet.paragraph.coverage_only:
            raise WritingPlanError(
                "legacy bibliography-coverage paragraph cannot be written; "
                "regenerate the section's problem-driven plan"
            )
        schema = json.dumps(
            ParagraphTextProposal.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        maximum_units = _paragraph_maximum_units(packet)
        minimum_units = _paragraph_minimum_units(packet)
        length_instruction = (
            f"at least {minimum_units} and no more than {maximum_units}"
            if minimum_units
            else f"no more than {maximum_units}"
        )
        source_instruction = (
            "The plan has already narrowed the support set to the sources necessary "
            "for this central claim. Build one coherent argument around claim_focus; "
            "synthesize the sources and do not turn the paragraph into a source list."
        )
        revision_text = (
            " Editorial revision requirement (this overrides old wording and any "
            "conflicting legacy claim-focus detail): "
            f"{revision_instruction} Replace the old paragraph; do not merely paraphrase "
            "it. For repetition, role overlap, or excessive size, keep only one unique "
            "argument move and omit repeated cases. The output contract still requires "
            "exactly one paragraph, so turn any request to delete, merge, move, or split "
            "material into one concise evidence-bounded replacement paragraph."
            if revision_instruction
            else ""
        )
        context_instruction = (
            " The user payload also contains read-only editorial_context from the "
            "existing chapter. Use it only to avoid repetition, preserve the chapter's "
            "argument progression, and keep this paragraph in its assigned role. It is "
            "not evidence and must never authorize a new fact, number, source, or claim."
            if editorial_context
            else ""
        )
        user_payload: object = packet.model_dump(mode="json")
        if editorial_context:
            user_payload = {
                "locked_evidence_packet": packet.model_dump(mode="json"),
                "editorial_context": editorial_context,
            }
        messages = [
            {
                "role": "system",
                "content": (
                    "Write exactly one scholarly paragraph from the locked evidence "
                    "packet. Return JSON only. The application already selected and "
                    "locked all evidence and sources; do not output IDs, DOI values, "
                    "references, or citation markers. Do not introduce claims, numbers, "
                    "methods, or papers outside the packet. Follow purpose and claim_focus "
                    "and answer paragraph.central_question through the declared "
                    "argument_move. Lead with the paragraph's central judgment, then "
                    "compare or synthesize evidence along comparison_axis when supplied. "
                    "Do not organize the paragraph as one-paper-at-a-time notes. "
                    "This paper is a literature review, not an original empirical study. "
                    "Never write that 本文、本研究、本论文 or 我们 proposed, designed, "
                    "used, measured, retrieved, validated, discovered, or obtained a cited "
                    "method, dataset, experiment, or result. When reporting source-specific "
                    "work, use the author names supplied in packet.sources as the grammatical "
                    "subject (for example, 'Yang 等针对 EMI-02……'). The current review may "
                    "only claim its own scope, organization, comparison, synthesis, and "
                    "cautious judgment. Avoid abstract-like summaries and perform only this "
                    "paragraph's declared argument move. "
                    "For a global editorial repair, brevity and a unique rhetorical role "
                    "take priority over preserving the previous paragraph's coverage or "
                    "length; discuss only sources that remain in the refined packet. Never "
                    "mention the revision process or write meta-prose such as 'this paragraph "
                    "no longer repeats the preceding text', 'according to the review', or "
                    "'as a transition'. The replacement must read like ordinary submit-ready "
                    "scholarly prose. "
                    f"Stay close to target_words={packet.paragraph.target_words}; the "
                    f"paragraph must contain {length_instruction} counted units under "
                    f"counting_policy={packet.counting_policy}. A metadata-only/background-"
                    "only source authorizes only a cautious paraphrase of its supplied "
                    "supported_claim and use_boundary. It does NOT authorize any number, "
                    "instrument specification, channel count, accuracy, named method detail, "
                    "dataset property, experiment result, or comparison absent from those "
                    "fields—even if you know the paper. When detail is unavailable, deepen "
                    "the paragraph through bounded comparison and limitations rather than "
                    "inventing specifics. Encode line breaks and control characters "
                    "legally inside the JSON string. "
                    f"{output_language_instruction(packet.output_language)} "
                    f"{source_instruction}{revision_text}{context_instruction} "
                    f"The response must satisfy this JSON Schema: {schema}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                raw = self._client.complete(
                    messages,
                    response_format={"type": "json_object"},
                )
            except LLMResponseError as exc:
                # A transient provider failure (empty or truncated output) must not
                # abort the whole chapter. Retry once with a tightening instruction
                # before reporting a contract violation.
                last_error = exc
                if attempt == 0:
                    messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                "The provider returned empty or truncated output. Return "
                                "the complete JSON object for exactly one paragraph, "
                                "compressing rather than truncating so it fits within the "
                                "configured output limit."
                            ),
                        },
                    ]
                    continue
                break
            try:
                proposal = _parse_paragraph_text(raw)
                _ensure_paragraph_has_no_authored_citation(proposal)
                _ensure_review_genre_attribution(proposal)
                _ensure_paragraph_not_too_long(packet, proposal)
                _ensure_paragraph_not_too_short(packet, proposal)
                _ensure_paragraph_language(packet, proposal)
                return proposal
            except (
                ValidationError,
                ParagraphLengthError,
                ParagraphCitationError,
                ParagraphGenreAttributionError,
                ParagraphLanguageError,
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
                                f"{_short_error(exc)} "
                                f"{output_language_instruction(packet.output_language)}"
                            ),
                        },
                    ]
                    continue
                if attempt == 1 and isinstance(exc, ParagraphTooShortError):
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "The paragraph is still materially under its writing "
                                f"budget. Expand the same argument to {minimum_units}-"
                                f"{maximum_units} counted units by adding deeper comparison, "
                                "synthesis, limitations, and cautious implications already "
                                "authorized by the locked evidence. Do not add sources or "
                                "repeat neighbouring paragraphs. Return one text field."
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
                                "comparison and the exact planned support scope. "
                                f"{output_language_instruction(packet.output_language)} "
                                "Return exactly one text field without citations or IDs."
                            ),
                        },
                    ]
                    continue
                if attempt == 2 and isinstance(exc, ParagraphLengthError):
                    if isinstance(exc, ParagraphTooShortError):
                        break
                    compacted = _compact_paragraph_to_limit(
                        proposal,
                        maximum_units=maximum_units,
                        counting_policy=packet.counting_policy,
                    )
                    _ensure_paragraph_has_no_authored_citation(compacted)
                    _ensure_review_genre_attribution(compacted)
                    _ensure_paragraph_not_too_long(packet, compacted)
                    _ensure_paragraph_language(packet, compacted)
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
        revision_instructions: dict[int, str] | None = None,
        on_paragraph_progress: Callable[[int, int, str], None] | None = None,
        max_workers: int = 1,
        should_continue: Callable[[], bool] | None = None,
    ) -> SectionDraft:
        if section_plan.section_id != section_packet.section_id:
            raise WritingPlanError("section plan does not match the evidence packet")
        packet_builder = ParagraphEvidencePacketBuilder()
        forced_numbers = force_paragraph_numbers or set()

        def _write_paragraph(
            paragraph_plan: WritingParagraphPlan,
        ) -> tuple[int, DraftParagraphProposal, str]:
            paragraph_packet = packet_builder.build(section_packet, paragraph_plan)
            should_force = force or paragraph_plan.paragraph_number in forced_numbers
            text_proposal = None
            source = "generated"
            if not should_force and existing_draft is not None:
                text_proposal = _reuse_existing_paragraph(
                    existing_draft,
                    paragraph_packet,
                )
                if text_proposal is not None:
                    source = "draft"
            if text_proposal is None and not should_force and cache is not None:
                text_proposal = cache.load(paragraph_packet)
                if text_proposal is not None:
                    source = "cache"
            if text_proposal is None:
                editorial_context = None
                if should_force and existing_draft is not None:
                    editorial_context = [
                        {
                            "paragraph_number": number,
                            "role": paragraph.role,
                            "text": paragraph.text,
                        }
                        for number, paragraph in enumerate(
                            existing_draft.paragraphs,
                            1,
                        )
                        if number != paragraph_plan.paragraph_number
                    ]
                text_proposal = writer.write(
                    paragraph_packet,
                    revision_instruction=(revision_instructions or {}).get(
                        paragraph_plan.paragraph_number
                    ),
                    editorial_context=editorial_context,
                )
                if cache:
                    cache.save(paragraph_packet, text_proposal)
            proposal = DraftParagraphProposal(
                role=paragraph_plan.role,
                text=text_proposal.text,
                evidence_card_ids=paragraph_plan.evidence_card_ids,
                source_dois=paragraph_plan.source_dois,
            )
            return paragraph_plan.paragraph_number, proposal, source

        # Initial drafts are independent once the evidence is locked, so they can be
        # produced concurrently. Targeted revisions rely on ``editorial_context`` from
        # the surrounding paragraphs and therefore stay serial to preserve argument
        # progression.
        total = len(section_plan.paragraphs)
        parallel = max_workers > 1 and not force and not forced_numbers
        if parallel:
            results_by_number: dict[int, tuple[DraftParagraphProposal, str]] = {}
            executor = ThreadPoolExecutor(max_workers=max_workers)
            pending: dict[Future[tuple[int, DraftParagraphProposal, str]], int] = {}
            plans = iter(section_plan.paragraphs)

            def _submit_next() -> bool:
                if should_continue is not None and not should_continue():
                    return False
                try:
                    next_plan = next(plans)
                except StopIteration:
                    return False
                pending[executor.submit(_write_paragraph, next_plan)] = (
                    next_plan.paragraph_number
                )
                return True

            try:
                for _ in range(min(max_workers, total)):
                    if not _submit_next():
                        break
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        number, proposal, source = future.result()
                        results_by_number[number] = (proposal, source)
                        if on_paragraph_progress is not None:
                            on_paragraph_progress(number, total, source)
                        _submit_next()
                if len(results_by_number) != total:
                    raise WritingPausedError(
                        "paragraph generation paused before the next task was submitted"
                    )
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            paragraph_proposals = [
                results_by_number[paragraph_plan.paragraph_number][0]
                for paragraph_plan in section_plan.paragraphs
            ]
        else:
            paragraph_proposals = []
            for paragraph_plan in section_plan.paragraphs:
                if should_continue is not None and not should_continue():
                    raise WritingPausedError(
                        "paragraph generation paused before the next task was submitted"
                    )
                number, proposal, source = _write_paragraph(paragraph_plan)
                paragraph_proposals.append(proposal)
                if on_paragraph_progress is not None:
                    on_paragraph_progress(number, total, source)
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
        _ensure_review_genre_attribution(proposal)
        _ensure_paragraph_not_too_long(packet, proposal)
        _ensure_paragraph_not_too_short(packet, proposal)
        _ensure_paragraph_language(packet, proposal)
    except (
        ParagraphCitationError,
        ParagraphGenreAttributionError,
        ParagraphLengthError,
        ParagraphLanguageError,
    ):
        return None
    return proposal


class WritingPlanRuntimeCache:
    """Persist successful section plans so one failure does not discard earlier work."""

    def __init__(self, root: Path, *, handoff: V04WritingHandoff) -> None:
        self._root = root / _handoff_fingerprint(handoff)[:16]
        self._shared_root = root / "shared"

    def load_section(self, packet: SectionEvidencePacket) -> WritingSectionPlan | None:
        path = self._root / f"{packet.section_id}.json"
        if path.is_file():
            return self._load_path(packet, path)
        return self._load_path(packet, self._shared_path(packet))

    def _load_path(
        self,
        packet: SectionEvidencePacket,
        path: Path,
    ) -> WritingSectionPlan | None:
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
        payload = json.dumps(
            {
                "schema_version": "0.4-plan-cache.0",
                "signature": _section_plan_signature(packet),
                "plan": plan.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
        self._root.mkdir(parents=True, exist_ok=True)
        _atomic_write(self._root / f"{packet.section_id}.json", payload)
        self._shared_root.mkdir(parents=True, exist_ok=True)
        _atomic_write(self._shared_path(packet), payload)

    def _shared_path(self, packet: SectionEvidencePacket) -> Path:
        signature = _section_plan_signature(packet)
        return self._shared_root / f"{packet.section_id}_{signature[:20]}.json"


class ParagraphWritingRuntimeCache:
    """Persist valid paragraphs across interruptions and compatible plan revisions."""

    def __init__(self, root: Path, *, plan_fingerprint: str) -> None:
        self._root = root / plan_fingerprint[:16]
        self._shared_root = root / "shared"

    def load(self, packet: ParagraphEvidencePacket) -> ParagraphTextProposal | None:
        current = self._path(packet)
        if current.is_file():
            proposal = self._load_path(packet, current)
            if proposal is not None:
                self._save_path(self._shared_path(packet), packet, proposal)
            return proposal
        return self._load_path(packet, self._shared_path(packet))

    def _load_path(
        self,
        packet: ParagraphEvidencePacket,
        path: Path,
    ) -> ParagraphTextProposal | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != _paragraph_signature(packet):
                return None
            proposal = ParagraphTextProposal.model_validate(payload["proposal"])
            _ensure_paragraph_not_too_long(packet, proposal)
            _ensure_paragraph_not_too_short(packet, proposal)
            _ensure_paragraph_has_no_authored_citation(proposal)
            _ensure_review_genre_attribution(proposal)
            _ensure_paragraph_language(packet, proposal)
            return proposal
        except Exception:
            return None

    def save(
        self,
        packet: ParagraphEvidencePacket,
        proposal: ParagraphTextProposal,
    ) -> None:
        _ensure_paragraph_not_too_long(packet, proposal)
        _ensure_paragraph_not_too_short(packet, proposal)
        _ensure_paragraph_has_no_authored_citation(proposal)
        _ensure_review_genre_attribution(proposal)
        _ensure_paragraph_language(packet, proposal)
        for path in (self._path(packet), self._shared_path(packet)):
            self._save_path(path, packet, proposal)

    def completed_count(
        self,
        section_packet: SectionEvidencePacket,
        section_plan: WritingSectionPlan,
    ) -> int:
        """Count durable, contract-valid paragraph checkpoints for one section."""

        builder = ParagraphEvidencePacketBuilder()
        return sum(
            self.load(builder.build(section_packet, paragraph)) is not None
            for paragraph in section_plan.paragraphs
        )

    def _save_path(
        self,
        path: Path,
        packet: ParagraphEvidencePacket,
        proposal: ParagraphTextProposal,
    ) -> None:
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

    def _shared_path(self, packet: ParagraphEvidencePacket) -> Path:
        signature = _paragraph_signature(packet)
        return (
            self._shared_root
            / packet.section_id
            / f"{packet.paragraph.paragraph_id}_{signature[:20]}.json"
        )


def _assign_required_sources_to_problem_paragraphs(
    packet: SectionEvidencePacket,
    proposal: SectionPlanProposal,
    *,
    evidence_aliases: dict[str, SectionEvidenceItem],
    source_aliases: dict[str, SectionSourceRecord],
    repair_invalid_permissions: bool = False,
) -> SectionPlanProposal:
    """Complete the chapter coverage table without adding coverage prose.

    The semantic planner often produces a useful argument skeleton but omits part of
    a large, hard-required bibliography. Code assigns each omitted source to the most
    lexically compatible existing problem paragraph with spare citation capacity. If
    a low-permission metadata source needs a home, a non-detailed paragraph is safely
    generalized to synthesis; no new paragraph or claim is manufactured.
    """

    alias_by_doi = {source.doi: alias for alias, source in source_aliases.items()}
    required_aliases = [
        alias_by_doi[doi]
        for doi in packet.required_source_dois
        if doi in alias_by_doi
    ]
    paragraphs = list(proposal.paragraphs)
    unknown_evidence = [
        ref
        for paragraph in paragraphs
        for ref in paragraph.evidence_refs
        if ref not in evidence_aliases
    ]
    unknown_sources = [
        ref
        for paragraph in paragraphs
        for ref in paragraph.source_refs
        if ref not in source_aliases
    ]
    if unknown_evidence or unknown_sources:
        raise WritingPlanError(
            "problem-driven plan used unknown short evidence/source aliases"
        )

    def evidence_dois(paragraph: ParagraphPlanProposal) -> set[str]:
        return {
            evidence_aliases[ref].doi
            for ref in paragraph.evidence_refs
            if ref in evidence_aliases
            and _permission_allows(
                source_aliases[alias_by_doi[evidence_aliases[ref].doi]].permitted_use,
                paragraph.role,
            )
        }

    def permitted_source_refs(paragraph: ParagraphPlanProposal) -> list[str]:
        return [
            ref
            for ref in paragraph.source_refs
            if ref in source_aliases
            and _permission_allows(source_aliases[ref].permitted_use, paragraph.role)
            and source_aliases[ref].doi not in evidence_dois(paragraph)
        ]

    normalized: list[ParagraphPlanProposal] = []
    for paragraph in paragraphs:
        # The proposal schema tolerates up to eight evidence refs so an over-selecting
        # model does not force a retry; cap to the executable five-card contract here.
        evidence_refs = list(dict.fromkeys(paragraph.evidence_refs))[:5]
        source_refs = list(dict.fromkeys(paragraph.source_refs))
        if repair_invalid_permissions:
            evidence_refs = [
                ref
                for ref in evidence_refs
                if _permission_allows(
                    source_aliases[
                        alias_by_doi[evidence_aliases[ref].doi]
                    ].permitted_use,
                    paragraph.role,
                )
            ]
            source_refs = [
                ref
                for ref in source_refs
                if _permission_allows(
                    source_aliases[ref].permitted_use,
                    paragraph.role,
                )
            ]
        normalized.append(
            paragraph.model_copy(
                update={
                    "evidence_refs": evidence_refs,
                    "source_refs": source_refs,
                }
            )
        )
    paragraphs = []
    globally_seen_dois: set[str] = set()
    for paragraph in normalized:
        paragraph_evidence_dois = evidence_dois(paragraph)
        globally_seen_dois.update(paragraph_evidence_dois)
        retained_refs: list[str] = []
        for ref in paragraph.source_refs:
            source = source_aliases[ref]
            if not _permission_allows(source.permitted_use, paragraph.role):
                # The first attempt preserves this mismatch so the semantic planner
                # receives a precise repair instruction. If the repaired response
                # repeats it, the second pass removes the invalid ref above and code
                # assigns compatible support instead of failing the whole section.
                retained_refs.append(ref)
                continue
            if source.doi in paragraph_evidence_dois or source.doi in globally_seen_dois:
                continue
            retained_refs.append(ref)
            globally_seen_dois.add(source.doi)
        paragraphs.append(
            paragraph.model_copy(update={"source_refs": retained_refs})
        )

    def covered_dois() -> set[str]:
        return {
            *(
                doi
                for paragraph in paragraphs
                for doi in evidence_dois(paragraph)
            ),
            *(
                source_aliases[ref].doi
                for paragraph in paragraphs
                for ref in permitted_source_refs(paragraph)
            ),
        }

    missing_aliases = [
        alias
        for alias in required_aliases
        if source_aliases[alias].doi not in covered_dois()
    ]
    # Allocate the least flexible permissions first. Otherwise a broadly usable
    # detailed/section-support source can greedily occupy the few synthesis slots
    # that a background-only source is allowed to use, producing a false
    # "no remaining capacity" failure even when total paragraph capacity is ample.
    permission_priority = {
        "background_only": 0,
        "section_support": 1,
        "detailed_claims": 2,
    }
    missing_aliases.sort(
        key=lambda alias: (
            permission_priority[source_aliases[alias].permitted_use],
            source_aliases[alias].doi,
        )
    )
    required_dois = {source_aliases[alias].doi for alias in required_aliases}
    for source_alias in missing_aliases:
        source = source_aliases[source_alias]
        candidates: list[tuple[float, int, int]] = []
        for index, paragraph in enumerate(paragraphs):
            current_dois = evidence_dois(paragraph) | {
                source_aliases[ref].doi
                for ref in permitted_source_refs(paragraph)
            }
            if (
                len(current_dois) >= packet.max_sources_per_paragraph
                or not _permission_allows(source.permitted_use, paragraph.role)
            ):
                continue
            candidates.append(
                (
                    _source_paragraph_fit(source, paragraph),
                    -len(current_dois),
                    -index,
                )
            )
        if not candidates:
            convertible = []
            for index, paragraph in enumerate(paragraphs):
                synthesized = paragraph.model_copy(update={"role": "synthesis"})
                synthesized_dois = evidence_dois(synthesized) | {
                    source_aliases[ref].doi
                    for ref in permitted_source_refs(synthesized)
                }
                if (
                    len(synthesized_dois) < packet.max_sources_per_paragraph
                ):
                    convertible.append(index)
            if convertible:
                index = min(
                    convertible,
                    key=lambda item: len(
                        evidence_dois(
                            paragraphs[item].model_copy(update={"role": "synthesis"})
                        )
                        | {
                            source_aliases[ref].doi
                            for ref in permitted_source_refs(
                                paragraphs[item].model_copy(
                                    update={"role": "synthesis"}
                                )
                            )
                        }
                    ),
                )
                paragraphs[index] = paragraphs[index].model_copy(
                    update={"role": "synthesis"}
                )
                candidates = [
                    (
                        _source_paragraph_fit(source, paragraphs[index]),
                        -len(paragraphs[index].source_refs),
                        -index,
                    )
                ]
        if not candidates:
            # A semantic proposal may fill every citation slot with optional
            # metadata sources before placing all contract-required sources. Do
            # not fail or raise the citation-cluster limit: replace the weakest
            # optional source ref in a compatible paragraph with the missing
            # required source. Evidence cards and other required sources remain
            # untouched.
            occurrence_counts: dict[str, int] = {}
            for planned in paragraphs:
                for doi in (
                    evidence_dois(planned)
                    | {
                        source_aliases[ref].doi
                        for ref in permitted_source_refs(planned)
                    }
                ):
                    occurrence_counts[doi] = occurrence_counts.get(doi, 0) + 1
            replacements: list[tuple[float, int, str]] = []
            for index, paragraph in enumerate(paragraphs):
                if not _permission_allows(source.permitted_use, paragraph.role):
                    continue
                optional_refs = [
                    ref
                    for ref in paragraph.source_refs
                    if (
                        source_aliases[ref].doi not in required_dois
                        or occurrence_counts.get(source_aliases[ref].doi, 0) > 1
                    )
                ]
                for optional_ref in optional_refs:
                    replacements.append(
                        (
                            _source_paragraph_fit(source, paragraph)
                            - _source_paragraph_fit(
                                source_aliases[optional_ref],
                                paragraph,
                            ),
                            -index,
                            optional_ref,
                        )
                    )
            if replacements:
                _, negative_index, optional_ref = max(replacements)
                index = -negative_index
                retained_refs = [
                    ref
                    for ref in paragraphs[index].source_refs
                    if ref != optional_ref
                ]
                paragraphs[index] = paragraphs[index].model_copy(
                    update={"source_refs": retained_refs}
                )
                candidates = [
                    (
                        _source_paragraph_fit(source, paragraphs[index]),
                        -len(
                            evidence_dois(paragraphs[index])
                            | {
                                source_aliases[ref].doi
                                for ref in permitted_source_refs(paragraphs[index])
                            }
                        ),
                        -index,
                    )
                ]
        if not candidates:
            raise WritingPlanError(
                "problem-driven paragraph plan has no remaining capacity for required "
                f"source {source.doi}"
            )
        _, _, negative_index = max(candidates)
        chosen = -negative_index
        refs = [*paragraphs[chosen].source_refs, source_alias]
        paragraphs[chosen] = paragraphs[chosen].model_copy(
            update={"source_refs": list(dict.fromkeys(refs))}
        )
    for index, paragraph in enumerate(paragraphs):
        if paragraph.evidence_refs or paragraph.source_refs:
            continue
        if paragraph.role == "detailed_evidence":
            compatible_evidence = [
                alias
                for alias, item in evidence_aliases.items()
                if _permission_allows(
                    source_aliases[alias_by_doi[item.doi]].permitted_use,
                    paragraph.role,
                )
            ]
            if compatible_evidence:
                paragraphs[index] = paragraph.model_copy(
                    update={"evidence_refs": [compatible_evidence[0]]}
                )
                continue
        compatible_sources = [
            alias
            for alias, source in source_aliases.items()
            if _permission_allows(source.permitted_use, paragraph.role)
        ]
        if not compatible_sources:
            raise WritingPlanError(
                f"paragraph {index + 1} has no compatible source after coverage assignment"
            )
        chosen_source = max(
            compatible_sources,
            key=lambda alias: _source_paragraph_fit(
                source_aliases[alias],
                paragraph,
            ),
        )
        paragraphs[index] = paragraph.model_copy(
            update={"source_refs": [chosen_source]}
        )
    return proposal.model_copy(update={"paragraphs": paragraphs})


def _source_paragraph_fit(
    source: SectionSourceRecord,
    paragraph: ParagraphPlanProposal,
) -> float:
    source_tokens = _semantic_tokens(
        " ".join(
            value
            for value in (
                source.title,
                source.abstract or "",
                source.supported_claim or "",
            )
            if value
        )
    )
    paragraph_tokens = _semantic_tokens(
        " ".join(
            (
                paragraph.purpose,
                paragraph.claim_focus,
                paragraph.central_question,
                paragraph.comparison_axis or "",
            )
        )
    )
    if not source_tokens or not paragraph_tokens:
        return 0.0
    return len(source_tokens & paragraph_tokens) / len(source_tokens | paragraph_tokens)


def _semantic_tokens(text: str) -> set[str]:
    latin = {
        token.casefold()
        for token in re.findall(r"[A-Za-z]{3,}", text)
    }
    compact_han = "".join(re.findall(r"[\u3400-\u9fff]", text))
    han_bigrams = {
        compact_han[index : index + 2]
        for index in range(max(0, len(compact_han) - 1))
    }
    return latin | han_bigrams


def _drop_semantically_misaligned_optional_evidence(
    proposal: SectionPlanProposal,
    *,
    evidence_aliases: dict[str, SectionEvidenceItem],
    source_aliases: dict[str, SectionSourceRecord],
) -> SectionPlanProposal:
    """Remove detailed cards that do not support a general paragraph intent.

    Background and synthesis paragraphs may be supported by admitted metadata records.
    Keeping an unrelated full-text card merely to make the packet look stronger confuses
    both the writer and reviewer: the writer borrows detail from the wrong study, while the
    reviewer correctly reports a topic/evidence mismatch. Detailed-evidence paragraphs keep
    their cards and continue through the stricter normal validation path.
    """

    source_by_doi = {source.doi: source for source in source_aliases.values()}
    paragraphs: list[ParagraphPlanProposal] = []
    for paragraph in proposal.paragraphs:
        if paragraph.role in {"detailed_evidence", "section_support"}:
            paragraphs.append(paragraph)
            continue
        focus_tokens = _semantic_tokens(
            " ".join(
                str(value or "")
                for value in (
                    paragraph.purpose,
                    paragraph.claim_focus,
                    paragraph.central_question,
                    paragraph.comparison_axis,
                )
            )
        )
        scored: list[tuple[float, str]] = []
        for ref in paragraph.evidence_refs:
            evidence = evidence_aliases.get(ref)
            if evidence is None:
                scored.append((1.0, ref))
                continue
            source = source_by_doi.get(evidence.doi)
            authority_tokens = _semantic_tokens(
                " ".join(
                    (
                        evidence.normalized_claim,
                        source.title if source is not None else "",
                        (source.supported_claim or "") if source is not None else "",
                    )
                )
            )
            denominator = max(1, min(len(focus_tokens), len(authority_tokens)))
            scored.append((len(focus_tokens & authority_tokens) / denominator, ref))
        retained = [ref for score, ref in scored if score >= 0.07]
        if not retained and not paragraph.source_refs and scored:
            retained = [max(scored)[1]]
        paragraphs.append(
            paragraph.model_copy(update={"evidence_refs": retained})
        )
    return proposal.model_copy(update={"paragraphs": paragraphs})


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
    paragraphs = _repair_compiled_required_source_coverage(packet, paragraphs)
    planned_source_dois = {
        doi for paragraph in paragraphs for doi in paragraph.source_dois
    }
    missing_required = [
        doi
        for doi in packet.required_source_dois
        if doi not in planned_source_dois
    ]
    if missing_required:
        raise WritingPlanError(
            "section plan omitted admitted sources required by the reference policy: "
            + ", ".join(missing_required)
        )
    return WritingSectionPlan(
        section_id=packet.section_id,
        title=packet.title,
        purpose=packet.purpose,
        target_words=packet.target_words,
        counting_policy=packet.counting_policy,
        paragraphs=paragraphs,
    )


def _repair_compiled_required_source_coverage(
    packet: SectionEvidencePacket,
    paragraphs: list[WritingParagraphPlan],
) -> list[WritingParagraphPlan]:
    """Keep hard-required sources after paragraph compilation trims optional support.

    Proposal allocation and paragraph compilation enforce the same citation-cluster
    limit at different representations. A proposal can therefore appear to cover a
    required DOI, then lose that DOI when compilation removes excess optional metadata
    support. Repair the compiled representation once, without changing prose or
    inventing evidence: use compatible spare capacity first, otherwise replace only an
    optional metadata-only DOI. Evidence-backed and other hard-required sources are never
    evicted.
    """

    if not packet.required_source_dois:
        return paragraphs
    source_by_doi = {source.doi: source for source in packet.sources}
    evidence_doi_by_id = {
        item.evidence_id: item.doi for item in packet.evidence_items
    }
    required = set(packet.required_source_dois)
    updated = list(paragraphs)

    def covered() -> set[str]:
        return {
            doi for paragraph in updated for doi in paragraph.source_dois
        }

    def occurrence_counts() -> dict[str, int]:
        counts: dict[str, int] = {}
        for paragraph in updated:
            for doi in set(paragraph.source_dois):
                counts[doi] = counts.get(doi, 0) + 1
        return counts

    for missing_doi in (
        doi for doi in packet.required_source_dois if doi not in covered()
    ):
        source = source_by_doi.get(missing_doi)
        if source is None:
            continue
        candidates: list[tuple[int, float, int, int, str | None]] = []
        for index, paragraph in enumerate(updated):
            if not _permission_allows(source.permitted_use, paragraph.role):
                continue
            paragraph_dois = list(dict.fromkeys(paragraph.source_dois))
            eviction: str | None = None
            if len(paragraph_dois) >= packet.max_sources_per_paragraph:
                occurrences = occurrence_counts()
                evidence_dois = {
                    evidence_doi_by_id[evidence_id]
                    for evidence_id in paragraph.evidence_card_ids
                    if evidence_id in evidence_doi_by_id
                }
                removable = [
                    doi
                    for doi in reversed(paragraph_dois)
                    if (
                        (doi not in required or occurrences.get(doi, 0) > 1)
                        and doi not in evidence_dois
                    )
                ]
                if not removable:
                    continue
                eviction = removable[0]
            candidates.append(
                (
                    1 if eviction is None else 0,
                    _source_paragraph_fit(source, paragraph),
                    -len(paragraph_dois),
                    -index,
                    eviction,
                )
            )
        if not candidates:
            occurrences = occurrence_counts()
            repurposeable = [
                (len(paragraph.evidence_card_ids), -index, index)
                for index, paragraph in enumerate(updated)
                if paragraph.role == "detailed_evidence"
                and all(
                    doi not in required or occurrences.get(doi, 0) > 1
                    for doi in paragraph.source_dois
                )
            ]
            if not repurposeable:
                continue
            _, _, chosen_index = min(repurposeable)
            chosen = updated[chosen_index]
            claim_focus = (source.supported_claim or source.title).strip()
            if packet.output_language == "Chinese":
                purpose = "提供与本章问题直接相关且不超出元数据权限的背景语境。"
                central_question = "该来源为本章问题提供了什么边界明确的背景信息？"
            else:
                purpose = (
                    "Provide permission-bounded background context for the section."
                )
                central_question = (
                    "What bounded background context does this source provide?"
                )
            updated[chosen_index] = chosen.model_copy(
                update={
                    "role": "background",
                    "purpose": purpose,
                    "claim_focus": claim_focus,
                    "central_question": central_question,
                    "argument_move": "frame_problem",
                    "comparison_axis": None,
                    "evidence_card_ids": [],
                    "source_dois": [missing_doi],
                    "coverage_only": False,
                }
            )
            continue
        _, _, _, negative_index, eviction = max(candidates)
        chosen_index = -negative_index
        chosen = updated[chosen_index]
        source_dois = [
            doi for doi in chosen.source_dois if doi != eviction
        ]
        source_dois.append(missing_doi)
        updated[chosen_index] = chosen.model_copy(
            update={"source_dois": list(dict.fromkeys(source_dois))}
        )
    return updated


def _compile_paragraph(
    packet: SectionEvidencePacket,
    proposal: ParagraphPlanProposal,
    *,
    number: int,
    target_words: int,
    evidence_aliases: dict[str, SectionEvidenceItem],
    source_aliases: dict[str, SectionSourceRecord],
) -> WritingParagraphPlan:
    if (
        proposal.central_question == "legacy_unspecified"
        or proposal.argument_move == "legacy_unspecified"
    ):
        raise WritingPlanError(
            f"paragraph {number} is missing its problem-driven argument contract"
        )
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
    purpose = proposal.purpose
    claim_focus = proposal.claim_focus
    central_question = proposal.central_question
    comparison_axis = proposal.comparison_axis
    if (
        not evidence_ids
        and proposal.role in {"background", "synthesis"}
        and _contains_unsupported_metadata_detail(proposal)
    ):
        # A metadata-only background/synthesis paragraph must not assert numeric
        # performance specifications. Deterministically paraphrase the bounded
        # metadata instead of failing the whole section over one clause.
        bounded = next(
            (
                source.supported_claim
                for source in permitted_sources
                if source.supported_claim
            ),
            None,
        )
        claim_focus = (bounded or proposal.claim_focus).strip()
        comparison_axis = None
        if packet.output_language == "Chinese":
            purpose = "综合所选来源的边界内背景信息，不陈述具体性能数值。"
            central_question = "所选来源提供了哪些边界明确的背景或共识？"
        else:
            purpose = (
                "Synthesize permission-bounded background from the selected sources "
                "without asserting specific performance figures."
            )
            central_question = (
                "What bounded background or consensus do the selected sources provide?"
            )
    if len(source_dois) > packet.max_sources_per_paragraph:
        # Source-count policy is deterministic. Keep evidence-backed sources first,
        # then trim optional metadata sources instead of discarding a semantically
        # useful section plan and asking the model to reproduce the whole JSON.
        source_dois = source_dois[: packet.max_sources_per_paragraph]
        permitted_dois = set(source_dois)
        evidence_items = [
            item for item in evidence_items if item.doi in permitted_dois
        ]
        evidence_ids = [item.evidence_id for item in evidence_items]
    argument_move = proposal.argument_move
    if (
        argument_move
        in {"compare_studies", "synthesize_consensus", "analyze_difference"}
        and len(source_dois) < 2
    ):
        # A model can choose a valid representative source but leave an internally
        # inconsistent comparison label. Preserve the locked evidence and safely
        # downgrade the rhetorical move instead of inventing a second source or
        # discarding the whole section plan.
        argument_move = (
            "evaluate_limitation"
            if proposal.role in {"detailed_evidence", "section_support"}
            else "frame_problem"
        )
        comparison_axis = None

    return WritingParagraphPlan(
        paragraph_id=f"{packet.section_id}_p{number:02d}",
        section_id=packet.section_id,
        paragraph_number=number,
        role=proposal.role,
        purpose=purpose,
        claim_focus=claim_focus,
        central_question=central_question,
        argument_move=argument_move,
        comparison_axis=comparison_axis,
        target_words=target_words,
        evidence_card_ids=evidence_ids,
        source_dois=source_dois,
    )


def _contains_unsupported_metadata_detail(proposal: ParagraphPlanProposal) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            proposal.purpose,
            proposal.claim_focus,
            proposal.central_question,
            proposal.comparison_axis,
        )
    )
    return bool(
        re.search(
            r"\d+(?:\.\d+)?(?:\s*[-–—]\s*\d+(?:\.\d+)?)?\s*"
            r"(?:%|m/s|m·s|cm(?:-1|⁻¹)?|nm|μm|km|K\b|channels?\b|通道|误差|精度)",
            text,
            re.IGNORECASE,
        )
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
    return max(3, min(12, round(target_words / 400)))


def _allocate_targets(
    weights: list[int],
    target_words: int,
    *,
    minimums: list[int] | None = None,
) -> list[int]:
    floors = minimums or [80] * len(weights)
    if len(floors) != len(weights):
        raise WritingPlanError("paragraph target floors do not match paragraph weights")
    if target_words < sum(floors):
        raise WritingPlanError("section target is too small for planned paragraphs")
    distributable = target_words - sum(floors)
    total_weight = sum(weights)
    raw = [distributable * weight / total_weight for weight in weights]
    targets = [floor + int(value) for floor, value in zip(floors, raw, strict=True)]
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
    output_language: str,
) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-writing-plan-v1",
            "topic": topic,
            "output_language": output_language,
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
    if not policy.references.all_bibliography_items_must_be_cited_and_discussed:
        # A bibliography count is a retrieval/release requirement, not permission to
        # manufacture prose around every selected paper. The planner may use only the
        # sources needed by each central claim; the final audit still checks the actual
        # citation count and routes a genuine shortage back to literature retrieval.
        return []
    return [record.doi for record in records]


def _apply_required_source_coverage(
    handoff: V04WritingHandoff,
    sections: list[WritingSectionPlan],
    *,
    required_source_dois: list[str],
) -> list[WritingSectionPlan]:
    repaired_sections: list[WritingSectionPlan] = []
    packet_builder = SectionEvidencePacketBuilder()
    for section in sections:
        packet = packet_builder.build(handoff, section.section_id)
        packet_source_dois = {source.doi for source in packet.sources}
        packet = packet.model_copy(
            update={
                "required_source_dois": [
                    doi
                    for doi in required_source_dois
                    if doi in packet_source_dois
                ]
            }
        )
        repaired_sections.append(
            section.model_copy(
                update={
                    "paragraphs": _repair_compiled_required_source_coverage(
                        packet,
                        section.paragraphs,
                    )
                }
            )
        )
    covered = {
        doi
        for section in repaired_sections
        for paragraph in section.paragraphs
        for doi in paragraph.source_dois
    }
    missing_required = [doi for doi in required_source_dois if doi not in covered]
    if missing_required:
        raise WritingPlanDependencyError(
            "required literature has no permission-compatible paragraph route: "
            + ", ".join(missing_required)
        )
    return repaired_sections


def _paragraph_requires_rewrite(
    previous: WritingParagraphPlan,
    current: WritingParagraphPlan,
) -> bool:
    """Ignore budget-only shifts that existing prose can satisfy without rewriting."""

    fields = (
        "paragraph_id",
        "section_id",
        "paragraph_number",
        "role",
        "purpose",
        "claim_focus",
        "coverage_only",
        "evidence_card_ids",
        "source_dois",
    )
    return any(getattr(previous, field) != getattr(current, field) for field in fields)


def _paragraph_signature(packet: ParagraphEvidencePacket) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-paragraph-writer-v3-length-floor",
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
    # The full-manuscript reviewer treats a paragraph above 1200 counted units as a
    # structural defect.  Keep the writer contract aligned with that downstream gate.
    return min(
        1200,
        max(packet.paragraph.target_words, int(packet.paragraph.target_words * 1.6)),
    )


def _paragraph_minimum_units(packet: ParagraphEvidencePacket) -> int:
    # Paragraph budgets are internal planning targets, not user-declared hard minima.
    # The manuscript-level audit owns the actual length requirement. Keep only a
    # catastrophic under-generation floor here so a usable concise paragraph does
    # not trigger three expensive rewrites.
    effective_target = min(packet.paragraph.target_words, 1100)
    if effective_target < 300:
        return 0
    return int(effective_target * 0.5)


def _paragraph_minimum_tolerance(minimum_units: int) -> int:
    """Allow harmless rounding drift around an internal paragraph budget.

    This floor is a planning heuristic, not a user-declared manuscript minimum.
    Rejecting an otherwise valid paragraph for a two-character shortfall causes
    expensive, unstable rewrites without improving the finished paper.  Keep the
    tolerance small enough that an empty or severely truncated paragraph still enters
    repair. Manuscript-level length compliance remains deterministic downstream.
    """

    return max(4, int(minimum_units * 0.1 + 0.999))


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


def _ensure_paragraph_not_too_short(
    packet: ParagraphEvidencePacket,
    proposal: ParagraphTextProposal,
) -> None:
    minimum_units = _paragraph_minimum_units(packet)
    if not minimum_units:
        return
    counted_units = count_writing_units(
        proposal.text,
        counting_policy=packet.counting_policy,
    )
    enforced_minimum = minimum_units - _paragraph_minimum_tolerance(minimum_units)
    if counted_units < enforced_minimum:
        raise ParagraphTooShortError(
            f"paragraph has {counted_units} counted units; planned minimum is "
            f"{minimum_units}, tolerated minimum is {enforced_minimum}"
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


def _ensure_review_genre_attribution(proposal: ParagraphTextProposal) -> None:
    detail = false_self_attribution_detail(proposal.text)
    if detail:
        raise ParagraphGenreAttributionError(detail)


def _ensure_paragraph_language(
    packet: ParagraphEvidencePacket,
    proposal: ParagraphTextProposal,
) -> None:
    detail = language_mismatch_detail(
        proposal.text,
        output_language=packet.output_language,
    )
    if detail:
        raise ParagraphLanguageError(detail)


def _paragraph_repair_instruction(
    exc: Exception,
    *,
    maximum_units: int,
) -> str:
    if isinstance(exc, ParagraphTooShortError):
        return (
            "Rewrite only the paragraph text and develop the same evidence-bound central "
            "argument more fully. Add comparison, synthesis, limitations, and cautious "
            "implications already supported by the packet; do not add sources or filler. "
            "Return exactly one text field."
        )
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
    if isinstance(exc, ParagraphGenreAttributionError):
        return (
            "Rewrite only the paragraph text as literature-review prose. Do not claim "
            "that the current paper or its authors conducted a cited method, dataset, "
            "measurement, experiment, or result. Use the responsible source author names "
            "from packet.sources as the subject and preserve the locked evidence scope. "
            "Return exactly one text field."
        )
    if isinstance(exc, ParagraphLanguageError):
        return (
            "Rewrite only the paragraph text in the confirmed output language. Preserve "
            "the evidence-bound meaning, remove full sentences in other languages, and "
            "return exactly one text field."
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
    # Multiple Streamlit sessions can finish the same cached paragraph together.
    # A shared ``.tmp`` filename lets one session replace a file while another still
    # owns it on Windows (WinError 32). Per-writer temp names keep the final replace
    # atomic and make repeated writes idempotent.
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
