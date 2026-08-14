"""Deterministic structural edits authorized by the full-manuscript reviewer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil

from veriwrite_agent.models.writing import (
    DraftParagraphProposal,
    SectionDraftProposal,
    V04WritingProject,
    WritingSectionState,
)
from veriwrite_agent.models.writing_plan import (
    GroundedWritingPlan,
    WritingParagraphPlan,
    WritingSectionPlan,
)
from veriwrite_agent.models.writing_quality import ManuscriptQualityReview
from veriwrite_agent.services.grounded_writing import (
    GroundedSectionDraftService,
    SectionEvidencePacketBuilder,
)
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.services.writing_planning import (
    GroundedWritingPlanner,
    WritingPlanError,
)


MAX_STRUCTURAL_PARAGRAPH_TARGET = 1150


@dataclass(frozen=True)
class ManuscriptStructuralEditResult:
    plan: GroundedWritingPlan
    project: V04WritingProject
    target_remap: dict[tuple[str, int], tuple[str, int]]


def semantically_replan_manuscript_sections(
    plan: GroundedWritingPlan,
    project: V04WritingProject,
    *,
    section_ids: set[str],
    planner: GroundedWritingPlanner,
) -> GroundedWritingPlan:
    """Let an LLM rebuild claims and source assignment after code changes structure.

    Paragraph deletion, numbering and citation capacity are deterministic operations.  The
    decision that a source supports one surviving claim better than another is semantic and
    therefore belongs to the planner.  Its proposal is still compiled through the ordinary
    evidence permissions, source limits and required-reference coverage checks.
    """

    if not section_ids:
        return plan
    replacements: dict[str, WritingSectionPlan] = {}
    evidence_doi_by_id = {
        card.evidence_id: card.doi
        for card in project.handoff.evidence_library.evidence_cards
    }
    for section in plan.sections:
        if section.section_id not in section_ids:
            continue
        packet = SectionEvidencePacketBuilder().build(
            project.handoff,
            section.section_id,
        )
        packet_source_dois = {source.doi for source in packet.sources}
        previously_used_dois = {
            *(
                doi
                for paragraph in section.paragraphs
                for doi in paragraph.source_dois
            ),
            *(
                evidence_doi_by_id[evidence_id]
                for paragraph in section.paragraphs
                for evidence_id in paragraph.evidence_card_ids
                if evidence_id in evidence_doi_by_id
            ),
        }
        packet = packet.model_copy(
            update={
                "target_words": section.target_words,
                "required_source_dois": [
                    doi
                    for doi in plan.required_source_dois
                    if doi in packet_source_dois and doi in previously_used_dois
                ],
            }
        )
        try:
            replacements[section.section_id] = planner.replan_section(
                packet,
                paragraph_count=len(section.paragraphs),
            )
        except WritingPlanError:
            # Semantic replanning is an optimization, not the only repair path.
            # The caller has already refined the affected paragraph intents from
            # the independent editor findings. If two planner attempts still
            # violate a deterministic contract, preserve that refined plan and
            # continue with the bounded paragraph rewrites instead of blocking the
            # entire manuscript on an optional whole-section replan.
            continue
    if not replacements:
        return plan
    sections = [replacements.get(section.section_id, section) for section in plan.sections]
    return GroundedWritingPlan.model_validate(
        plan.model_copy(
            update={
                "sections": sections,
                "plan_fingerprint": _semantic_plan_fingerprint(plan, sections),
            }
        ).model_dump(mode="json")
    )


def merge_redundant_manuscript_paragraphs(
    plan: GroundedWritingPlan,
    project: V04WritingProject,
    review: ManuscriptQualityReview | None,
) -> ManuscriptStructuralEditResult:
    """Merge reviewer-confirmed redundant paragraphs into an adjacent main paragraph.

    The prose to remove may carry a source that appears nowhere else.  Instead of silently
    deleting that citation, this function moves its locked sources and evidence cards into
    the adjacent paragraph plan.  The paragraph writer then regenerates that recipient from
    the merged evidence packet.  Text changes remain LLM-owned; paragraph identity, support
    migration, numbering, citations, and plan fingerprints remain code-owned.
    """

    targets = _merge_targets(review)
    if not targets:
        return ManuscriptStructuralEditResult(plan, project, {})
    state_by_id = {state.section_id: state for state in project.sections}
    policy = project.handoff.requirement_policy or RequirementPolicyCompiler().compile(
        project.handoff.requirement
    )
    max_sources_per_paragraph = (
        policy.references.max_references_per_citation_cluster or 4
    )
    evidence_source_by_id = {
        card.evidence_id: card.doi
        for card in project.handoff.evidence_library.evidence_cards
    }
    refined_sections: list[WritingSectionPlan] = []
    refined_states: list[WritingSectionState] = []
    remap: dict[tuple[str, int], tuple[str, int]] = {}
    changed = False
    for section in plan.sections:
        section_targets = targets.get(section.section_id, set())
        state = state_by_id[section.section_id]
        if not section_targets or state.draft is None:
            refined_sections.append(section)
            refined_states.append(state)
            continue
        valid_targets = {
            number
            for number in section_targets
            if 1 <= number <= len(section.paragraphs)
        }
        selected_targets = _select_removable_targets(
            section,
            valid_targets,
            max_sources_per_paragraph=max_sources_per_paragraph,
        )
        if not selected_targets:
            refined_sections.append(section)
            refined_states.append(state)
            continue

        recipient_by_target = {
            number: _recipient_number(
                number,
                paragraph_count=len(section.paragraphs),
                removed=selected_targets,
            )
            for number in selected_targets
        }
        remaining_numbers = [
            number
            for number in range(1, len(section.paragraphs) + 1)
            if number not in selected_targets
        ]
        packet = SectionEvidencePacketBuilder().build(
            project.handoff,
            section.section_id,
        )
        plans = _redistribute_section_support(
            section,
            remaining_numbers=remaining_numbers,
            max_sources_per_paragraph=max_sources_per_paragraph,
            evidence_source_by_id=evidence_source_by_id,
            source_permission_by_doi={
                source.doi: source.permitted_use for source in packet.sources
            },
            recipient_by_target=recipient_by_target,
        )
        old_to_new = {
            old_number: new_number
            for new_number, old_number in enumerate(remaining_numbers, 1)
        }
        paragraphs = [
            _renumber_plan(
                plans[old_number],
                section_id=section.section_id,
                paragraph_number=old_to_new[old_number],
            )
            for old_number in remaining_numbers
        ]
        fixed_numbers = {
            old_to_new[number]
            for number in valid_targets - selected_targets
            if number in old_to_new
        }
        paragraphs = _rebalance_section_targets(
            paragraphs,
            section_target=section.target_words,
            fixed_numbers=fixed_numbers,
        )
        refined_section = section.model_copy(
            update={"paragraphs": paragraphs, "target_words": section.target_words}
        )
        proposals = []
        for old_number in remaining_numbers:
            source = state.draft.paragraphs[old_number - 1]
            # Preserve the old binding in the intermediate draft.  If support was moved,
            # PlannedSectionDraftService will see that it no longer matches the new plan
            # and regenerate only that surviving paragraph.  Rebinding the old prose here
            # would falsely make an untouched sentence appear to support newly moved
            # sources.
            proposals.append(
                DraftParagraphProposal(
                    role=source.role,
                    text=source.text,
                    evidence_card_ids=source.evidence_card_ids,
                    source_dois=source.source_dois,
                )
            )
        rebuilt = GroundedSectionDraftService().create(
            packet,
            SectionDraftProposal(
                section_id=section.section_id,
                paragraphs=proposals,
            ),
        )
        refined_sections.append(refined_section)
        refined_states.append(
            WritingSectionState(
                section_id=section.section_id,
                status=rebuilt.status,
                draft=rebuilt,
            )
        )
        for target_number, recipient_number in recipient_by_target.items():
            remap[(section.section_id, target_number)] = (
                section.section_id,
                old_to_new[recipient_number],
            )
        for old_number, new_number in old_to_new.items():
            remap[(section.section_id, old_number)] = (
                section.section_id,
                new_number,
            )
        changed = True

    if not changed:
        return ManuscriptStructuralEditResult(plan, project, {})
    refined_plan = GroundedWritingPlan.model_validate(
        plan.model_copy(
            update={
                "sections": refined_sections,
                "plan_fingerprint": _structural_plan_fingerprint(
                    plan,
                    refined_sections,
                ),
            }
        ).model_dump(mode="json")
    )
    refined_project = V04WritingProject.model_validate(
        project.model_copy(
            update={
                "status": "drafting",
                "sections": refined_states,
                "updated_at": datetime.now(timezone.utc),
            }
        ).model_dump(mode="json")
    )
    return ManuscriptStructuralEditResult(refined_plan, refined_project, remap)


def _merge_targets(
    review: ManuscriptQualityReview | None,
) -> dict[str, set[int]]:
    targets: dict[str, set[int]] = {}
    if review is None:
        return targets
    for finding in review.findings:
        if (
            finding.severity != "blocking"
            or finding.disposition != "targeted_repair"
            or finding.code
            not in {
                "cross_section_repetition",
                "paragraph_repetition",
                "section_role_overlap",
            }
        ):
            continue
        targets.setdefault(finding.section_id, set()).add(
            finding.paragraph_number
        )
    return targets


def _recipient_number(
    target: int,
    *,
    paragraph_count: int,
    removed: set[int],
) -> int:
    for number in range(target - 1, 0, -1):
        if number not in removed:
            return number
    for number in range(target + 1, paragraph_count + 1):
        if number not in removed:
            return number
    raise ValueError("a redundant paragraph has no surviving merge recipient")


def _select_removable_targets(
    section: WritingSectionPlan,
    targets: set[int],
    *,
    max_sources_per_paragraph: int,
) -> set[int]:
    """Remove only as many duplicate paragraphs as the evidence and word budget allow."""

    if not targets:
        return set()
    unique_sources = {
        doi for paragraph in section.paragraphs for doi in paragraph.source_dois
    }
    minimum_by_sources = ceil(
        len(unique_sources) / max(1, max_sources_per_paragraph)
    )
    minimum_by_length = ceil(
        section.target_words / MAX_STRUCTURAL_PARAGRAPH_TARGET
    )
    minimum_remaining = max(2, minimum_by_sources, minimum_by_length)
    removable = max(0, len(section.paragraphs) - minimum_remaining)
    if removable == 0:
        return set()

    source_use_count: dict[str, int] = {}
    for paragraph in section.paragraphs:
        for doi in paragraph.source_dois:
            source_use_count[doi] = source_use_count.get(doi, 0) + 1
    ranked = sorted(
        targets,
        key=lambda number: (
            sum(
                source_use_count.get(doi, 0) == 1
                for doi in section.paragraphs[number - 1].source_dois
            ),
            -number,
        ),
    )
    selected: set[int] = set()
    for number in ranked:
        if len(selected) >= removable:
            break
        trial = {*selected, number}
        remaining = [
            paragraph
            for paragraph in section.paragraphs
            if paragraph.paragraph_number not in trial
        ]
        surviving_sources = {
            doi for paragraph in remaining for doi in paragraph.source_dois
        }
        # Preserve all support occurrences already attached to surviving claims, then
        # reserve one slot for every source that would be orphaned by the deletions.
        # Counting only unique section sources can over-delete: six survivors may have
        # 24 slots in total but need 25 once legitimate repeated bindings are retained.
        required_slots = sum(len(paragraph.source_dois) for paragraph in remaining)
        required_slots += len(unique_sources - surviving_sources)
        if required_slots > len(remaining) * max_sources_per_paragraph:
            continue
        selected.add(number)
    return selected


def _redistribute_section_support(
    section: WritingSectionPlan,
    *,
    remaining_numbers: list[int],
    max_sources_per_paragraph: int,
    evidence_source_by_id: dict[str, str],
    source_permission_by_doi: dict[str, str],
    recipient_by_target: dict[int, int],
) -> dict[int, WritingParagraphPlan]:
    """Preserve surviving authority and migrate only support orphaned by deletion.

    A structural de-duplication pass is not a semantic replanner.  Reassigning every
    source in the section while retaining the old claims can pair a metadata-only DOI
    with a detailed claim and manufacture a V0.3 recovery request.  Keep every surviving
    paragraph's original bindings intact; only sources and cards that would otherwise
    disappear are moved to the closest compatible survivor.
    """

    original = {
        paragraph.paragraph_number: paragraph for paragraph in section.paragraphs
    }
    source_order = list(
        dict.fromkeys(
            doi
            for paragraph in section.paragraphs
            for doi in paragraph.source_dois
        )
    )
    owners = {
        doi: [
            paragraph.paragraph_number
            for paragraph in section.paragraphs
            if doi in paragraph.source_dois
        ]
        for doi in source_order
    }
    removed_numbers = sorted(set(original) - set(remaining_numbers))
    assignments: dict[int, list[str]] = {
        number: list(original[number].source_dois) for number in remaining_numbers
    }
    card_assignments: dict[int, list[str]] = {
        number: list(original[number].evidence_card_ids)
        for number in remaining_numbers
    }

    if any(
        len(sources) > max_sources_per_paragraph
        for sources in assignments.values()
    ):
        raise ValueError(
            "surviving paragraph already exceeds the confirmed citation-cluster limit"
        )

    def nearest_removed_owner(doi: str) -> int:
        removed_owners = [owner for owner in owners.get(doi, []) if owner in removed_numbers]
        return removed_owners[0] if removed_owners else owners.get(doi, [1])[0]

    def compatibility_rank(number: int, doi: str, owner: int) -> tuple[int, ...]:
        paragraph = original[number]
        owner_paragraph = original[owner]
        permission = source_permission_by_doi.get(doi)
        if permission == "background_only":
            semantic_rank = (
                0
                if paragraph.argument_move == "author_judgment"
                else 1
                if paragraph.role == "background"
                else 2
                if paragraph.evidence_card_ids
                else 3
            )
        else:
            semantic_rank = (
                0
                if paragraph.argument_move == owner_paragraph.argument_move
                else 1
                if paragraph.role == owner_paragraph.role
                else 2
            )
        designated_recipient = recipient_by_target.get(owner)
        return (
            semantic_rank,
            0 if number == designated_recipient else 1,
            abs(number - owner),
            len(assignments[number]),
            number,
        )

    surviving_sources = {
        doi for sources in assignments.values() for doi in sources
    }
    for doi in source_order:
        if doi in surviving_sources:
            continue
        candidates = [
            number
            for number in remaining_numbers
            if len(assignments[number]) < max_sources_per_paragraph
        ]
        if not candidates:
            raise ValueError(
                "section sources cannot fit the confirmed citation-cluster limit"
            )
        owner = nearest_removed_owner(doi)
        chosen = min(
            candidates,
            key=lambda number: compatibility_rank(number, doi, owner),
        )
        assignments[chosen].append(doi)
        surviving_sources.add(doi)

    # Retain evidence already owned by surviving claims.  Then migrate cards from removed
    # paragraphs without evicting those bindings.  A duplicated DOI is allowed when that is
    # the only capacity-safe way to keep a unique page-level card available to the rewrite.
    surviving_cards = {
        card_id for cards in card_assignments.values() for card_id in cards
    }
    for owner in removed_numbers:
        for card_id in original[owner].evidence_card_ids:
            if card_id in surviving_cards:
                continue
            doi = evidence_source_by_id.get(card_id)
            if doi is None:
                continue
            candidates = [
                number
                for number in remaining_numbers
                if doi in assignments[number] and len(card_assignments[number]) < 5
            ]
            if not candidates:
                candidates = [
                    number
                    for number in remaining_numbers
                    if len(assignments[number]) < max_sources_per_paragraph
                    and len(card_assignments[number]) < 5
                ]
            if not candidates:
                continue
            chosen = min(
                candidates,
                key=lambda number: compatibility_rank(number, doi, owner),
            )
            if doi not in assignments[chosen]:
                assignments[chosen].append(doi)
            card_assignments[chosen].append(card_id)
            surviving_cards.add(card_id)

    updated: dict[int, WritingParagraphPlan] = {}
    for number in remaining_numbers:
        paragraph = original[number]
        sources = assignments[number]
        cards = card_assignments[number]
        role = paragraph.role
        background_only_dois = [
            doi
            for doi in sources
            if source_permission_by_doi.get(doi) == "background_only"
        ]
        requires_bounded_background = bool(background_only_dois) and role != "background"
        if requires_bounded_background:
            # A source-coverage requirement may leave a metadata-only DOI orphaned when
            # its repeated paragraph is removed.  Moving that DOI into a detailed role
            # would pass reference coverage but fail the deterministic permission gate.
            # Keep the original intent as a deferred enhancement and compile a cautious
            # background task now; a later PDF-backed pass can restore the detailed claim.
            role = "background"
        elif role == "detailed_evidence" and not cards:
            role = "synthesis"
        updates: dict[str, object] = {
            "role": role,
            "evidence_card_ids": cards,
            "source_dois": sources,
        }
        if requires_bounded_background:
            updates.update(
                {
                    "purpose": (
                        "State a bounded background overview and the limits of the "
                        "available records after structural consolidation."
                    ),
                    "claim_focus": (
                        f"{section.title} includes directions represented by the admitted "
                        "sources, but metadata-only records permit only a general account "
                        "of their scope."
                    ),
                    "central_question": (
                        "What general scope is supported, and which detailed conclusions "
                        "remain outside the available evidence?"
                    ),
                    "argument_move": "author_judgment",
                    "comparison_axis": None,
                    "deferred_argument": (
                        paragraph.deferred_argument or paragraph.argument_move
                    ),
                    "deferred_comparison_axis": (
                        paragraph.deferred_comparison_axis or paragraph.comparison_axis
                    ),
                    "deferred_purpose": paragraph.deferred_purpose or paragraph.purpose,
                    "deferred_claim_focus": (
                        paragraph.deferred_claim_focus or paragraph.claim_focus
                    ),
                    "deferred_central_question": (
                        paragraph.deferred_central_question
                        or paragraph.central_question
                    ),
                    "deferred_recovery_dois": list(
                        dict.fromkeys(
                            [
                                *paragraph.deferred_recovery_dois,
                                *background_only_dois,
                            ]
                        )
                    ),
                }
            )
        updated[number] = paragraph.model_copy(update=updates)
    return updated


def _rebalance_section_targets(
    paragraphs: list[WritingParagraphPlan],
    *,
    section_target: int,
    fixed_numbers: set[int],
) -> list[WritingParagraphPlan]:
    """Keep concise repair targets fixed and spread the remaining budget evenly."""

    fixed_total = sum(
        paragraph.target_words
        for paragraph in paragraphs
        if paragraph.paragraph_number in fixed_numbers
    )
    recipients = [
        index
        for index, paragraph in enumerate(paragraphs)
        if paragraph.paragraph_number not in fixed_numbers
    ]
    if not recipients:
        recipients = list(range(len(paragraphs)))
        fixed_total = 0
    remaining = section_target - fixed_total
    floors = [50] * len(recipients)
    if remaining < sum(floors):
        raise ValueError("section budget is too small after structural editing")
    share, remainder = divmod(remaining, len(recipients))
    targets = [share + (1 if order < remainder else 0) for order in range(len(recipients))]
    updated = list(paragraphs)
    for index, target in zip(recipients, targets, strict=True):
        updated[index] = updated[index].model_copy(update={"target_words": target})
    return updated


def _renumber_plan(
    paragraph: WritingParagraphPlan,
    *,
    section_id: str,
    paragraph_number: int,
) -> WritingParagraphPlan:
    return paragraph.model_copy(
        update={
            "paragraph_id": f"{section_id}_p{paragraph_number:02d}",
            "paragraph_number": paragraph_number,
        }
    )


def _structural_plan_fingerprint(
    plan: GroundedWritingPlan,
    sections: list[WritingSectionPlan],
) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-writing-plan-v3-structural-editor",
            "topic": plan.topic,
            "output_language": plan.output_language,
            "required_source_dois": plan.required_source_dois,
            "sections": [section.model_dump(mode="json") for section in sections],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _semantic_plan_fingerprint(
    plan: GroundedWritingPlan,
    sections: list[WritingSectionPlan],
) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "grounded-writing-plan-v5-semantic-global-editor",
            "topic": plan.topic,
            "output_language": plan.output_language,
            "required_source_dois": plan.required_source_dois,
            "sections": [section.model_dump(mode="json") for section in sections],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
