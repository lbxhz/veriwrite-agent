"""Live DeepSeek convergence test for the V0.4 targeted manuscript-repair loop.

The FakeLLM suite proves the deterministic repair logic is sound, but it cannot show
whether a *real* writer obeys a narrowed plan well enough for the *real* reviewer to
drop its blocking findings on the next round. This script drives one full repair round
against DeepSeek and reports the blocking/targeted_repair count before and after:

    1. independent full-manuscript review        -> blocking findings
    2. refine_writing_plan_for_manuscript_review -> narrowed plan (code-owned)
    3. rewrite each targeted paragraph           -> real writer, revision_instruction
    4. re-review                                 -> did the findings converge?

It reuses the prior run's ``writing_plan.json`` and ``writing_project.json`` and never
mutates the active Streamlit project. Dry-run by default; pass ``--yes`` to call the API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.writing import V04WritingProject
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.services.grounded_writing import SectionEvidencePacketBuilder
from veriwrite_agent.services.writing_planning import (
    LLMGroundedParagraphWriter,
    ParagraphEvidencePacketBuilder,
)
from veriwrite_agent.services.writing_quality import (
    LLMManuscriptQualityReviewer,
    refine_writing_plan_for_manuscript_review,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    REPO_ROOT
    / "runtime"
    / "generated_runs"
    / "formal_course_paper_grounded_20260805"
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _signature(finding) -> tuple[str, int, str]:
    return (finding.section_id, finding.paragraph_number, finding.code)


def _rewrite_targeted_paragraphs(
    project: V04WritingProject,
    refined: GroundedWritingPlan,
    targeted: list,
    writer: LLMGroundedParagraphWriter,
    section_packets: dict[str, object],
) -> V04WritingProject:
    plan_by_id = {section.section_id: section for section in refined.sections}
    by_target: dict[tuple[str, int], list] = {}
    for finding in targeted:
        by_target.setdefault(
            (finding.section_id, finding.paragraph_number), []
        ).append(finding)

    new_sections = []
    for state in project.sections:
        targets = {
            number: findings
            for (section_id, number), findings in by_target.items()
            if section_id == state.section_id
        }
        if not targets or state.draft is None:
            new_sections.append(state)
            continue
        section_packet = section_packets[state.section_id]
        section_plan = plan_by_id[state.section_id]
        original_paragraphs = list(state.draft.paragraphs)
        paragraphs = list(original_paragraphs)
        for number, findings in sorted(targets.items()):
            index = number - 1
            paragraph_plan = section_plan.paragraphs[index]
            packet = ParagraphEvidencePacketBuilder().build(
                section_packet,
                paragraph_plan,
            )
            instruction = " ".join(
                dict.fromkeys(f.revision_instruction for f in findings)
            )
            # Mirror PlannedSectionDraftService: pass the surrounding original
            # paragraphs so the writer can avoid repeating them, and stay serial.
            editorial_context = [
                {"paragraph_number": n, "role": p.role, "text": p.text}
                for n, p in enumerate(original_paragraphs, 1)
                if n != number
            ]
            proposal = writer.write(
                packet,
                revision_instruction=instruction,
                editorial_context=editorial_context,
            )
            old = paragraphs[index]
            paragraphs[index] = old.model_copy(
                update={
                    "text": proposal.text,
                    "role": paragraph_plan.role,
                    "evidence_card_ids": paragraph_plan.evidence_card_ids,
                    "source_dois": paragraph_plan.source_dois,
                }
            )
        new_draft = state.draft.model_copy(update={"paragraphs": paragraphs})
        new_sections.append(state.model_copy(update={"draft": new_draft}))
    return project.model_copy(update={"sections": new_sections})


def run(args: argparse.Namespace) -> None:
    handoff_path = SOURCE_RUN / "writing_handoff.json"
    plan_path = SOURCE_RUN / "writing_plan.json"
    project_path = SOURCE_RUN / "writing_project.json"
    for path in (handoff_path, plan_path, project_path):
        if not path.is_file():
            raise SystemExit(f"missing input artifact: {path}")

    handoff = V04WritingHandoff.model_validate(_load(handoff_path))
    plan = GroundedWritingPlan.model_validate(_load(plan_path))
    project = V04WritingProject.model_validate(_load(project_path))

    if args.dry_run:
        print(
            "[converge] DRY RUN: would run review -> refine -> rewrite -> re-review "
            "against real DeepSeek. Pass --yes to execute.",
            flush=True,
        )
        return

    settings = LLMSettings().for_quality_review().model_copy(
        update={"timeout_seconds": args.timeout_seconds, "max_retries": 3}
    )
    writer_settings = LLMSettings().for_structured_output().model_copy(
        update={"timeout_seconds": args.timeout_seconds, "max_retries": 3}
    )

    reviewer = LLMManuscriptQualityReviewer(DeepSeekClient(settings))

    review = reviewer.review(plan, project)
    blocking = [
        finding for finding in review.findings if finding.severity == "blocking"
    ]
    targeted = [
        finding
        for finding in review.findings
        if finding.disposition == "targeted_repair"
    ]
    print(
        f"[converge] round 1 review status={review.review_status} "
        f"findings={len(review.findings)} blocking={len(blocking)} "
        f"targeted_repair={len(targeted)}",
        flush=True,
    )
    for finding in targeted:
        print(
            f"  - {finding.section_id} p{finding.paragraph_number} "
            f"[{finding.code}] {finding.detail[:80]}",
            flush=True,
        )

    if not targeted:
        print("[converge] no targeted_repair findings to repair; loop is empty.", flush=True)
        return

    refined = refine_writing_plan_for_manuscript_review(
        plan,
        review,
        evidence_doi_by_id={
            card.evidence_id: card.doi
            for card in project.handoff.evidence_library.evidence_cards
        },
    )

    writer = LLMGroundedParagraphWriter(DeepSeekClient(writer_settings))
    section_packets = {
        section.section_id: SectionEvidencePacketBuilder().build(
            handoff,
            section.section_id,
        )
        for section in refined.sections
    }
    updated_project = _rewrite_targeted_paragraphs(
        project,
        refined,
        targeted,
        writer,
        section_packets,
    )

    review2 = reviewer.review(refined, updated_project)
    blocking2 = [
        finding for finding in review2.findings if finding.severity == "blocking"
    ]
    targeted2 = [
        finding
        for finding in review2.findings
        if finding.disposition == "targeted_repair"
    ]
    print(
        f"[converge] round 2 review status={review2.review_status} "
        f"findings={len(review2.findings)} blocking={len(blocking2)} "
        f"targeted_repair={len(targeted2)}",
        flush=True,
    )
    for finding in targeted2:
        print(
            f"  - {finding.section_id} p{finding.paragraph_number} "
            f"[{finding.code}] {finding.detail[:80]}",
            flush=True,
        )

    before = {_signature(f) for f in targeted}
    after = {_signature(f) for f in targeted2}
    resolved = before - after
    remaining = before & after
    introduced = after - before
    print(
        f"[converge] resolved={len(resolved)} remaining={len(remaining)} "
        f"introduced={len(introduced)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live DeepSeek convergence test for targeted manuscript repair."
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--yes",
        dest="dry_run",
        action="store_false",
        help="actually call the DeepSeek API (default is a dry run)",
    )
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
