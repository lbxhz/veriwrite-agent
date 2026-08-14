"""Live DeepSeek smoke test for the two least-exercised V0.4 writing stages.

The FakeLLM test suite returns perfect JSON, so it cannot surface the failures a
real provider produces: empty output, truncation at ``max_tokens``, and complex
schema-contract violations. This script exercises the two stages that depend most
on those behaviors — writing **planning** and **final matter** — against the real
DeepSeek API while reusing the prior run's evidence library, selection, and body.

    --stage plan    re-plan every section (reuse_cache=False)
    --stage matter  generate final matter from the existing confirmed body
    --stage all     both (default)

Safety: never mutates the active Streamlit project and dry-runs by default. Pass
``--yes`` to actually call the API.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.base import ChatMessage
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.writing import BodyDraftPackage, V04WritingProject
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.services.final_delivery import LLMFinalMatterWriter
from veriwrite_agent.services.writing_planning import GroundedWritingPlanner
from veriwrite_agent.services.writing_quality import LLMManuscriptQualityReviewer

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    REPO_ROOT
    / "runtime"
    / "generated_runs"
    / "formal_course_paper_grounded_20260805"
)


@dataclass
class CallRecord:
    label: str = ""
    duration: float = 0.0


class CountingClient:
    """Wrap DeepSeekClient to count calls and measure per-call latency."""

    def __init__(self, inner: DeepSeekClient, label: str) -> None:
        self._inner = inner
        self._label = label
        self.calls: list[CallRecord] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        started = perf_counter()
        try:
            return self._inner.complete(messages, response_format=response_format)
        finally:
            self.calls.append(
                CallRecord(label=self._label, duration=perf_counter() - started)
            )


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _settings(args: argparse.Namespace) -> LLMSettings:
    return LLMSettings().for_structured_output().model_copy(
        update={"timeout_seconds": args.timeout_seconds, "max_retries": 3}
    )


def _load_handoff() -> V04WritingHandoff:
    handoff_path = SOURCE_RUN / "writing_handoff.json"
    if not handoff_path.is_file():
        raise SystemExit(f"missing input artifact: {handoff_path}")
    return V04WritingHandoff.model_validate(_load(handoff_path))


def _load_body(handoff: V04WritingHandoff) -> BodyDraftPackage:
    """Rebuild the confirmed body from the prior run's writing project.

    This mirrors ``WritingProjectService.assemble_body`` but skips the quality-review
    gate so the final-matter stage can be exercised in isolation from the paragraph
    writing stage.
    """
    project_path = SOURCE_RUN / "writing_project.json"
    if not project_path.is_file():
        raise SystemExit(f"missing input artifact: {project_path}")
    project = V04WritingProject.model_validate(_load(project_path))
    drafts = [state.draft for state in project.sections if state.draft is not None]
    if not drafts:
        raise SystemExit("writing_project.json has no confirmed drafts")
    topic = project.handoff.outline.outline.topic
    markdown = f"# {topic}\n\n" + "\n\n".join(draft.markdown for draft in drafts)
    citations = [citation for draft in drafts for citation in draft.citations]
    source_dois = list(dict.fromkeys(citation.doi for citation in citations))
    return BodyDraftPackage(
        topic=topic,
        markdown=markdown,
        counted_words=sum(draft.counted_words for draft in drafts),
        citations=citations,
        source_dois=source_dois,
    )


def _report(label: str, client: CountingClient) -> None:
    if not client.calls:
        print(f"[{label}] no calls", flush=True)
        return
    total = sum(call.duration for call in client.calls)
    longest = max(call.duration for call in client.calls)
    print(
        f"[{label}] calls={len(client.calls)} total={total:.2f}s longest={longest:.2f}s",
        flush=True,
    )


def run_plan(args: argparse.Namespace, handoff: V04WritingHandoff) -> None:
    client = CountingClient(DeepSeekClient(_settings(args)), label="plan")
    planner = GroundedWritingPlanner(
        client,
        reuse_cache=False,
        max_elapsed_seconds=args.max_elapsed_seconds,
        max_model_calls=args.max_model_calls,
    )
    started = perf_counter()
    plan = planner.plan(handoff)
    elapsed = perf_counter() - started
    print(f"[plan] ok wall={elapsed:.2f}s sections={len(plan.sections)}", flush=True)
    for section in plan.sections:
        print(
            f"  - {section.section_id} paragraphs={len(section.paragraphs)} "
            f"target_words={section.target_words}",
            flush=True,
        )
    _report("plan calls", client)


def run_review(args: argparse.Namespace) -> None:
    plan = GroundedWritingPlan.model_validate(_load(SOURCE_RUN / "writing_plan.json"))
    project = V04WritingProject.model_validate(
        _load(SOURCE_RUN / "writing_project.json")
    )
    client = CountingClient(DeepSeekClient(_settings(args)), label="review")
    reviewer = LLMManuscriptQualityReviewer(client)
    started = perf_counter()
    review = reviewer.review(plan, project)
    elapsed = perf_counter() - started
    blocking = sum(f.severity == "blocking" for f in review.findings)
    warnings = sum(f.severity == "warning" for f in review.findings)
    print(
        f"[review] ok wall={elapsed:.2f}s status={review.review_status} "
        f"findings={len(review.findings)} blocking={blocking} warnings={warnings}",
        flush=True,
    )
    for finding in review.findings:
        print(
            f"  - {finding.section_id} p{finding.paragraph_number} "
            f"[{finding.severity}/{finding.disposition}] {finding.code}: "
            f"{finding.detail[:100]}",
            flush=True,
        )
    _report("review calls", client)


def run_matter(args: argparse.Namespace, handoff: V04WritingHandoff) -> None:
    body = _load_body(handoff)
    client = CountingClient(DeepSeekClient(_settings(args)), label="matter")
    writer = LLMFinalMatterWriter(client)
    started = perf_counter()
    matter = writer.draft(handoff, body)
    elapsed = perf_counter() - started
    fields = [
        name
        for name in ("title", "abstract", "keywords", "conclusion")
        if getattr(matter, name, None)
    ]
    print(f"[matter] ok wall={elapsed:.2f}s fields={fields}", flush=True)
    _report("matter calls", client)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live DeepSeek smoke test for V0.4 planning and final matter."
    )
    parser.add_argument(
        "--stage",
        choices=["plan", "review", "matter", "all"],
        default="all",
        help="which stage to exercise (default: all)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-elapsed-seconds", type=float, default=600.0)
    parser.add_argument("--max-model-calls", type=int, default=6)
    parser.add_argument(
        "--yes",
        dest="dry_run",
        action="store_false",
        help="actually call the DeepSeek API (default is a dry run)",
    )
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()

    handoff = _load_handoff()
    stages = (
        ["plan", "review", "matter"] if args.stage == "all" else [args.stage]
    )

    if args.dry_run:
        print(
            f"[smoke] DRY RUN: would run stage(s) {stages} against real DeepSeek. "
            "Pass --yes to execute.",
            flush=True,
        )
        return

    for stage in stages:
        try:
            if stage == "plan":
                run_plan(args, handoff)
            elif stage == "review":
                run_review(args)
            else:
                run_matter(args, handoff)
        except Exception as exc:
            print(
                f"[{stage}] FAILED {type(exc).__name__}: {str(exc)[:2000]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
