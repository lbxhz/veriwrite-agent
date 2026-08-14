"""Small-scale live DeepSeek smoke test for the V0.4 paragraph writing path.

This script intentionally avoids rerunning planning, literature selection, or the
full document pipeline. It reuses an already-confirmed ``writing_handoff.json`` and
``writing_plan.json`` from a previous formal run, then exercises only:

    * one section's paragraph drafting (concurrently, ``max_workers`` configurable)
    * one section quality review against the real DeepSeek API

Its purpose is to catch real-provider failures (truncation, empty output, schema
violation, rate limits) quickly and to measure wall time and call counts, without
paying for the entire manuscript.

Safety: it never mutates the active Streamlit project and, by default, performs a
dry run that only reports what it would do. Pass ``--yes`` to actually call the API.
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
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.services.grounded_writing import (
    SectionEvidencePacketBuilder,
)
from veriwrite_agent.services.writing_planning import (
    LLMGroundedParagraphWriter,
    ParagraphWritingRuntimeCache,
    PlannedSectionDraftService,
)
from veriwrite_agent.services.writing_quality import (
    LLMSectionQualityReviewer,
    repeated_sentence_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    REPO_ROOT
    / "runtime"
    / "generated_runs"
    / "formal_course_paper_grounded_20260805"
)
SMOKE_ROOT = REPO_ROOT / "runtime" / "smoke_deepseek_writing"


@dataclass
class CallRecord:
    started: float = 0.0
    duration: float = 0.0
    label: str = ""


class CountingClient:
    """Wrap a DeepSeekClient to count calls and measure per-call latency.

    It exposes the same ``complete`` surface as the project's ``LLMClient`` so it can
    be passed straight into ``LLMGroundedParagraphWriter`` and
    ``LLMSectionQualityReviewer`` without changing either service.
    """

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
        record = CallRecord(started=perf_counter(), label=self._label)
        try:
            return self._inner.complete(messages, response_format=response_format)
        finally:
            record.duration = perf_counter() - record.started
            self.calls.append(record)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _select_section(plan: GroundedWritingPlan, index: int):
    sections = plan.sections
    if not sections:
        raise SystemExit("writing plan has no sections")
    if not 0 <= index < len(sections):
        raise SystemExit(
            f"--section-index {index} out of range 0..{len(sections) - 1}"
        )
    return sections[index]


def _limit_paragraphs(section_plan, max_paragraphs: int | None):
    """Optionally shrink a section to its first N paragraphs for a truly tiny run."""
    if max_paragraphs is None:
        return section_plan
    if max_paragraphs < 2:
        raise SystemExit("--max-paragraphs must be at least 2")
    if max_paragraphs >= len(section_plan.paragraphs):
        return section_plan
    subset = section_plan.paragraphs[:max_paragraphs]
    new_target = sum(paragraph.target_words for paragraph in subset)
    return section_plan.model_copy(
        update={"paragraphs": subset, "target_words": new_target}
    )


def _summary(label: str, client: CountingClient) -> str:
    if not client.calls:
        return f"{label}: no calls"
    total = sum(record.duration for record in client.calls)
    longest = max(record.duration for record in client.calls)
    return (
        f"{label}: calls={len(client.calls)} "
        f"total={total:.2f}s longest={longest:.2f}s"
    )


def run(args: argparse.Namespace) -> None:
    handoff_path = SOURCE_RUN / "writing_handoff.json"
    plan_path = SOURCE_RUN / "writing_plan.json"
    for path in (handoff_path, plan_path):
        if not path.is_file():
            raise SystemExit(f"missing input artifact: {path}")

    handoff = V04WritingHandoff.model_validate(_load(handoff_path))
    plan = GroundedWritingPlan.model_validate(_load(plan_path))
    section_plan = _limit_paragraphs(
        _select_section(plan, args.section_index),
        args.max_paragraphs,
    )

    packet = SectionEvidencePacketBuilder().build(
        handoff,
        section_plan.section_id,
    )
    total_paragraphs = len(section_plan.paragraphs)

    print(
        "[smoke] section="
        f"{section_plan.section_id} paragraphs={total_paragraphs} "
        f"target_words={section_plan.target_words} "
        f"workers={args.max_workers} "
        f"review={not args.skip_review}",
        flush=True,
    )

    if args.dry_run:
        print(
            "[smoke] DRY RUN: would call DeepSeek for "
            f"{total_paragraphs} paragraph drafts (max_workers={args.max_workers})"
            + ("" if args.skip_review else " plus 1 section quality review")
            + ". Pass --yes to execute.",
            flush=True,
        )
        return

    settings = LLMSettings().for_structured_output().model_copy(
        update={"timeout_seconds": args.timeout_seconds, "max_retries": 3}
    )
    base_client = DeepSeekClient(settings)

    writer_client = CountingClient(base_client, label="paragraph")
    reviewer_client = CountingClient(base_client, label="review")

    writer = LLMGroundedParagraphWriter(writer_client)
    reviewer = LLMSectionQualityReviewer(reviewer_client)
    cache = ParagraphWritingRuntimeCache(
        SMOKE_ROOT / "paragraph_cache",
        plan_fingerprint=plan.plan_fingerprint,
    )

    progress: list[tuple[int, int, str]] = []

    def on_progress(number: int, total: int, source: str) -> None:
        progress.append((number, total, source))

    started = perf_counter()
    draft = PlannedSectionDraftService().draft(
        packet,
        section_plan,
        writer,
        cache=cache,
        force=args.force,
        on_paragraph_progress=on_progress,
        max_workers=args.max_workers,
    )
    draft_seconds = perf_counter() - started

    sources = [source for _, _, source in progress]
    by_source = {
        source: sources.count(source)
        for source in sorted(set(sources), key=sources.index)
    }
    print(f"[smoke] draft wall={draft_seconds:.2f}s sources={by_source}", flush=True)

    paragraphs_text = [paragraph.text for paragraph in draft.paragraphs]
    repeats = repeated_sentence_pairs(paragraphs_text)
    print(
        f"[smoke] cross-paragraph repeated sentence pairs={len(repeats)}",
        flush=True,
    )

    review_summary = "skipped"
    if not args.skip_review:
        review = reviewer.review(
            section_plan,
            draft,
            packet,
            output_language=plan.output_language,
        )
        blocking = sum(f.severity == "blocking" for f in review.findings)
        warnings = sum(f.severity == "warning" for f in review.findings)
        review_summary = (
            f"findings={len(review.findings)} blocking={blocking} warnings={warnings}"
        )
        print(f"[smoke] review {review_summary}", flush=True)
        for finding in review.findings:
            print(
                f"  - p{finding.paragraph_number} [{finding.severity}] "
                f"{finding.code}: {finding.detail[:120]}",
                flush=True,
            )

    print(_summary("paragraph calls", writer_client), flush=True)
    print(_summary("review calls", reviewer_client), flush=True)
    print(
        json.dumps(
            {
                "section_id": section_plan.section_id,
                "paragraphs": total_paragraphs,
                "draft_wall_seconds": round(draft_seconds, 2),
                "paragraph_sources": by_source,
                "repeated_sentence_pairs": len(repeats),
                "review": review_summary,
                "paragraph_call_count": len(writer_client.calls),
                "review_call_count": len(reviewer_client.calls),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small-scale live DeepSeek smoke test for paragraph writing."
    )
    parser.add_argument(
        "--section-index",
        type=int,
        default=0,
        help="index of the section in the writing plan to exercise (default: 0)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="paragraph draft concurrency (default: 2)",
    )
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=None,
        help="draft only the first N paragraphs of the section (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore existing draft and paragraph cache, forcing fresh generation",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="skip the section quality review call",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="provider timeout per request (default: 180)",
    )
    parser.add_argument(
        "--yes",
        dest="dry_run",
        action="store_false",
        help="actually call the DeepSeek API (default is a dry run)",
    )
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
