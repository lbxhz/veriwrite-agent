"""Diagnose the V0.4 planner's first-attempt contract violation against real DeepSeek.

The planner already retries a section when its first LLM response fails deterministic
validation, so a successful ``smoke_04_e2e`` run masks *why* every section needs two
calls. This script reuses the exact ``GroundedWritingPlanner._plan_section`` path but
records the first raw JSON, then replays the deterministic validation steps offline to
print the precise error (``_short_error``) that triggered the retry, together with a
truncated dump of the offending response.

Safety: dry-run by default. Pass ``--yes`` to call DeepSeek.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.base import ChatMessage
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.writing_handoff import V04WritingHandoff
from veriwrite_agent.models.writing_plan import SectionPlanProposal
from veriwrite_agent.services.grounded_writing import SectionEvidencePacketBuilder
from veriwrite_agent.services.writing_planning import (
    GroundedWritingPlanner,
    WritingPlanError,
    _assign_required_sources_to_problem_paragraphs,
    _compile_section_plan,
    _drop_semantically_misaligned_optional_evidence,
    _paragraph_count,
    _required_source_dois,
    _short_error,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    REPO_ROOT
    / "runtime"
    / "generated_runs"
    / "formal_course_paper_grounded_20260805"
)


class RecordingClient:
    """Record every raw completion while delegating to the real client."""

    def __init__(self, inner: DeepSeekClient) -> None:
        self._inner = inner
        self.raws: list[str] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        raw = self._inner.complete(messages, response_format=response_format)
        self.raws.append(raw)
        return raw


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _replay_first_attempt(
    packet,
    raw: str,
    *,
    evidence_aliases,
    source_aliases,
    paragraph_count: int,
    trace_index: int | None = None,
) -> str:
    """Run the same deterministic validation the planner runs, returning the error."""
    try:
        proposal = SectionPlanProposal.model_validate_json(raw)
        if trace_index is not None:
            _dump_paragraph(proposal.paragraphs, trace_index, "after parse")
        proposal = _assign_required_sources_to_problem_paragraphs(
            packet,
            proposal,
            evidence_aliases=evidence_aliases,
            source_aliases=source_aliases,
            repair_invalid_permissions=True,
        )
        if trace_index is not None:
            _dump_paragraph(proposal.paragraphs, trace_index, "after assign#1")
        proposal = _drop_semantically_misaligned_optional_evidence(
            proposal,
            evidence_aliases=evidence_aliases,
            source_aliases=source_aliases,
        )
        if trace_index is not None:
            _dump_paragraph(proposal.paragraphs, trace_index, "after drop")
        proposal = _assign_required_sources_to_problem_paragraphs(
            packet,
            proposal,
            evidence_aliases=evidence_aliases,
            source_aliases=source_aliases,
            repair_invalid_permissions=True,
        )
        if trace_index is not None:
            _dump_paragraph(proposal.paragraphs, trace_index, "after assign#2")
        _compile_section_plan(
            packet,
            proposal,
            evidence_aliases=evidence_aliases,
            source_aliases=source_aliases,
            expected_paragraph_count=paragraph_count,
        )
        return "PASSED"
    except (ValidationError, WritingPlanError) as exc:
        return _short_error(exc)


def _dump_paragraph(paragraphs, index: int, label: str) -> None:
    if not 0 <= index < len(paragraphs):
        print(f"    [{label}] paragraph {index} OUT OF RANGE (n={len(paragraphs)})", flush=True)
        return
    paragraph = paragraphs[index]
    print(
        f"    [{label}] p{index + 1} role={paragraph.role} "
        f"evidence={paragraph.evidence_refs} sources={paragraph.source_refs}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    handoff_path = SOURCE_RUN / "writing_handoff.json"
    if not handoff_path.is_file():
        raise SystemExit(f"missing input artifact: {handoff_path}")
    handoff = V04WritingHandoff.model_validate(_load(handoff_path))
    required_source_dois = _required_source_dois(handoff)
    builder = SectionEvidencePacketBuilder()
    outline_sections = handoff.outline.outline.sections

    if args.dry_run:
        print(
            f"[diagnose] DRY RUN: would call DeepSeek for "
            f"{len(outline_sections)} section plan(s). Pass --yes to execute.",
            flush=True,
        )
        return

    settings = LLMSettings().for_structured_output().model_copy(
        update={"timeout_seconds": args.timeout_seconds, "max_retries": 3}
    )

    for index, outline_section in enumerate(outline_sections):
        section_id = outline_section.section_id
        packet = builder.build(
            handoff,
            section_id,
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
        paragraph_count = _paragraph_count(packet.target_words)
        evidence_aliases = {
            f"E{n:03d}": item for n, item in enumerate(packet.evidence_items, 1)
        }
        source_aliases = {
            f"S{n:03d}": source for n, source in enumerate(packet.sources, 1)
        }

        client = RecordingClient(DeepSeekClient(settings))
        planner = GroundedWritingPlanner(
            client,
            reuse_cache=False,
            max_elapsed_seconds=args.max_elapsed_seconds,
            max_model_calls=args.max_model_calls,
        )
        try:
            planner._plan_section(packet)
        except Exception as exc:
            print(
                f"[{index}] {section_id}: PLANNER FAILED "
                f"{type(exc).__name__}: {str(exc)[:300]}",
                flush=True,
            )
            continue

        print(
            f"[{index}] {section_id}: calls={len(client.raws)} "
            f"paragraphs={paragraph_count}",
            flush=True,
        )
        if not client.raws:
            print("  (no raw captured)", flush=True)
            continue
        first = client.raws[0]
        verdict = _replay_first_attempt(
            packet,
            first,
            evidence_aliases=evidence_aliases,
            source_aliases=source_aliases,
            paragraph_count=paragraph_count,
            trace_index=args.trace_index,
        )
        print(f"  first-attempt verdict: {verdict}", flush=True)
        if verdict != "PASSED":
            print(f"  first-attempt raw (truncated):\n{first[:1200]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose V0.4 planner first-attempt contract violations."
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-elapsed-seconds", type=float, default=600.0)
    parser.add_argument("--max-model-calls", type=int, default=6)
    parser.add_argument(
        "--trace-index",
        type=int,
        default=None,
        help="0-indexed paragraph to trace through the deterministic pipeline",
    )
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
