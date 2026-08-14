"""Offline acceptance test for the saved V0.4 -> V0.5 Agent workflow.

This test deliberately uses the active project's real contracts and historical final-edit
checkpoint, but replaces every model call with deterministic fakes.  It never mutates the
snapshot, runtime checkpoints, PDF vault, or Streamlit session.  A non-zero exit means the
user should not be asked to spend time on a live provider run yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from threading import Lock

from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.writing import V04WritingProject
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.models.writing_quality import ManuscriptEditorialCheckpoint
from veriwrite_agent.services.final_delivery import (
    FinalPaperAssembler,
    FinalPaperDocxExporter,
    LLMFinalMatterWriter,
)
from veriwrite_agent.services.grounded_writing import (
    SectionEvidencePacketBuilder,
    WritingProjectService,
)
from veriwrite_agent.services.writing_autopilot import ContinuousSectionWritingService
from veriwrite_agent.services.writing_evidence_recovery import (
    WritingEvidenceRecoveryService,
)
from veriwrite_agent.services.writing_planning import LLMGroundedParagraphWriter
from veriwrite_agent.services.writing_quality import (
    FullManuscriptEditorialService,
    LLMManuscriptQualityReviewer,
    LLMSectionQualityReviewer,
)
from veriwrite_agent.ui.mvp_console import MvpProjectSnapshot
from veriwrite_agent.ui.writing_console import build_manuscript_editor_repair


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = REPO_ROOT / "runtime" / "mvp_projects" / "active_project.json"


class OfflineParagraphClient:
    """Return distinct, length-compliant CJK text without using network or cache."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = Lock()

    def complete(self, messages, *, response_format=None) -> str:
        payload = json.loads(messages[-1]["content"])
        packet = payload.get("locked_evidence_packet", payload)
        paragraph = packet["paragraph"]
        number = int(paragraph["paragraph_number"])
        target = max(220, int(paragraph["target_words"]))
        seed = int.from_bytes(
            hashlib.sha256(
                f"{packet['section_id']}:{number}".encode("utf-8")
            ).digest()[:4],
            "big",
        )
        generator = random.Random(seed)
        text = "".join(
            chr(0x4E00 + generator.randrange(0x4FFF)) for _ in range(target)
        )
        with self._lock:
            self.calls += 1
        return json.dumps({"text": text}, ensure_ascii=True)


class FailingParagraphClient:
    """Inject a provider outage and expose the number of attempted calls."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, response_format=None) -> str:
        self.calls += 1
        raise ConnectionError("offline injected provider failure")


def _cjk_block(offset: int, length: int) -> str:
    return "".join(chr(0x4E00 + ((offset + index) % 0x4FFF)) for index in range(length))


def _final_matter_payload() -> str:
    # The introduction contains an explicit first/next/finally roadmap.  Every field uses
    # a disjoint character range so deterministic overlap detection cannot pass a copied
    # paragraph merely because the fake text is templated.
    introduction = (
        "\u9996\u5148"
        + _cjk_block(900, 70)
        + "\u5176\u6b21"
        + _cjk_block(1100, 70)
        + "\u6700\u540e"
        + _cjk_block(1300, 100)
    )
    payload = {
        "title": _cjk_block(50, 16),
        "abstract": _cjk_block(200, 330),
        "keywords": [_cjk_block(550, 4), _cjk_block(570, 4), _cjk_block(590, 4)],
        "introduction": introduction,
        "current_status_analysis": (
            _cjk_block(1600, 260) + "\n\n" + _cjk_block(1900, 260)
        ),
        "problems": _cjk_block(2300, 260),
        "technology_trends": _cjk_block(2700, 260),
        "conclusion": _cjk_block(3100, 260),
    }
    return json.dumps(payload, ensure_ascii=True)


def _load_snapshot(path: Path) -> MvpProjectSnapshot:
    if not path.is_file():
        raise RuntimeError(f"snapshot does not exist: {path}")
    return MvpProjectSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def _replay_saved_structural_repair(
    state: dict[str, object],
) -> tuple[GroundedWritingPlan, V04WritingProject, dict[str, tuple[int, ...]]]:
    raw_checkpoint = state.get("mvp_final_repair_checkpoint_json")
    if not isinstance(raw_checkpoint, str):
        plan = GroundedWritingPlan.model_validate_json(
            str(state["v04_writing_plan_json"])
        )
        project = V04WritingProject.model_validate_json(
            str(state["v04_writing_project_json"])
        )
        return plan, project, {}

    checkpoint = json.loads(raw_checkpoint)
    saved = dict(checkpoint["state"])
    raw_editor = saved.get("mvp_manuscript_editor_checkpoint_json")
    if not isinstance(raw_editor, str):
        raise RuntimeError("final repair checkpoint has no manuscript-editor result")
    editor = ManuscriptEditorialCheckpoint.model_validate_json(raw_editor)
    repair = build_manuscript_editor_repair(saved, editor)
    return repair.plan, repair.project, repair.paragraph_numbers


def _assert_executable(
    plan: GroundedWritingPlan,
    project: V04WritingProject,
) -> None:
    packets = [
        SectionEvidencePacketBuilder().build(project.handoff, section.section_id)
        for section in plan.sections
    ]
    errors = WritingEvidenceRecoveryService().validate_resolution(
        plan,
        packets,
        affected_section_ids=[section.section_id for section in plan.sections],
    )
    if errors:
        raise RuntimeError("plan is not executable: " + "; ".join(errors[:8]))


def _assert_bounded_failure_controls(
    plan: GroundedWritingPlan,
    project: V04WritingProject,
) -> None:
    pending = next(
        (section for section in project.sections if section.status != "confirmed"),
        None,
    )
    if pending is None:
        raise RuntimeError("failure-control smoke needs at least one pending section")
    confirmed_before = sum(
        section.status == "confirmed" for section in project.sections
    )

    paused_client = OfflineParagraphClient()
    paused = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(paused_client),
        reviewer=LLMSectionQualityReviewer(FakeLLMClient(json.dumps({"findings": []}))),
    ).run(
        project,
        plan,
        section_id=pending.section_id,
        confirmed_by="offline-smoke",
        should_continue=lambda: False,
    )
    if paused.stop_code != "paused" or paused_client.calls:
        raise RuntimeError(
            "pause control started a model call or returned the wrong stop code: "
            f"code={paused.stop_code} calls={paused_client.calls}"
        )

    failing_client = FailingParagraphClient()
    failed = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(failing_client),
        reviewer=LLMSectionQualityReviewer(FakeLLMClient(json.dumps({"findings": []}))),
        paragraph_workers=1,
    ).run(
        project,
        plan,
        section_id=pending.section_id,
        confirmed_by="offline-smoke",
        should_continue=lambda: True,
    )
    confirmed_after = sum(
        section.status == "confirmed" for section in failed.project.sections
    )
    if failed.stop_code != "generation_failed":
        raise RuntimeError(
            "provider failure did not stop at the generation boundary: "
            f"code={failed.stop_code} reason={failed.stop_reason}"
        )
    if not 1 <= failing_client.calls <= 3:
        raise RuntimeError(
            "provider failure was retried outside the bounded allowance: "
            f"calls={failing_client.calls}"
        )
    if confirmed_after != confirmed_before:
        raise RuntimeError(
            "provider failure changed the number of accepted chapters: "
            f"before={confirmed_before} after={confirmed_after}"
        )
    print(
        "[offline] failure controls ok "
        f"pause_calls={paused_client.calls} provider_calls={failing_client.calls}",
        flush=True,
    )


def run(snapshot_path: Path, *, replay_checkpoint: bool = True) -> None:
    snapshot = _load_snapshot(snapshot_path)
    state = dict(snapshot.state)
    if replay_checkpoint:
        plan, project, targets = _replay_saved_structural_repair(state)
        source_label = "checkpoint replay"
    else:
        plan = GroundedWritingPlan.model_validate_json(
            str(state["v04_writing_plan_json"])
        )
        project = V04WritingProject.model_validate_json(
            str(state["v04_writing_project_json"])
        )
        targets = {}
        source_label = "current state"
    _assert_executable(plan, project)
    confirmed_before = sum(section.status == "confirmed" for section in project.sections)
    print(
        f"[offline] {source_label} ok paragraphs="
        f"{sum(len(section.paragraphs) for section in plan.sections)} "
        f"confirmed={confirmed_before}/{len(project.sections)} targets={targets}",
        flush=True,
    )
    _assert_bounded_failure_controls(plan, project)

    paragraph_client = OfflineParagraphClient()
    section_review_client = FakeLLMClient(json.dumps({"findings": []}))
    result = ContinuousSectionWritingService(
        writer=LLMGroundedParagraphWriter(paragraph_client),
        reviewer=LLMSectionQualityReviewer(section_review_client),
        paragraph_workers=2,
    ).run(
        project,
        plan,
        confirmed_by="offline-smoke",
        auto_confirm=True,
        should_continue=lambda: True,
    )
    if result.stopped_section_id is not None or result.project.status != "body_complete":
        raise RuntimeError(
            "V0.4 did not converge: "
            f"section={result.stopped_section_id} code={result.stop_code} "
            f"reason={result.stop_reason}"
        )
    if any(section.status != "confirmed" for section in result.project.sections):
        raise RuntimeError("V0.4 reported completion with an unconfirmed section")
    print(
        f"[offline] V0.4 ok writer_calls={paragraph_client.calls} "
        f"reviewer_calls={len(section_review_client.calls)}",
        flush=True,
    )

    manuscript_review_client = FakeLLMClient(json.dumps({"findings": []}))
    editor = FullManuscriptEditorialService(
        LLMManuscriptQualityReviewer(manuscript_review_client)
    ).run(plan, result.project)
    if editor.status != "passed" or editor.blocking_count:
        details = "; ".join(
            f"{finding.section_id}:{finding.paragraph_number}:{finding.code}:"
            f"{finding.detail}"
            for finding in editor.review.findings
            if finding.severity == "blocking"
            or finding.disposition == "targeted_repair"
        )
        raise RuntimeError(
            f"V0.5 manuscript editor did not pass: status={editor.status} "
            f"blocking={editor.blocking_count} {details}"
        )
    print(
        f"[offline] V0.5 editor ok calls={len(manuscript_review_client.calls)}",
        flush=True,
    )

    body = WritingProjectService().assemble_body(result.project)
    final_client = FakeLLMClient(_final_matter_payload())
    final_matter = LLMFinalMatterWriter(final_client).draft(project.handoff, body)
    package = FinalPaperAssembler().assemble(
        handoff=project.handoff,
        body=body,
        final_matter=final_matter,
        manuscript_review=editor.review,
    )
    if package.audit.blocking_count or package.status != "ready_for_confirmation":
        details = "; ".join(
            issue.detail for issue in package.audit.issues if issue.severity == "blocking"
        )
        raise RuntimeError(
            f"final release gate failed: status={package.status} {details[:2000]}"
        )
    attestation_codes = [
        issue.code
        for issue in package.audit.issues
        if issue.severity == "warning"
        and issue.code
        in {
            "theme_element_requires_user_review",
            "original_analysis_requires_user_review",
            "reference_tool_usage_requires_attestation",
        }
    ]
    attested = package.model_copy(
        update={"user_review_attestations": list(dict.fromkeys(attestation_codes))}
    )
    confirmed = FinalPaperAssembler().confirm(
        attested,
        confirmed_by="offline-smoke",
    )
    docx = FinalPaperDocxExporter().export(confirmed)
    if not docx.startswith(b"PK"):
        raise RuntimeError("DOCX export did not produce an OOXML package")
    print(
        f"[offline] final delivery ok calls={len(final_client.calls)} "
        f"references={package.audit.reference_count} docx_bytes={len(docx)}",
        flush=True,
    )
    print("[offline] PASS - safe to proceed to a bounded live acceptance run", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a no-network V0.4/V0.5 acceptance test on a saved project."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="MVP snapshot path (default: runtime/mvp_projects/active_project.json)",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Validate the snapshot's active plan/project without replaying final repair.",
    )
    args = parser.parse_args()
    run(args.snapshot.resolve(), replay_checkpoint=not args.current_only)


if __name__ == "__main__":
    main()
