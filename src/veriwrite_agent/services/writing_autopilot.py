"""Recoverable continuous section writing with an independent review loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from veriwrite_agent.models.writing import (
    SectionDraft,
    SectionDraftIssue,
    V04WritingProject,
    WritingSectionState,
)
from veriwrite_agent.models.writing_plan import GroundedWritingPlan, WritingSectionPlan
from veriwrite_agent.services.grounded_writing import (
    SectionEvidencePacketBuilder,
    WritingProjectService,
    count_writing_units,
)
from veriwrite_agent.services.writing_planning import (
    LLMGroundedParagraphWriter,
    ParagraphWritingRuntimeCache,
    PlannedSectionDraftService,
)
from veriwrite_agent.services.writing_quality import (
    LLMSectionQualityReviewer,
    PROSE_REPAIRABLE_SECTION_CODES,
    SECTION_QUALITY_REVIEW_CODES,
    apply_section_quality_review,
    defer_noncritical_section_findings,
    mark_section_quality_review_degraded,
)
from veriwrite_agent.services.writing_evidence_recovery import (
    WritingEvidenceRecoveryRequest,
    WritingEvidenceRecoveryService,
)

AutopilotStage = Literal[
    "generating",
    "reviewing",
    "revising",
    "ready",
    "confirmed",
    "stopped",
]

AutopilotStopCode = Literal[
    "evidence_gap",
    "policy_blocked",
    "generation_failed",
    "review_failed",
    "review_exhausted",
    "deterministic_blocked",
]


@dataclass(frozen=True)
class ContinuousWritingPolicy:
    """Bound token spending while keeping clean chapters moving automatically."""

    max_revision_passes: int = 2
    max_total_review_rounds: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.max_revision_passes <= 3:
            raise ValueError("max_revision_passes must be between zero and three")
        if not 1 <= self.max_total_review_rounds <= 6:
            raise ValueError("max_total_review_rounds must be between one and six")


@dataclass(frozen=True)
class ContinuousWritingEvent:
    section_id: str
    section_title: str
    stage: AutopilotStage
    detail: str
    revision_pass: int = 0


@dataclass(frozen=True)
class ContinuousWritingResult:
    project: V04WritingProject
    events: tuple[ContinuousWritingEvent, ...]
    stopped_section_id: str | None = None
    stop_reason: str | None = None
    stop_code: AutopilotStopCode | None = None
    recovery_request: WritingEvidenceRecoveryRequest | None = None

    @property
    def completed(self) -> bool:
        return self.project.status == "body_complete"


CheckpointCallback = Callable[[V04WritingProject, ContinuousWritingEvent], None]
ParagraphProgressCallback = Callable[[str, str, int, int, str], None]


class ContinuousSectionWritingService:
    """Generate, independently review, repair, and confirm consecutive chapters.

    A chapter advances only after deterministic evidence checks and an independent
    LLM review both pass. Review findings trigger paragraph-level rewrites rather
    than whole-chapter regeneration. Every material state can be checkpointed by
    the caller so a browser refresh does not discard completed work.
    """

    def __init__(
        self,
        *,
        writer: LLMGroundedParagraphWriter,
        reviewer: LLMSectionQualityReviewer,
        cache: ParagraphWritingRuntimeCache | None = None,
        policy: ContinuousWritingPolicy | None = None,
    ) -> None:
        self._writer = writer
        self._reviewer = reviewer
        self._cache = cache
        self._policy = policy or ContinuousWritingPolicy()
        self._drafts = PlannedSectionDraftService()
        self._projects = WritingProjectService()
        self._packets = SectionEvidencePacketBuilder()
        self._evidence_recovery = WritingEvidenceRecoveryService()

    def run(
        self,
        project: V04WritingProject,
        plan: GroundedWritingPlan,
        *,
        confirmed_by: str,
        section_id: str | None = None,
        auto_confirm: bool = True,
        on_checkpoint: CheckpointCallback | None = None,
        on_paragraph_progress: ParagraphProgressCallback | None = None,
    ) -> ContinuousWritingResult:
        if plan.status != "confirmed":
            raise ValueError("continuous writing requires a confirmed writing plan")
        plan_by_id = {section.section_id: section for section in plan.sections}
        project_ids = [section.section_id for section in project.sections]
        if project_ids != list(plan_by_id):
            raise ValueError("writing plan and project section order do not match")
        if section_id is not None and section_id not in plan_by_id:
            raise ValueError("requested section is not in the confirmed writing plan")
        if not auto_confirm and section_id is None:
            raise ValueError("manual writing requires one explicit section_id")

        current = project
        events: list[ContinuousWritingEvent] = []
        section_states = [
            state
            for state in current.sections
            if section_id is None or state.section_id == section_id
        ]
        preflight_result = self._preflight_pending_sections(
            current,
            section_states,
            plan_by_id,
            plan_fingerprint=plan.plan_fingerprint,
            on_checkpoint=on_checkpoint,
        )
        if preflight_result is not None:
            return preflight_result
        for section_state in section_states:
            if section_state.status == "confirmed":
                continue
            section_plan = plan_by_id[section_state.section_id]
            result = self._run_section(
                current,
                section_plan,
                plan_fingerprint=plan.plan_fingerprint,
                output_language=plan.output_language,
                confirmed_by=confirmed_by,
                auto_confirm=auto_confirm,
                on_checkpoint=on_checkpoint,
                on_paragraph_progress=on_paragraph_progress,
            )
            current = result.project
            events.extend(result.events)
            if result.stopped_section_id is not None:
                return ContinuousWritingResult(
                    project=current,
                    events=tuple(events),
                    stopped_section_id=result.stopped_section_id,
                    stop_reason=result.stop_reason,
                    stop_code=result.stop_code,
                    recovery_request=result.recovery_request,
                )
            if not auto_confirm:
                return ContinuousWritingResult(project=current, events=tuple(events))
        return ContinuousWritingResult(project=current, events=tuple(events))

    def _preflight_pending_sections(
        self,
        project: V04WritingProject,
        section_states: list[WritingSectionState],
        plan_by_id: dict[str, WritingSectionPlan],
        *,
        plan_fingerprint: str,
        on_checkpoint: CheckpointCallback | None,
    ) -> ContinuousWritingResult | None:
        """Audit every pending chapter before the first prose-model call.

        The writing plan already contains the evidence assignment for all chapters.
        Discovering shortages lazily, one chapter at a time, can repeatedly interrupt
        the user for restricted PDFs.  This deterministic preflight combines all
        currently visible shortages into one durable recovery request.
        """

        pending = [state for state in section_states if state.status != "confirmed"]
        for state in pending:
            section_plan = plan_by_id[state.section_id]
            packet = self._packets.build(project.handoff, state.section_id)
            if packet.ai_writing_mode == "generation_blocked":
                reason = "; ".join(packet.ai_policy_reasons) or "AI writing is disabled"
                return self._stopped(
                    project,
                    section_plan,
                    reason,
                    on_checkpoint,
                    stop_code="policy_blocked",
                )

        all_gaps = []
        for state in pending:
            section_plan = plan_by_id[state.section_id]
            packet = self._packets.build(project.handoff, state.section_id)
            repair_numbers, _ = _repair_targets(state.draft, section_plan)
            all_gaps.extend(
                self._evidence_recovery.audit_section(
                    section_plan,
                    packet,
                    paragraph_numbers=(repair_numbers or None),
                )
            )
        if not all_gaps:
            return None

        gaps = tuple(all_gaps)
        request = self._evidence_recovery.request(
            plan_fingerprint=plan_fingerprint,
            gaps=gaps,
        )
        first_plan = plan_by_id[gaps[0].section_id]
        return self._stopped(
            project,
            first_plan,
            (
                f"Evidence preflight found {len(gaps)} gap(s) across "
                f"{len(request.affected_section_ids)} pending chapter(s). "
                "The Agent combined their full-text needs into one recovery batch."
            ),
            on_checkpoint,
            stop_code="evidence_gap",
            recovery_request=request,
        )

    def _run_section(
        self,
        project: V04WritingProject,
        section_plan: WritingSectionPlan,
        *,
        plan_fingerprint: str,
        output_language: str,
        confirmed_by: str,
        auto_confirm: bool,
        on_checkpoint: CheckpointCallback | None,
        on_paragraph_progress: ParagraphProgressCallback | None,
    ) -> ContinuousWritingResult:
        section_id = section_plan.section_id
        title = section_plan.title
        state = next(item for item in project.sections if item.section_id == section_id)
        packet = self._packets.build(project.handoff, section_id)
        if packet.ai_writing_mode == "generation_blocked":
            reason = "; ".join(packet.ai_policy_reasons) or "AI writing is disabled"
            return self._stopped(
                project,
                section_plan,
                reason,
                on_checkpoint,
                stop_code="policy_blocked",
            )

        non_prose_blockers = _non_prose_blocking_issues(state.draft)
        if non_prose_blockers:
            return self._stopped(
                project,
                section_plan,
                _deterministic_blocker_detail(non_prose_blockers),
                on_checkpoint,
                stop_code="deterministic_blocked",
            )

        initial_repair_numbers, _ = _repair_targets(state.draft, section_plan)
        evidence_gaps = self._evidence_recovery.audit_section(
            section_plan,
            packet,
            paragraph_numbers=(initial_repair_numbers or None),
        )
        if evidence_gaps:
            recovery = self._evidence_recovery.request(
                plan_fingerprint=plan_fingerprint,
                gaps=evidence_gaps,
            )
            return self._stopped(
                project,
                section_plan,
                (
                    "写作计划要求的比较或细节超过了当前全文证据权限；"
                    "系统将先补齐相关全文证据，再只重写受影响章节。"
                ),
                on_checkpoint,
                stop_code="evidence_gap",
                recovery_request=recovery,
            )

        draft = state.draft
        # Older checkpoints may contain a chapter whose secondary reviewer failed
        # after the deterministic draft gates had already passed.  Treat that saved
        # availability fault exactly like the current degraded-review path instead of
        # spending more calls or stopping a resumable run.
        if draft is not None and draft.quality_review_status == "failed":
            blocking = [issue for issue in draft.issues if issue.severity == "blocking"]
            if not blocking:
                draft = mark_section_quality_review_degraded(
                    draft,
                    "restored reviewer-availability failure",
                )
                project = self._projects.save_draft(project, draft)
        if draft is not None and draft.quality_review_status == "passed":
            blocking = [issue for issue in draft.issues if issue.severity == "blocking"]
            if not blocking:
                if not auto_confirm:
                    ready_event = ContinuousWritingEvent(
                        section_id=section_id,
                        section_title=title,
                        stage="ready",
                        detail="本章已通过自动审计，等待用户阅读并采用。",
                    )
                    self._emit(project, ready_event, on_checkpoint)
                    return ContinuousWritingResult(project=project, events=(ready_event,))
                project = self._projects.confirm_section(
                    project,
                    section_id,
                    confirmed_by=confirmed_by,
                )
                confirmed_event = ContinuousWritingEvent(
                    section_id=section_id,
                    section_title=title,
                    stage="confirmed",
                    detail="现有草稿已通过审稿，无需重写，自动进入下一章。",
                )
                self._emit(project, confirmed_event, on_checkpoint)
                return ContinuousWritingResult(
                    project=project,
                    events=(confirmed_event,),
                )
        if _review_attempts_exhausted(draft, self._policy):
            deferred = defer_noncritical_section_findings(draft)
            if deferred.quality_review_status == "passed":
                project = self._projects.save_draft(project, deferred)
                if not auto_confirm:
                    ready_event = ContinuousWritingEvent(
                        section_id=section_id,
                        section_title=title,
                        stage="ready",
                        detail=(
                            "Non-critical reviewer findings reached the local repair "
                            "limit and were deferred to the manuscript editor."
                        ),
                    )
                    self._emit(project, ready_event, on_checkpoint)
                    return ContinuousWritingResult(
                        project=project,
                        events=(ready_event,),
                    )
                project = self._projects.confirm_section(
                    project,
                    section_id,
                    confirmed_by=confirmed_by,
                )
                confirmed_event = ContinuousWritingEvent(
                    section_id=section_id,
                    section_title=title,
                    stage="confirmed",
                    detail=(
                        "Trust gates passed; remaining non-critical editorial advice "
                        "was deferred to the full-manuscript editor."
                    ),
                )
                self._emit(project, confirmed_event, on_checkpoint)
                return ContinuousWritingResult(
                    project=project,
                    events=(confirmed_event,),
                )
            return self._stopped(
                project,
                section_plan,
                _exhaustion_detail(draft),
                on_checkpoint,
                stop_code="review_exhausted",
            )
        previous_draft = draft
        repair_numbers, revision_instructions = _repair_targets(
            draft,
            section_plan,
        )
        event = ContinuousWritingEvent(
            section_id=section_id,
            section_title=title,
            stage="revising" if repair_numbers else "generating",
            detail=(
                f"正在定点重写 {len(repair_numbers)} 个问题段落。"
                if repair_numbers
                else "正在根据锁定证据生成本章草稿。"
            ),
        )
        self._emit(project, event, on_checkpoint)
        try:
            draft = self._drafts.draft(
                packet,
                section_plan,
                self._writer,
                cache=self._cache,
                existing_draft=draft,
                force_paragraph_numbers=repair_numbers,
                revision_instructions=revision_instructions,
                on_paragraph_progress=(
                    lambda completed, total, source: on_paragraph_progress(
                        section_id,
                        title,
                        completed,
                        total,
                        source,
                    )
                    if on_paragraph_progress is not None
                    else None
                ),
            )
            draft = _carry_review_history(draft, previous_draft)
        except Exception as exc:
            saved = (
                self._cache.completed_count(packet, section_plan)
                if self._cache is not None
                else 0
            )
            return self._stopped(
                project,
                section_plan,
                (
                    f"chapter generation failed: {exc}; local paragraph checkpoints "
                    f"available {saved}/{len(section_plan.paragraphs)}; resume will "
                    "start from the first missing paragraph"
                ),
                on_checkpoint,
                stop_code="generation_failed",
            )
        project = self._projects.save_draft(project, draft)
        self._emit(project, event, on_checkpoint)

        non_prose_blockers = _non_prose_blocking_issues(draft)
        if non_prose_blockers:
            return self._stopped(
                project,
                section_plan,
                _deterministic_blocker_detail(non_prose_blockers),
                on_checkpoint,
                stop_code="deterministic_blocked",
            )

        revision_pass = 0
        while True:
            review_event = ContinuousWritingEvent(
                section_id=section_id,
                section_title=title,
                stage="reviewing",
                detail="独立审稿模型正在检查整章论证、证据强度和主题边界。",
                revision_pass=revision_pass,
            )
            self._emit(project, review_event, on_checkpoint)
            try:
                review = self._reviewer.review(
                    section_plan,
                    draft,
                    packet,
                    output_language=output_language,
                )
                draft = apply_section_quality_review(draft, review)
            except Exception as exc:
                draft = mark_section_quality_review_degraded(draft, exc)
                project = self._projects.save_draft(project, draft)
                self._emit(project, review_event, on_checkpoint)
                break
            project = self._projects.save_draft(project, draft)
            self._emit(project, review_event, on_checkpoint)

            repair_numbers, revision_instructions = _editorial_repair_targets(
                draft,
                section_plan,
            )
            if not repair_numbers:
                break
            if (
                _review_progress_stalled(draft)
                or revision_pass >= self._policy.max_revision_passes
                or draft.quality_review_rounds
                >= self._policy.max_total_review_rounds
            ):
                deferred = defer_noncritical_section_findings(draft)
                if deferred.quality_review_status == "passed":
                    draft = deferred
                    project = self._projects.save_draft(project, draft)
                    self._emit(project, review_event, on_checkpoint)
                    break
                return self._stopped(
                    project,
                    section_plan,
                    _exhaustion_detail(draft),
                    on_checkpoint,
                    stop_code="review_exhausted",
                )
            revision_pass += 1
            revise_event = ContinuousWritingEvent(
                section_id=section_id,
                section_title=title,
                stage="revising",
                detail=(
                    f"仅重写独立审稿标记的 {len(repair_numbers)} 个问题段落。"
                ),
                revision_pass=revision_pass,
            )
            self._emit(project, revise_event, on_checkpoint)
            try:
                previous_draft = draft
                draft = self._drafts.draft(
                    packet,
                    section_plan,
                    self._writer,
                    cache=self._cache,
                    existing_draft=draft,
                    force_paragraph_numbers=repair_numbers,
                    revision_instructions=revision_instructions,
                    on_paragraph_progress=(
                        lambda completed, total, source: on_paragraph_progress(
                            section_id,
                            title,
                            completed,
                            total,
                            source,
                        )
                        if on_paragraph_progress is not None
                        else None
                    ),
                )
                draft = _carry_review_history(draft, previous_draft)
            except Exception as exc:
                saved = (
                    self._cache.completed_count(packet, section_plan)
                    if self._cache is not None
                    else 0
                )
                return self._stopped(
                    project,
                    section_plan,
                    (
                        f"targeted paragraph revision failed: {exc}; local paragraph "
                        f"checkpoints available {saved}/{len(section_plan.paragraphs)}"
                    ),
                    on_checkpoint,
                    stop_code="generation_failed",
                )
            project = self._projects.save_draft(project, draft)
            self._emit(project, revise_event, on_checkpoint)

        if draft.quality_review_status != "passed":
            return self._stopped(
                project,
                section_plan,
                "chapter does not have a passing independent review",
                on_checkpoint,
                stop_code="review_exhausted",
            )
        blocking = [issue for issue in draft.issues if issue.severity == "blocking"]
        if blocking:
            return self._stopped(
                project,
                section_plan,
                f"deterministic audit reports {len(blocking)} blocking issue(s)",
                on_checkpoint,
                stop_code="deterministic_blocked",
            )

        if not auto_confirm:
            ready_event = ContinuousWritingEvent(
                section_id=section_id,
                section_title=title,
                stage="ready",
                detail="本章已通过自动审计，等待用户阅读并采用。",
                revision_pass=revision_pass,
            )
            self._emit(project, ready_event, on_checkpoint)
            return ContinuousWritingResult(project=project, events=(ready_event,))

        project = self._projects.confirm_section(
            project,
            section_id,
            confirmed_by=confirmed_by,
        )
        confirmed_event = ContinuousWritingEvent(
            section_id=section_id,
            section_title=title,
            stage="confirmed",
            detail="证据审计与独立审稿均已通过，自动进入下一章。",
            revision_pass=revision_pass,
        )
        self._emit(project, confirmed_event, on_checkpoint)
        return ContinuousWritingResult(project=project, events=(confirmed_event,))

    @staticmethod
    def _emit(
        project: V04WritingProject,
        event: ContinuousWritingEvent,
        callback: CheckpointCallback | None,
    ) -> None:
        if callback is not None:
            callback(project, event)

    @staticmethod
    def _stopped(
        project: V04WritingProject,
        section_plan: WritingSectionPlan,
        reason: str,
        callback: CheckpointCallback | None,
        *,
        stop_code: AutopilotStopCode,
        recovery_request: WritingEvidenceRecoveryRequest | None = None,
    ) -> ContinuousWritingResult:
        event = ContinuousWritingEvent(
            section_id=section_plan.section_id,
            section_title=section_plan.title,
            stage="stopped",
            detail=reason,
        )
        if callback is not None:
            callback(project, event)
        return ContinuousWritingResult(
            project=project,
            events=(event,),
            stopped_section_id=section_plan.section_id,
            stop_reason=reason,
            stop_code=stop_code,
            recovery_request=recovery_request,
        )


def _repair_targets(
    draft: SectionDraft | None,
    section_plan: WritingSectionPlan,
) -> tuple[set[int], dict[int, str]]:
    if draft is None:
        return set(), {}
    issues = [
        issue
        for issue in draft.issues
        if issue.paragraph_number is not None
        and issue.severity == "blocking"
        and issue.code in PROSE_REPAIRABLE_SECTION_CODES
    ]
    numbers = {issue.paragraph_number for issue in issues if issue.paragraph_number}
    instructions = {
        number: " ".join(
            issue.detail for issue in issues if issue.paragraph_number == number
        )
        for number in numbers
    }
    length_numbers, length_instructions = _length_repair_targets(
        draft,
        section_plan,
    )
    numbers.update(length_numbers)
    for number, instruction in length_instructions.items():
        instructions[number] = " ".join(
            part for part in (instructions.get(number), instruction) if part
        )
    return numbers, instructions


def _editorial_repair_targets(
    draft: SectionDraft,
    section_plan: WritingSectionPlan,
) -> tuple[set[int], dict[int, str]]:
    issues = [
        issue
        for issue in draft.issues
        if issue.paragraph_number is not None
        and issue.severity == "blocking"
        and issue.code in PROSE_REPAIRABLE_SECTION_CODES
    ]
    numbers = {issue.paragraph_number for issue in issues if issue.paragraph_number}
    instructions = {
        number: " ".join(
            issue.detail for issue in issues if issue.paragraph_number == number
        )
        for number in numbers
    }
    length_numbers, length_instructions = _length_repair_targets(
        draft,
        section_plan,
    )
    numbers.update(length_numbers)
    for number, instruction in length_instructions.items():
        instructions[number] = " ".join(
            part for part in (instructions.get(number), instruction) if part
        )
    return numbers, instructions


def _length_repair_targets(
    draft: SectionDraft,
    section_plan: WritingSectionPlan,
) -> tuple[set[int], dict[int, str]]:
    issue_codes = {issue.code for issue in draft.issues}
    low = "word_count_low" in issue_codes
    high = "word_count_high" in issue_codes
    if not low and not high:
        return set(), {}
    actual_by_number = {
        number: count_writing_units(
            paragraph.text,
            counting_policy=section_plan.counting_policy,
        )
        for number, paragraph in enumerate(draft.paragraphs, 1)
    }
    plan_by_number = {
        paragraph.paragraph_number: paragraph for paragraph in section_plan.paragraphs
    }
    if low:
        numbers = {
            number
            for number, actual in actual_by_number.items()
            if actual < int(plan_by_number[number].target_words * 0.75)
        }
        direction = "Expand"
    else:
        numbers = {
            number
            for number, actual in actual_by_number.items()
            if actual > int(plan_by_number[number].target_words * 1.25)
        }
        direction = "Condense"
    if not numbers:
        numbers = set(plan_by_number)
    return numbers, {
        number: (
            f"{direction} this paragraph toward its planned target of "
            f"{plan_by_number[number].target_words} counted units while preserving "
            "the central claim and locked evidence scope."
        )
        for number in numbers
    }


def _carry_review_history(
    draft: SectionDraft,
    previous_draft: SectionDraft | None,
) -> SectionDraft:
    if previous_draft is None or not previous_draft.quality_review_rounds:
        return draft
    return draft.model_copy(
        update={
            "quality_review_rounds": previous_draft.quality_review_rounds,
            "quality_review_history": previous_draft.quality_review_history,
        }
    )


def _review_attempts_exhausted(
    draft: SectionDraft | None,
    policy: ContinuousWritingPolicy,
) -> bool:
    if draft is None:
        return False
    if draft.quality_review_status == "failed":
        return draft.quality_review_rounds >= policy.max_total_review_rounds
    if draft.quality_review_status != "findings":
        return False
    blocking_editorial = any(
        issue.severity == "blocking" and issue.code in SECTION_QUALITY_REVIEW_CODES
        for issue in draft.issues
    )
    return (
        blocking_editorial
        and (
            _review_progress_stalled(draft)
            or draft.quality_review_rounds >= policy.max_total_review_rounds
        )
    )


def _review_progress_stalled(draft: SectionDraft) -> bool:
    """Return true when a rewrite leaves the same blocking finding identities."""

    history = draft.quality_review_history
    if len(history) < 2:
        return False
    previous, current = history[-2:]
    if not current.blocking_signatures:
        return False
    return current.blocking_signatures == previous.blocking_signatures


def _non_prose_blocking_issues(
    draft: SectionDraft | None,
) -> list[SectionDraftIssue]:
    if draft is None:
        return []
    return [
        issue
        for issue in draft.issues
        if issue.severity == "blocking"
        and issue.code not in PROSE_REPAIRABLE_SECTION_CODES
    ]


def _deterministic_blocker_detail(issues: list[SectionDraftIssue]) -> str:
    codes = ", ".join(sorted({issue.code for issue in issues}))
    return (
        "deterministic audit found a non-prose dependency problem "
        f"({codes}); rewriting the same paragraph cannot repair it"
    )


def _exhaustion_detail(draft: SectionDraft | None) -> str:
    if draft is None:
        return "自动审稿未能形成可用草稿。"
    identities = sorted(
        {
            f"第 {issue.paragraph_number} 段 {issue.code}"
            for issue in draft.issues
            if issue.severity == "blocking"
            and issue.code in SECTION_QUALITY_REVIEW_CODES
            and issue.paragraph_number is not None
        }
    )
    affected = "、".join(identities) or "章节级阻塞问题"
    return (
        f"本章已经独立审稿 {draft.quality_review_rounds} 轮，仍重复出现：{affected}。"
        "系统已停止盲目重写；这更可能是写作计划、证据边界或审稿规则不匹配，"
        "需要检查系统约束，而不是继续点击重新生成。"
    )
