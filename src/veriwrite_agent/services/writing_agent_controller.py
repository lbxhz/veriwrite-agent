"""Controller decisions for the V0.4/V0.5 bounded writing Agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from veriwrite_agent.models.agent_runtime import (
    AcquireFullTextAction,
    AgentActionPayload,
    AgentActionRequest,
    ArtifactReference,
    AssembleFinalDeliveryAction,
    ControllerDecision,
    CriticFinding,
    CriticReport,
    RefineLiteratureSearchAction,
    RebuildEvidenceAction,
    RequestUserInputAction,
    ReviseWritingPlanAction,
    RunCriticAction,
    WriteOrReviseSectionsAction,
    action_idempotency_key,
)
from veriwrite_agent.models.final_delivery import FinalPaperPackage
from veriwrite_agent.models.writing_plan import GroundedWritingPlan
from veriwrite_agent.models.writing_quality import ManuscriptEditorialCheckpoint
from veriwrite_agent.services.writing_autopilot import ContinuousWritingResult
from veriwrite_agent.services.writing_quality import (
    EVIDENCE_INTEGRITY_DETERMINISTIC_CODES,
    PLAN_BINDING_DETERMINISTIC_CODES,
)


@dataclass(frozen=True)
class WritingAgentAssessment:
    """One independent critique followed by one executable controller decision."""

    critic: CriticReport
    decision: ControllerDecision


class WritingAgentController:
    """Route writing failures to the earliest stage that can actually fix them."""

    _EVIDENCE_CODES = frozenset({"unsupported_claim", "overstated_evidence"})
    _PLAN_CODES = frozenset(
        {
            "paragraph_repetition",
            "topic_drift",
            "coherence_gap",
            "terminology_inconsistent",
            "academic_style_problem",
            "false_self_attribution",
            "oversized_paragraph",
        }
    )
    _LITERATURE_AUDIT_CODES = frozenset(
        {
            "reference_count_below_minimum",
            "reference_count_below_target",
            "topic_admission_incomplete",
        }
    )

    def assess_section_run(
        self,
        result: ContinuousWritingResult,
        plan: GroundedWritingPlan,
        *,
        plan_reference: ArtifactReference,
        project_reference: ArtifactReference,
    ) -> WritingAgentAssessment:
        """Convert a section executor result into critic evidence and one next action."""

        evaluated = [project_reference.artifact_id]
        if result.stopped_section_id is None:
            report = self._report(
                scope="runtime",
                evaluated_artifact_ids=evaluated,
                outcome="pass",
                findings=[],
            )
            action = self._action(
                RunCriticAction(scope="manuscript"),
                reason_code="body_requires_global_editor",
                rationale=(
                    "All requested chapters passed their local gates; the complete body "
                    "must now be reviewed across section boundaries."
                ),
                input_artifact_ids=[project_reference.artifact_id],
            )
            return WritingAgentAssessment(
                critic=report,
                decision=self._decision(
                    "continue",
                    current_stage="writing",
                    target_stage="editing",
                    report=report,
                    reason_code="section_execution_passed",
                    explanation=(
                        "The section executor passed. Preserve all accepted chapters and "
                        "run the independent full-manuscript editor next."
                    ),
                    next_action=action,
                ),
            )

        if result.stop_code == "evidence_gap" and result.recovery_request is not None:
            return self._evidence_rollback(
                result,
                plan_reference=plan_reference,
                project_reference=project_reference,
            )
        if result.stop_code == "policy_blocked":
            finding = self._finding(
                code="ai_writing_policy_blocked",
                category="requirement",
                responsibility_stage="requirements",
                artifact_id=project_reference.artifact_id,
                location=result.stopped_section_id,
                detail=result.stop_reason or "The confirmed policy blocks AI writing.",
                suggested_action="request_user",
            )
            report = self._report(
                scope="section",
                evaluated_artifact_ids=evaluated,
                outcome="blocked",
                findings=[finding],
            )
            action = self._action(
                RequestUserInputAction(
                    request_type="policy_change",
                    prompt=(
                        "The confirmed course policy blocks AI prose generation. "
                        "Continue only if the user explicitly changes that policy."
                    ),
                    affected_artifact_ids=evaluated,
                ),
                reason_code="confirmed_policy_blocks_generation",
                rationale="A policy boundary cannot be changed by the Agent.",
                input_artifact_ids=evaluated,
                requires_user_approval=True,
            )
            return WritingAgentAssessment(
                critic=report,
                decision=self._decision(
                    "request_user",
                    current_stage="writing",
                    report=report,
                    reason_code="policy_change_requires_user",
                    explanation="Only the user may change the confirmed AI-use policy.",
                    next_action=action,
                ),
            )

        section_id = result.stopped_section_id
        section_state = next(
            state for state in result.project.sections if state.section_id == section_id
        )
        blocking_issues = [
            issue
            for issue in (section_state.draft.issues if section_state.draft else [])
            if issue.severity == "blocking"
        ]
        findings = [
            self._finding(
                code=issue.code,
                category=(
                    "evidence"
                    if issue.code
                    in {
                        *self._EVIDENCE_CODES,
                        *EVIDENCE_INTEGRITY_DETERMINISTIC_CODES,
                    }
                    else "citation"
                    if issue.code in PLAN_BINDING_DETERMINISTIC_CODES
                    else "argument"
                ),
                responsibility_stage=(
                    "evidence"
                    if issue.code in EVIDENCE_INTEGRITY_DETERMINISTIC_CODES
                    else "planning"
                    if issue.code
                    in {*self._EVIDENCE_CODES, *PLAN_BINDING_DETERMINISTIC_CODES}
                    else "writing"
                ),
                artifact_id=project_reference.artifact_id,
                location=(
                    f"{section_id}:{issue.paragraph_number}"
                    if issue.paragraph_number is not None
                    else section_id
                ),
                detail=issue.detail,
                suggested_action=(
                    "rollback"
                    if issue.code in EVIDENCE_INTEGRITY_DETERMINISTIC_CODES
                    else (
                        "revise"
                        if result.stop_code == "review_exhausted"
                        or issue.code in PLAN_BINDING_DETERMINISTIC_CODES
                        else "retry"
                    )
                ),
            )
            for issue in blocking_issues
        ]
        if not findings:
            findings = [
                self._finding(
                    code=f"section_{result.stop_code or 'execution_failed'}",
                    category="runtime",
                    responsibility_stage="writing",
                    artifact_id=project_reference.artifact_id,
                    location=section_id,
                    detail=result.stop_reason or "The section executor stopped.",
                    suggested_action=(
                        "revise" if result.stop_code == "review_exhausted" else "retry"
                    ),
                )
            ]

        if result.stop_code == "deterministic_blocked":
            issue_codes = {issue.code for issue in blocking_issues}
            if issue_codes & EVIDENCE_INTEGRITY_DETERMINISTIC_CODES:
                source_dois = list(
                    dict.fromkeys(
                        [
                            *(
                                citation.doi
                                for citation in (
                                    section_state.draft.citations
                                    if section_state.draft
                                    else []
                                )
                            ),
                            *(
                                doi
                                for paragraph in next(
                                    section
                                    for section in plan.sections
                                    if section.section_id == section_id
                                ).paragraphs
                                for doi in paragraph.source_dois
                            ),
                            *plan.required_source_dois,
                        ]
                    )
                )
                action = self._action(
                    RebuildEvidenceAction(
                        affected_section_ids=[section_id],
                        source_dois=source_dois,
                    ),
                    reason_code="deterministic_evidence_integrity_blocked",
                    rationale=(
                        "The saved evidence dependency is not trusted; rebuild it before "
                        "rewriting prose."
                    ),
                    input_artifact_ids=[
                        plan_reference.artifact_id,
                        project_reference.artifact_id,
                    ],
                )
                report = self._report(
                    scope="section",
                    evaluated_artifact_ids=evaluated,
                    outcome="rollback",
                    findings=findings,
                )
                return WritingAgentAssessment(
                    critic=report,
                    decision=self._decision(
                        "rollback",
                        current_stage="writing",
                        target_stage="evidence",
                        report=report,
                        reason_code="evidence_integrity_requires_rebuild",
                        explanation=(
                            "Preserve accepted chapters and rebuild only the affected "
                            "section's evidence dependency."
                        ),
                        next_action=action,
                    ),
                )
            if issue_codes & PLAN_BINDING_DETERMINISTIC_CODES:
                report = self._report(
                    scope="section",
                    evaluated_artifact_ids=evaluated,
                    outcome="revise",
                    findings=findings,
                )
                action = self._action(
                    ReviseWritingPlanAction(
                        affected_section_ids=[section_id],
                        finding_ids=[finding.finding_id for finding in findings],
                    ),
                    reason_code="deterministic_binding_requires_replanning",
                    rationale=(
                        "The paragraph's locked support assignment is invalid; changing "
                        "wording cannot repair that assignment."
                    ),
                    input_artifact_ids=[
                        plan_reference.artifact_id,
                        project_reference.artifact_id,
                    ],
                )
                return WritingAgentAssessment(
                    critic=report,
                    decision=self._decision(
                        "revise",
                        current_stage="writing",
                        target_stage="planning",
                        report=report,
                        reason_code="support_binding_requires_replanning",
                        explanation=(
                            "Preserve accepted chapters and rebuild only the affected "
                            "paragraph support plan."
                        ),
                        next_action=action,
                    ),
                )

        if result.stop_code == "review_exhausted":
            report = self._report(
                scope="section",
                evaluated_artifact_ids=evaluated,
                outcome="revise",
                findings=findings,
            )
            action = self._action(
                ReviseWritingPlanAction(
                    affected_section_ids=[section_id],
                    finding_ids=[finding.finding_id for finding in findings],
                ),
                reason_code="repeated_review_requires_replanning",
                rationale=(
                    "Repeated paragraph rewrites did not resolve the same critique; "
                    "change the claim and evidence allocation instead of wording alone."
                ),
                input_artifact_ids=[
                    plan_reference.artifact_id,
                    project_reference.artifact_id,
                ],
            )
            return WritingAgentAssessment(
                critic=report,
                decision=self._decision(
                    "revise",
                    current_stage="writing",
                    target_stage="planning",
                    report=report,
                    reason_code="paragraph_retry_budget_exhausted",
                    explanation=(
                        "Preserve accepted chapters and replan only the failed section."
                    ),
                    next_action=action,
                ),
            )

        paragraph_numbers = sorted(
            {
                issue.paragraph_number
                for issue in blocking_issues
                if issue.paragraph_number is not None
            }
        )
        mode = "targeted_revision" if paragraph_numbers else "draft"
        action = self._action(
            WriteOrReviseSectionsAction(
                section_ids=[section_id],
                paragraph_numbers=(
                    {section_id: paragraph_numbers} if paragraph_numbers else {}
                ),
                mode=mode,
            ),
            reason_code=f"{result.stop_code or 'section_failure'}_retry",
            rationale=(
                "Retry only the failed section or its explicitly identified paragraphs."
            ),
            input_artifact_ids=[
                plan_reference.artifact_id,
                project_reference.artifact_id,
            ],
        )
        report = self._report(
            scope="section",
            evaluated_artifact_ids=evaluated,
            outcome="revise",
            findings=findings,
        )
        return WritingAgentAssessment(
            critic=report,
            decision=self._decision(
                "retry",
                current_stage="writing",
                report=report,
                reason_code="bounded_section_retry",
                explanation="The failure is local and can be retried without rollback.",
                next_action=action,
            ),
        )

    def assess_manuscript_editor(
        self,
        checkpoint: ManuscriptEditorialCheckpoint,
        *,
        project_reference: ArtifactReference,
    ) -> WritingAgentAssessment:
        """Decide whether V0.5 advances or reopens only editor-identified paragraphs."""

        findings = [
            self._finding(
                code=finding.code,
                category=(
                    "citation"
                    if finding.code == "false_self_attribution"
                    else "structure"
                ),
                severity=finding.severity,
                responsibility_stage="writing",
                artifact_id=project_reference.artifact_id,
                location=f"{finding.section_id}:{finding.paragraph_number}",
                detail=finding.detail,
                suggested_action=(
                    "revise"
                    if finding.severity == "blocking"
                    or finding.disposition == "targeted_repair"
                    else "report_only"
                ),
            )
            for finding in checkpoint.review.findings
        ]
        if checkpoint.status == "passed":
            report = self._report(
                scope="manuscript",
                evaluated_artifact_ids=[project_reference.artifact_id],
                outcome="pass",
                findings=findings,
            )
            action = self._action(
                AssembleFinalDeliveryAction(),
                reason_code="global_editor_passed",
                rationale="The body passed cross-section editing and can be assembled.",
                input_artifact_ids=[project_reference.artifact_id],
            )
            return WritingAgentAssessment(
                critic=report,
                decision=self._decision(
                    "continue",
                    current_stage="editing",
                    target_stage="delivery",
                    report=report,
                    reason_code="manuscript_editor_passed",
                    explanation="Generate final matter only after global editing passes.",
                    next_action=action,
                ),
            )

        blocking = [finding for finding in findings if finding.severity == "blocking"]
        targets: dict[str, list[int]] = {}
        for finding in blocking:
            if not finding.location or ":" not in finding.location:
                continue
            section_id, paragraph = finding.location.rsplit(":", 1)
            targets.setdefault(section_id, []).append(int(paragraph))
        targets = {
            section_id: sorted(set(numbers))
            for section_id, numbers in targets.items()
        }
        report = self._report(
            scope="manuscript",
            evaluated_artifact_ids=[project_reference.artifact_id],
            outcome="revise",
            findings=findings,
        )
        action = self._action(
            WriteOrReviseSectionsAction(
                section_ids=list(targets),
                paragraph_numbers=targets,
                mode="targeted_revision",
            ),
            reason_code="global_editor_targeted_repair",
            rationale=(
                "Reopen only paragraphs located by the full-manuscript editor and "
                "preserve all other accepted text."
            ),
            input_artifact_ids=[project_reference.artifact_id],
        )
        return WritingAgentAssessment(
            critic=report,
            decision=self._decision(
                "revise",
                current_stage="editing",
                target_stage="writing",
                report=report,
                reason_code="global_editor_found_blockers",
                explanation=(
                    "The manuscript cannot advance, but only editor-located paragraphs "
                    "need to be reopened."
                ),
                next_action=action,
            ),
        )

    def assess_final_package(
        self,
        package: FinalPaperPackage,
        plan: GroundedWritingPlan,
        *,
        package_reference: ArtifactReference,
    ) -> WritingAgentAssessment:
        """Route final audit failures or request the one required user confirmation."""

        findings = [
            self._finding(
                code=issue.code,
                category=(
                    "literature"
                    if issue.code in self._LITERATURE_AUDIT_CODES
                    else "requirement"
                ),
                responsibility_stage=self._final_issue_stage(issue.code),
                artifact_id=package_reference.artifact_id,
                location=issue.requirement_path,
                detail=issue.detail,
                suggested_action=(
                    "rollback" if issue.severity == "blocking" else "report_only"
                ),
                severity=issue.severity,
            )
            for issue in package.audit.issues
        ]
        if package.status in {"ready_for_confirmation", "confirmed"}:
            report = self._report(
                scope="final_delivery",
                evaluated_artifact_ids=[package_reference.artifact_id],
                outcome="pass",
                findings=findings,
            )
            if package.status == "confirmed":
                return WritingAgentAssessment(
                    critic=report,
                    decision=self._decision(
                        "finish",
                        current_stage="delivery",
                        report=report,
                        reason_code="final_package_confirmed",
                        explanation="All gates passed and the user confirmed delivery.",
                    ),
                )
            action = self._action(
                RequestUserInputAction(
                    request_type="final_confirmation",
                    prompt="Review and confirm the final paper before export.",
                    affected_artifact_ids=[package_reference.artifact_id],
                ),
                reason_code="final_confirmation_required",
                rationale="The final artifact requires explicit user confirmation.",
                input_artifact_ids=[package_reference.artifact_id],
                requires_user_approval=True,
            )
            return WritingAgentAssessment(
                critic=report,
                decision=self._decision(
                    "request_user",
                    current_stage="delivery",
                    report=report,
                    reason_code="awaiting_final_confirmation",
                    explanation="No internal blocker remains; wait for final confirmation.",
                    next_action=action,
                ),
            )

        blocking = [finding for finding in findings if finding.severity == "blocking"]
        earliest = min(
            (finding.responsibility_stage for finding in blocking),
            key=(
                "requirements",
                "literature",
                "evidence",
                "planning",
                "writing",
                "editing",
                "delivery",
            ).index,
        )
        report = self._report(
            scope="final_delivery",
            evaluated_artifact_ids=[package_reference.artifact_id],
            outcome="rollback" if earliest != "delivery" else "revise",
            findings=[
                finding.model_copy(
                    update={
                        "suggested_action": (
                            "rollback" if earliest != "delivery" else "revise"
                        )
                    }
                )
                for finding in findings
            ],
        )
        if earliest == "literature":
            payload: AgentActionPayload = RefineLiteratureSearchAction(
                gap_ids=[finding.finding_id for finding in blocking],
                queries=[package.requirement_policy.topic],
                target_additional_count=max(1, len(blocking)),
            )
        elif earliest == "evidence":
            payload = RebuildEvidenceAction(
                affected_section_ids=[
                    section.section_id for section in plan.sections[:20]
                ],
                source_dois=(
                    [reference.doi for reference in package.references[:100]]
                    or plan.required_source_dois[:100]
                ),
            )
        elif earliest in {"planning", "writing", "editing"}:
            payload = ReviseWritingPlanAction(
                affected_section_ids=[section.section_id for section in plan.sections],
                finding_ids=[finding.finding_id for finding in blocking],
            )
        else:
            payload = AssembleFinalDeliveryAction()
        action = self._action(
            payload,
            reason_code="final_audit_requires_repair",
            rationale="Repair the earliest responsible stage and preserve later artifacts.",
            input_artifact_ids=[package_reference.artifact_id],
        )
        return WritingAgentAssessment(
            critic=report,
            decision=self._decision(
                "rollback" if earliest != "delivery" else "revise",
                current_stage="delivery",
                target_stage=earliest if earliest != "delivery" else "delivery",
                report=report,
                reason_code="final_audit_failed",
                explanation=(
                    f"The earliest responsible stage is {earliest}; repair it instead "
                    "of regenerating the complete paper."
                ),
                next_action=action,
            ),
        )

    def _evidence_rollback(
        self,
        result: ContinuousWritingResult,
        *,
        plan_reference: ArtifactReference,
        project_reference: ArtifactReference,
    ) -> WritingAgentAssessment:
        request = result.recovery_request
        assert request is not None
        findings = [
            self._finding(
                code=gap.reason,
                category="evidence",
                responsibility_stage=(
                    "evidence" if gap.missing_full_text_dois else "literature"
                ),
                artifact_id=project_reference.artifact_id,
                location=f"{gap.section_id}:{gap.paragraph_number}",
                detail=gap.detail,
                suggested_action="rollback",
            )
            for gap in request.gaps
        ]
        report = self._report(
            scope="section",
            evaluated_artifact_ids=[project_reference.artifact_id],
            outcome="rollback",
            findings=findings,
        )
        inputs = [plan_reference.artifact_id, project_reference.artifact_id]
        if request.requested_core_dois:
            payload: AgentActionPayload = AcquireFullTextAction(
                dois=request.requested_core_dois
            )
            target_stage = "evidence"
            reason_code = "full_text_evidence_required"
        else:
            queries = list(
                dict.fromkeys(
                    query
                    for values in request.search_queries_by_section.values()
                    for query in values
                )
            )
            payload = RefineLiteratureSearchAction(
                gap_ids=[
                    f"{gap.section_id}_{gap.paragraph_number}" for gap in request.gaps
                ],
                queries=queries,
                target_additional_count=max(2, len(request.gaps) * 2),
            )
            target_stage = "literature"
            reason_code = "replacement_literature_required"
        action = self._action(
            payload,
            reason_code=reason_code,
            rationale=(
                "The planned claim exceeds current evidence permission; recover the "
                "missing evidence before rewriting any prose."
            ),
            input_artifact_ids=inputs,
        )
        return WritingAgentAssessment(
            critic=report,
            decision=self._decision(
                "rollback",
                current_stage="writing",
                target_stage=target_stage,
                report=report,
                reason_code=reason_code,
                explanation=(
                    "Preserve all accepted chapters, repair the evidence dependency, "
                    "then replan and reopen only affected sections."
                ),
                next_action=action,
            ),
        )

    @staticmethod
    def _final_issue_stage(code: str) -> str:
        if code in WritingAgentController._LITERATURE_AUDIT_CODES:
            return "literature"
        if code == "document_identity_mismatch":
            return "evidence"
        if code.startswith(("body_", "citation_", "length_")) or code in {
            "uncited_bibliography_item",
            "unknown_citation_key",
            "reference_count_above_maximum",
        }:
            return "writing"
        return "delivery"

    def _action(
        self,
        payload: AgentActionPayload,
        *,
        reason_code: str,
        rationale: str,
        input_artifact_ids: list[str],
        requires_user_approval: bool = False,
    ) -> AgentActionRequest:
        return AgentActionRequest(
            action_id=self._id("act"),
            requested_by="controller",
            reason_code=reason_code,
            rationale=rationale,
            input_artifact_ids=input_artifact_ids,
            payload=payload,
            idempotency_key=action_idempotency_key(input_artifact_ids, payload),
            requires_user_approval=requires_user_approval,
        )

    def _report(
        self,
        *,
        scope: str,
        evaluated_artifact_ids: list[str],
        outcome: str,
        findings: list[CriticFinding],
    ) -> CriticReport:
        return CriticReport.model_validate(
            {
                "report_id": self._id("crit"),
                "scope": scope,
                "evaluated_artifact_ids": evaluated_artifact_ids,
                "outcome": outcome,
                "findings": findings,
                "evaluator": "veriwrite-writing-controller-v1",
                "rubric_version": "v04-v05-routing.1",
            }
        )

    def _decision(
        self,
        decision_type: str,
        *,
        current_stage: str,
        report: CriticReport,
        reason_code: str,
        explanation: str,
        target_stage: str | None = None,
        next_action: AgentActionRequest | None = None,
    ) -> ControllerDecision:
        return ControllerDecision.model_validate(
            {
                "decision_id": self._id("dec"),
                "decision_type": decision_type,
                "current_stage": current_stage,
                "target_stage": target_stage,
                "based_on_critic_report_ids": [report.report_id],
                "reason_code": reason_code,
                "explanation": explanation,
                "next_action": next_action,
            }
        )

    def _finding(
        self,
        *,
        code: str,
        category: str,
        responsibility_stage: str,
        artifact_id: str,
        detail: str,
        suggested_action: str,
        severity: str = "blocking",
        location: str | None = None,
    ) -> CriticFinding:
        return CriticFinding.model_validate(
            {
                "finding_id": self._id("find"),
                "code": self._normalize_code(code),
                "category": category,
                "severity": severity,
                "responsibility_stage": responsibility_stage,
                "artifact_id": artifact_id,
                "location": location,
                "detail": detail,
                "suggested_action": suggested_action,
            }
        )

    @staticmethod
    def _normalize_code(code: str) -> str:
        normalized = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in code.casefold()
        ).strip("_")
        return normalized[:80] or "unknown_finding"

    @staticmethod
    def _id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:16]}"
