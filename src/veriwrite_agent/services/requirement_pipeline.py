"""Orchestrate rule, LLM, reconciliation, and completeness stages."""

from __future__ import annotations

from typing import Protocol

from veriwrite_agent.models.requirement_workflow import (
    ParserRun,
    ReconciliationResult,
    RequirementReviewPackage,
)
from veriwrite_agent.models.requirements import RequirementSpec
from veriwrite_agent.services.requirement_completeness import (
    RequirementCompletenessChecker,
)
from veriwrite_agent.services.requirement_reconciler import RequirementReconciler


class RequirementParser(Protocol):
    def parse(self, text: str) -> RequirementSpec:
        """Return a validated requirement candidate."""


class RequirementReviewPipeline:
    """Prepare a review package without silently hiding disagreements."""

    def __init__(
        self,
        rule_parser: RequirementParser,
        *,
        llm_parser: RequirementParser | None = None,
        reconciler: RequirementReconciler | None = None,
        completeness_checker: RequirementCompletenessChecker | None = None,
    ) -> None:
        self._rule_parser = rule_parser
        self._llm_parser = llm_parser
        self._reconciler = reconciler or RequirementReconciler()
        self._completeness_checker = (
            completeness_checker or RequirementCompletenessChecker()
        )

    def prepare(self, text: str) -> RequirementReviewPackage:
        rule_spec = self._rule_parser.parse(text)
        rule_run = ParserRun(
            parser_name="rule_based",
            status="succeeded",
            spec=rule_spec,
        )
        parser_runs = [rule_run]
        llm_run: ParserRun | None = None

        if self._llm_parser is None:
            reconciliation = ReconciliationResult(merged_spec=rule_spec)
            parser_mode = "rule_only"
        else:
            parser_mode = "dual"
            try:
                llm_spec = self._llm_parser.parse(text)
            except Exception as exc:
                llm_run = ParserRun(
                    parser_name="llm",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                reconciliation = ReconciliationResult(merged_spec=rule_spec)
            else:
                llm_run = ParserRun(
                    parser_name="llm",
                    status="succeeded",
                    spec=llm_spec,
                )
                reconciliation = self._reconciler.reconcile(rule_spec, llm_spec)
            parser_runs.append(llm_run)

        completeness = self._completeness_checker.check(
            reconciliation.merged_spec,
            conflicts=reconciliation.conflicts,
            parser_runs=parser_runs,
        )
        status = (
            "needs_resolution"
            if completeness.blocking_count
            else "ready_for_confirmation"
        )
        return RequirementReviewPackage(
            parser_mode=parser_mode,
            rule_run=rule_run,
            llm_run=llm_run,
            reconciliation=reconciliation,
            completeness=completeness,
            status=status,
        )
