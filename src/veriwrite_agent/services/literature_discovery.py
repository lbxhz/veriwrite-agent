"""Orchestrate search, DOI deduplication, hard filters, and local ranking."""

from __future__ import annotations

from veriwrite_agent.literature.base import (
    InternationalJournalRankingProvider,
    JournalRankingProvider,
    LiteratureSearchProvider,
)
from veriwrite_agent.models.literature_discovery import (
    CandidateDecision,
    LiteratureCandidate,
    LiteratureDiscoveryResult,
    LiteratureSearchPlan,
)
from veriwrite_agent.services.requirement_policy import (
    candidate_source_restriction_reasons,
)


class LiteratureDiscoveryService:
    """Build a hard-filtered candidate pool without claiming identity verification."""

    def __init__(
        self,
        search_provider: LiteratureSearchProvider,
        ranking_provider: JournalRankingProvider,
        international_ranking_provider: InternationalJournalRankingProvider | None = None,
    ) -> None:
        self._search_provider = search_provider
        self._ranking_provider = ranking_provider
        self._international_ranking_provider = international_ranking_provider

    def discover(self, plan: LiteratureSearchPlan) -> LiteratureDiscoveryResult:
        if plan.discipline not in self._ranking_provider.available_disciplines:
            raise ValueError(f"unsupported journal discipline: {plan.discipline}")

        decisions: list[CandidateDecision] = []
        seen_dois: set[str] = set()
        scanned_count = 0
        duplicate_count = 0
        eligible_count = 0

        for candidate in self._search_provider.search(plan):
            if candidate.doi in seen_dois:
                duplicate_count += 1
                continue
            if scanned_count >= plan.max_candidates:
                break
            seen_dois.add(candidate.doi)
            scanned_count += 1

            decision = self._evaluate(candidate, plan)
            decisions.append(decision)
            if decision.status == "eligible":
                eligible_count += 1
                if eligible_count >= plan.target_eligible_count:
                    break

        target_reached = eligible_count >= plan.target_eligible_count
        return LiteratureDiscoveryResult(
            plan=plan,
            scanned_count=scanned_count,
            duplicate_count=duplicate_count,
            decisions=decisions,
            target_reached=target_reached,
            needs_user_confirmation=not target_reached,
        )

    def _evaluate(
        self,
        candidate: LiteratureCandidate,
        plan: LiteratureSearchPlan,
    ) -> CandidateDecision:
        ranking = self._ranking_provider.lookup(
            candidate.journal_title,
            plan.discipline,
        )
        norwegian_ranking = (
            self._international_ranking_provider.lookup(
                candidate.journal_title,
                candidate.issns,
            )
            if self._international_ranking_provider is not None
            else None
        )
        reasons: list[str] = []
        if candidate.source_type != plan.work_type:
            reasons.append("source_type_not_journal_article")
        if plan.year_from is not None and (
            candidate.year is None or candidate.year < plan.year_from
        ):
            reasons.append("publication_year_below_requirement")
        if plan.year_to is not None and (candidate.year is None or candidate.year > plan.year_to):
            reasons.append("publication_year_above_requirement")
        if ranking.status == "not_found" and plan.journal_ranking_policy == "required":
            reasons.append("journal_not_in_cug_2023_catalog")
        elif ranking.status == "ambiguous" and plan.journal_ranking_policy == "required":
            reasons.append("cug_2023_catalog_conflict")
        elif ranking.status == "matched" and ranking.resolved_tier not in plan.accepted_tiers:
            reasons.append("journal_tier_not_accepted")
        if plan.requirement_policy is not None:
            reasons.extend(
                candidate_source_restriction_reasons(
                    plan.requirement_policy,
                    candidate,
                )
            )

        return CandidateDecision(
            status="excluded" if reasons else "eligible",
            candidate=candidate,
            ranking=ranking,
            norwegian_ranking=norwegian_ranking,
            reason_codes=reasons,
        )
