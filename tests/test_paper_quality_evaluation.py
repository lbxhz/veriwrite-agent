from types import SimpleNamespace

from veriwrite_agent.models.paper_quality import PaperQualityScorecard
from veriwrite_agent.models.writing import SectionDraftIssue
from veriwrite_agent.services.paper_quality_evaluation import (
    PaperQualityEvaluationService,
)


def _fixture(*, degraded: bool = False):
    issues = []
    if degraded:
        issues = [
            SectionDraftIssue(
                code="unsupported_claim",
                severity="blocking",
                paragraph_number=1,
                detail="The claim is not supported by the locked evidence.",
            ),
            SectionDraftIssue(
                code="topic_drift",
                severity="warning",
                paragraph_number=2,
                detail="The paragraph leaves the task boundary.",
            ),
            SectionDraftIssue(
                code="language_mismatch",
                severity="warning",
                paragraph_number=3,
                detail="The paragraph language does not match the requirement.",
            ),
        ]
    paragraphs = [
        SimpleNamespace(evidence_card_ids=["card-1"], source_dois=["10.1000/a"]),
        SimpleNamespace(evidence_card_ids=["card-2"], source_dois=["10.1000/b"]),
    ]
    citations = [
        SimpleNamespace(
            doi="10.1000/a",
            evidence_card_ids=["card-1"],
            page_numbers=[1],
        ),
        SimpleNamespace(
            doi="10.1000/b",
            evidence_card_ids=["card-2"],
            page_numbers=[] if degraded else [2],
        ),
    ]
    draft = SimpleNamespace(paragraphs=paragraphs, citations=citations, issues=issues)
    project = SimpleNamespace(
        status="drafting" if degraded else "body_complete",
        sections=[SimpleNamespace(section_id="section_1", draft=draft)],
        handoff=SimpleNamespace(
            evidence_library=SimpleNamespace(
                records=[
                    SimpleNamespace(doi="10.1000/a"),
                    SimpleNamespace(doi="10.1000/b"),
                ]
            )
        ),
    )
    final_issues = []
    if degraded:
        final_issues = [
            SimpleNamespace(
                code="required_section_missing",
                severity="blocking",
            )
        ]
    package = SimpleNamespace(
        markdown="# Paper\n\nDegraded" if degraded else "# Paper\n\nClean",
        audit=SimpleNamespace(
            blocking_count=1 if degraded else 0,
            counted_units=1000,
            reference_count=2,
            issues=final_issues,
        ),
        references=[
            SimpleNamespace(doi="10.1000/a"),
            SimpleNamespace(doi="10.1000/b"),
        ],
    )
    plan = SimpleNamespace(
        sections=[
            SimpleNamespace(
                paragraphs=[
                    SimpleNamespace(argument_move="compare_studies"),
                    SimpleNamespace(argument_move="synthesize_consensus"),
                ]
            )
        ]
    )
    return package, project, plan


def test_clean_paper_receives_reproducible_six_dimension_scorecard() -> None:
    package, project, plan = _fixture()

    scorecard = PaperQualityEvaluationService().evaluate(package, project, plan)

    assert scorecard.release_gate == "passed"
    assert scorecard.overall_score == 100
    assert scorecard.grade == "excellent"
    assert len(scorecard.metrics) == 6
    assert sum(metric.weight for metric in scorecard.metrics) == 1


def test_blockers_remain_a_hard_gate_and_version_delta_is_visible() -> None:
    service = PaperQualityEvaluationService()
    clean = service.evaluate(*_fixture())
    degraded = service.evaluate(*_fixture(degraded=True))

    comparison = service.compare(clean, degraded)

    assert degraded.release_gate == "blocked"
    assert "writing:body_incomplete" in degraded.blocking_issues
    assert degraded.overall_score < clean.overall_score
    assert comparison.overall_delta < 0
    assert "evidence_traceability" in comparison.regressed_metrics
    assert "topic_relevance" in comparison.regressed_metrics
    assert "presentation_quality" in comparison.regressed_metrics


def test_scorecard_json_round_trip_keeps_the_evaluation_contract() -> None:
    scorecard = PaperQualityEvaluationService().evaluate(*_fixture())

    restored = PaperQualityScorecard.model_validate_json(
        scorecard.model_dump_json()
    )

    assert restored == scorecard


def test_metadata_only_background_citations_do_not_reduce_traceability() -> None:
    package, project, plan = _fixture()
    project.sections[0].draft.citations.append(
        SimpleNamespace(
            doi="10.1000/a",
            evidence_card_ids=[],
            page_numbers=[],
        )
    )

    scorecard = PaperQualityEvaluationService().evaluate(package, project, plan)

    metric = next(
        item for item in scorecard.metrics if item.code == "evidence_traceability"
    )
    assert metric.score == 100
    assert "背景引用 1 条" in metric.basis[-1]
