import json
from pathlib import Path

from veriwrite_agent.models.literature_selection import (
    BalancedLiteratureSelection,
    ConfirmedLiteratureSearchBlueprint,
    LiteratureSearchBlueprint,
    LiteratureThemePlan,
)
from veriwrite_agent.models.literature_verification import LiteratureVerificationBatch
from veriwrite_agent.models.requirements import ReferenceRequirement, RequirementSpec
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.services.literature_run_recovery import (
    LiteratureRunRecoveryService,
)
from veriwrite_agent.services.local_project_store import LocalProjectStore
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.ui.mvp_console import (
    autosave_local_project,
    restore_local_project_if_needed,
)


def _confirmed_requirement() -> ConfirmedRequirementSpec:
    return ConfirmedRequirementSpec(
        confirmed_by="student",
        requirement=RequirementSpec(
            document_type="research_direction_literature_review",
            topic="大气遥感",
            topic_source="explicit",
            references=ReferenceRequirement(minimum_total=2),
        ),
    )


def _confirmed_blueprint() -> ConfirmedLiteratureSearchBlueprint:
    requirement = _confirmed_requirement()
    policy = RequirementPolicyCompiler(current_year=2026).compile(requirement)
    return ConfirmedLiteratureSearchBlueprint(
        confirmed_by="student",
        blueprint=LiteratureSearchBlueprint(
            topic="大气遥感",
            discipline="大气科学",
            writing_through_line="发展与挑战",
            target_total=2,
            requirement_policy=policy,
            themes=[
                LiteratureThemePlan(
                    theme_id="history",
                    section_title="发展历史",
                    section_purpose="梳理历史",
                    research_questions=["如何发展？"],
                    primary_keywords=["history"],
                    search_queries=["atmospheric remote sensing history"],
                    target_count=1,
                ),
                LiteratureThemePlan(
                    theme_id="future",
                    section_title="未来方向",
                    section_purpose="梳理未来",
                    research_questions=["未来如何发展？"],
                    primary_keywords=["future"],
                    search_queries=["atmospheric remote sensing future"],
                    target_count=1,
                ),
            ],
        ),
    )


def test_local_autosave_round_trip_survives_a_new_session(tmp_path: Path) -> None:
    store = LocalProjectStore(tmp_path / "active_project.json")
    confirmed = _confirmed_requirement()
    original = {
        "mvp_project_id": "project-1",
        "mvp_project_name": "大气遥感",
        "mvp_navigation": "literature",
        "confirmed_json": confirmed.model_dump_json(),
        "literature_pool_multiplier": 4,
    }

    assert autosave_local_project(original, store) is True
    restored: dict[str, object] = {}
    assert restore_local_project_if_needed(restored, store) is True

    assert restored["mvp_project_id"] == "project-1"
    assert restored["mvp_navigation"] == "literature"
    assert restored["literature_pool_multiplier"] == 4
    ConfirmedRequirementSpec.model_validate_json(restored["confirmed_json"])


def test_local_autosave_does_not_replace_an_active_session(tmp_path: Path) -> None:
    store = LocalProjectStore(tmp_path / "active_project.json")
    store.save(
        json.dumps(
            {
                "schema_version": "mvp-ui-1.0",
                "project_id": "saved",
                "project_name": "saved",
                "active_stage": "overview",
                "state": {"confirmed_json": _confirmed_requirement().model_dump_json()},
            }
        )
    )
    active = {"review_json": "active-review"}

    assert restore_local_project_if_needed(active, store) is False
    assert active == {"review_json": "active-review"}


def test_latest_v02_run_can_rebuild_the_lost_ui_index(tmp_path: Path) -> None:
    confirmed = _confirmed_blueprint()
    selection = BalancedLiteratureSelection(
        blueprint=confirmed.blueprint,
        shortages={"history": 1, "future": 1},
        target_reached=False,
    )
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "confirmed_blueprint.json").write_text(
        confirmed.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (run_dir / "final_result.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "discovery": [],
                "selection": selection.model_dump(mode="json"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "verification_cache.json").write_text(
        LiteratureVerificationBatch().model_dump_json(indent=2),
        encoding="utf-8",
    )
    (run_dir / "selected.ris").write_text("", encoding="utf-8")

    recovered = LiteratureRunRecoveryService().latest(tmp_path)

    assert recovered is not None
    assert recovered.run_id == "run-1"
    assert recovered.selected_count == 0
    assert recovered.target_total == 2
    assert recovered.confirmed_requirement.requirement.topic == "大气遥感"
    assert (
        recovered.executable_policy.requirement_fingerprint
        == confirmed.blueprint.requirement_policy.requirement_fingerprint
    )
    state = recovered.session_state()
    assert state["requirement_recovered_from_executable_policy"] is True
    assert state["literature_run_dir"] == str(run_dir)
