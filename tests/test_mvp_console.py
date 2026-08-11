import json

import pytest
from pydantic import ValidationError

from veriwrite_agent.models.requirements import RequirementSpec
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.ui.mvp_console import (
    MvpProjectSnapshot,
    create_project_snapshot,
    inspect_mvp_status,
)


def test_empty_mvp_project_exposes_first_stage_and_locks_downstream() -> None:
    status = inspect_mvp_status({})

    assert status.progress == 0
    assert status.next_stage_id == "requirements"
    assert [stage.state for stage in status.stages] == [
        "ready",
        "locked",
        "locked",
        "locked",
        "locked",
    ]


def test_confirmed_requirement_unlocks_literature_without_skipping_it() -> None:
    confirmed = ConfirmedRequirementSpec(
        confirmed_by="tester",
        requirement=RequirementSpec(
            document_type="research_direction_literature_review",
            topic="大气遥感",
            topic_source="explicit",
        ),
    )

    status = inspect_mvp_status({"confirmed_json": confirmed.model_dump_json()})

    assert status.stages[0].state == "complete"
    assert status.stages[1].state == "ready"
    assert status.stages[2].state == "locked"
    assert status.next_stage_id == "literature"


def test_invalid_saved_contract_is_visible_as_a_blocker() -> None:
    status = inspect_mvp_status({"confirmed_json": "{}"})

    assert status.stages[0].state == "blocked"
    assert status.stages[0].blockers
    assert status.stages[1].state == "locked"


def test_project_snapshot_only_exports_whitelisted_json_state() -> None:
    snapshot = create_project_snapshot(
        {
            "mvp_project_id": "project-1",
            "mvp_project_name": "真实案例",
            "mvp_navigation": "requirements",
            "extracted_text": "课程要求",
            "extraction_warnings": ("warning",),
            "DEEPSEEK_API_KEY": "must-not-leak",
            "uploaded_file": object(),
        }
    )

    payload = json.loads(snapshot.model_dump_json())
    assert payload["state"] == {
        "extracted_text": "课程要求",
        "extraction_warnings": ["warning"],
    }
    assert "DEEPSEEK_API_KEY" not in snapshot.model_dump_json()


def test_project_snapshot_normalizes_dynamic_navigation_label() -> None:
    snapshot = create_project_snapshot(
        {
            "mvp_project_id": "project-1",
            "mvp_project_name": "快速测试",
            "mvp_navigation": "◐ V0.4 逐章写作",
        }
    )

    assert snapshot.active_stage == "writing"


def test_project_snapshot_rejects_unknown_state_fields() -> None:
    with pytest.raises(ValidationError, match="未知状态字段"):
        MvpProjectSnapshot(
            project_id="project-1",
            project_name="invalid",
            state={"secret": "value"},
        )


def test_independent_evaluation_is_a_valid_navigation_target() -> None:
    snapshot = create_project_snapshot(
        {
            "mvp_project_id": "project-1",
            "mvp_project_name": "外部论文评测",
            "mvp_navigation": "evaluation",
        }
    )

    assert snapshot.active_stage == "evaluation"


def test_active_agent_recovery_keeps_writing_visible_while_evidence_is_internal() -> None:
    recovery = {
        "schema_version": "0.4-evidence-recovery.0",
        "status": "pending_search",
        "source_plan_fingerprint": "a" * 64,
        "affected_section_ids": ["methods"],
        "gaps": [
            {
                "section_id": "methods",
                "section_title": "方法比较",
                "paragraph_number": 1,
                "reason": "comparison_requires_full_text",
                "claim_focus": "比较两类方法",
                "central_question": "两类方法有何差异？",
                "search_queries": ["atmospheric retrieval comparison"],
                "detail": "需要第二篇可追溯全文。",
            }
        ],
    }

    status = inspect_mvp_status(
        {
            "v04_evidence_recovery_json": json.dumps(recovery),
        }
    )

    assert status.stages[3].state == "in_progress"
    assert "内部回退" in status.stages[3].summary
    assert status.next_stage_id == "requirements"
