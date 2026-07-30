"""Streamlit views that connect the confirmed V0.1 hand-off to V0.2."""

from __future__ import annotations

import json

import streamlit as st
from pydantic import ValidationError

from veriwrite_agent.models.literature_selection import (
    BalancedLiteratureSelection,
    ConfirmedLiteratureSearchBlueprint,
    LiteratureSearchBlueprint,
)
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.services.literature_blueprint_confirmation import (
    LiteratureBlueprintConfirmationService,
)
from veriwrite_agent.ui.literature_workbench import LiteratureWorkbench
from veriwrite_agent.ui.workbench import project_root

LITERATURE_STATE_KEYS = (
    "literature_blueprint_json",
    "literature_blueprint_editor",
    "literature_confirmed_blueprint_json",
    "literature_result_json",
    "literature_ris",
    "literature_verification_json",
    "literature_run_dir",
)


def clear_literature_state() -> None:
    for key in LITERATURE_STATE_KEYS:
        st.session_state.pop(key, None)


def render_literature_console() -> None:
    """Render V0.2 only after V0.1 produced a confirmed requirement contract."""

    if "confirmed_json" not in st.session_state:
        return
    confirmed_requirement = ConfirmedRequirementSpec.model_validate_json(
        st.session_state["confirmed_json"]
    )

    st.divider()
    st.header("V0.2 文献检索与验证控制台")
    st.caption(
        "确认需求 → 生成临时检索蓝图 → 用户确认蓝图 → "
        "Crossref 检索 → RIS/DOI 验证 → 相关性与均衡选择"
    )
    _render_requirement_handoff(confirmed_requirement)

    if "literature_blueprint_json" not in st.session_state:
        st.subheader("4. 生成临时检索蓝图")
        st.info(
            "DeepSeek 只负责把已确认需求拆成主题、研究问题、英文检索词和文献配额；"
            "此操作不会搜索或生成任何论文。"
        )
        if st.button(
            "根据 V0.1 最终需求生成临时检索蓝图",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("正在生成临时检索蓝图…"):
                    blueprint = LiteratureWorkbench.live().plan(
                        confirmed_requirement
                    )
            except Exception as exc:
                st.error(f"检索蓝图生成失败：{exc}")
            else:
                serialized = blueprint.model_dump_json(indent=2)
                st.session_state["literature_blueprint_json"] = serialized
                st.session_state["literature_blueprint_editor"] = serialized
                st.rerun()
        return

    draft = LiteratureSearchBlueprint.model_validate_json(
        st.session_state["literature_blueprint_json"]
    )
    st.subheader("4. 检查并确认临时检索蓝图")
    _render_blueprint_summary(draft)

    if "literature_confirmed_blueprint_json" not in st.session_state:
        with st.form("literature_blueprint_confirmation"):
            edited_json = st.text_area(
                "可编辑蓝图 JSON",
                value=st.session_state.get(
                    "literature_blueprint_editor",
                    draft.model_dump_json(indent=2),
                ),
                height=440,
                help=(
                    "重点检查主题、研究问题、search_queries、target_count、"
                    "年份和 max_candidates。各主题 target_count 之和必须等于 target_total。"
                ),
            )
            confirmed_by = st.text_input(
                "蓝图确认人",
                value=confirmed_requirement.confirmed_by,
            )
            note = st.text_area("蓝图确认说明（可选）")
            accepted = st.checkbox(
                "我已检查主题范围和各主题文献配额，同意开始外部检索。"
            )
            submitted = st.form_submit_button(
                "确认检索蓝图",
                type="primary",
                width="stretch",
            )
        if submitted:
            if not accepted:
                st.error("请先明确确认主题范围和文献配额。")
            else:
                try:
                    edited = LiteratureSearchBlueprint.model_validate_json(
                        edited_json
                    )
                    confirmed_blueprint = (
                        LiteratureBlueprintConfirmationService().confirm(
                            edited,
                            confirmed_by=confirmed_by,
                            note=note or None,
                        )
                    )
                except (ValidationError, ValueError) as exc:
                    st.error(f"蓝图确认失败：{exc}")
                else:
                    st.session_state["literature_blueprint_json"] = (
                        edited.model_dump_json(indent=2)
                    )
                    st.session_state["literature_blueprint_editor"] = (
                        edited.model_dump_json(indent=2)
                    )
                    st.session_state["literature_confirmed_blueprint_json"] = (
                        confirmed_blueprint.model_dump_json(indent=2)
                    )
                    for key in (
                        "literature_result_json",
                        "literature_ris",
                        "literature_verification_json",
                        "literature_run_dir",
                    ):
                        st.session_state.pop(key, None)
                    st.rerun()
        return

    confirmed_blueprint = (
        ConfirmedLiteratureSearchBlueprint.model_validate_json(
            st.session_state["literature_confirmed_blueprint_json"]
        )
    )
    st.success(
        f"检索蓝图已由 {confirmed_blueprint.confirmed_by} 确认；"
        "现在才允许访问 Crossref、DOI.org 和 DeepSeek。"
    )
    controls = st.columns([1, 1])
    controls[0].download_button(
        "下载确认后的检索蓝图",
        st.session_state["literature_confirmed_blueprint_json"],
        file_name="confirmed_literature_search_blueprint.json",
        mime="application/json",
        width="stretch",
    )
    if controls[1].button(
        "撤销确认并修改蓝图",
        width="stretch",
    ):
        st.session_state.pop("literature_confirmed_blueprint_json", None)
        for key in (
            "literature_result_json",
            "literature_ris",
            "literature_verification_json",
            "literature_run_dir",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    st.subheader("5. 执行 V0.2 文献工作流")
    st.caption(
        "默认每个主题获取最终配额约 2 倍的候选；DOI/RIS 最多尝试 3 次。"
        "运行结果按确认蓝图缓存，中断后点击同一按钮可继续。"
    )
    if st.button(
        "开始或继续检索、验证与选择",
        type="primary",
        width="stretch",
    ):
        progress_bar = st.progress(0, text="准备运行 V0.2…")
        status_box = st.empty()
        stage_names = {
            "discovery": "Crossref 分主题检索",
            "verification": "RIS 与 DOI 真实性验证",
            "relevance": "已验证论文相关性评分",
            "complete": "完成",
        }

        def update_progress(
            stage: str,
            current: int,
            total: int,
            message: str,
        ) -> None:
            ratio = 1.0 if total == 0 else min(max(current / total, 0.0), 1.0)
            progress_bar.progress(
                ratio,
                text=f"{stage_names.get(stage, stage)}：{message}",
            )
            status_box.caption(
                f"阶段进度 {current}/{total} · 每一步均写入本地缓存"
            )

        try:
            workbench = LiteratureWorkbench.live(
                pool_multiplier=2,
                doi_max_attempts=3,
            )
            result = workbench.run(
                confirmed_blueprint,
                cache_root=project_root() / "runtime" / "literature_console",
                progress=update_progress,
            )
        except Exception as exc:
            st.error(
                "V0.2 运行中断，已完成阶段仍保存在本地；"
                f"修复问题后可继续。错误：{exc}"
            )
        else:
            st.session_state["literature_result_json"] = result.result_json()
            st.session_state["literature_ris"] = result.ris_text
            st.session_state["literature_verification_json"] = (
                result.verifications.model_dump_json(indent=2)
            )
            st.session_state["literature_run_dir"] = str(result.run_dir)
            st.rerun()

    if "literature_result_json" in st.session_state:
        _render_literature_result()


def _render_requirement_handoff(
    confirmed: ConfirmedRequirementSpec,
) -> None:
    requirement = confirmed.requirement
    with st.expander("查看 V0.1 → V0.2 交接内容", expanded=False):
        columns = st.columns(4)
        columns[0].metric("研究主题", requirement.topic or "未确认")
        columns[1].metric(
            "目标文献",
            requirement.references.target_total
            or requirement.references.minimum_total
            or 50,
        )
        columns[2].metric(
            "近年窗口",
            (
                f"{requirement.references.recent_year_window} 年"
                if requirement.references.recent_year_window
                else "未限制"
            ),
        )
        columns[3].metric("确认人", confirmed.confirmed_by)
        st.json(json.loads(confirmed.model_dump_json()))


def _render_blueprint_summary(
    blueprint: LiteratureSearchBlueprint,
) -> None:
    metrics = st.columns(4)
    metrics[0].metric("主题数", len(blueprint.themes))
    metrics[1].metric("最终目标", blueprint.target_total)
    metrics[2].metric("候选上限", blueprint.max_candidates)
    metrics[3].metric(
        "年份",
        (
            f"{blueprint.year_from or '不限'}–{blueprint.year_to or '不限'}"
        ),
    )
    st.dataframe(
        [
            {
                "主题 ID": theme.theme_id,
                "章节主题": theme.section_title,
                "研究问题": "；".join(theme.research_questions),
                "检索式": "；".join(theme.search_queries),
                "配额": theme.target_count,
                "优先级": theme.priority,
            }
            for theme in blueprint.themes
        ],
        width="stretch",
        hide_index=True,
    )


def _render_literature_result() -> None:
    payload = json.loads(st.session_state["literature_result_json"])
    selection = BalancedLiteratureSelection.model_validate(
        payload["selection"]
    )
    st.subheader("6. V0.2 最终结果")
    metrics = st.columns(5)
    metrics[0].metric("预筛候选", payload["prefiltered_count"])
    metrics[1].metric("真实性通过", payload["verified_count"])
    metrics[2].metric(
        "真实性排除",
        payload["verification_excluded_count"],
    )
    metrics[3].metric("最终入选", len(selection.selected))
    metrics[4].metric("主题缺口", sum(selection.shortages.values()))

    if selection.target_reached:
        st.success("最终文献数量和各主题配额均已达到。")
    else:
        st.warning(
            "当前没有达到全部主题配额。系统不会用其他主题论文静默凑数。"
            "可在撤销蓝图确认后增加关键词、减少限制范围，或将 max_candidates 上调至 500。"
        )
        st.json(selection.shortages)

    cug_unranked = sum(item.cug_tier is None for item in selection.selected)
    norwegian_fallback = sum(
        item.cug_tier is None and item.norwegian_level is not None
        for item in selection.selected
    )
    dual_unranked = sum(
        item.cug_tier is None and item.norwegian_level is None
        for item in selection.selected
    )
    if cug_unranked:
        st.info(
            f"有 {cug_unranked} 篇在所选地大学科目录中未取得唯一等级；"
            f"其中 {norwegian_fallback} 篇由挪威国家目录2025提供补充分级，"
            f"仍有 {dual_unranked} 篇在两个目录中均未取得唯一等级。"
        )
    norwegian_level_zero = sum(
        item.norwegian_level == 0 for item in selection.selected
    )
    if norwegian_level_zero:
        st.warning(
            f"有 {norwegian_level_zero} 篇对应挪威目录 Level 0（2025年未获认可）。"
            "这不推翻 DOI 真实性，但属于较低的期刊质量偏好。"
        )
    score_counts: dict[float, int] = {}
    for item in selection.selected:
        score_counts[item.relevance_score] = (
            score_counts.get(item.relevance_score, 0) + 1
        )
    if score_counts.get(1.0, 0) > len(selection.selected) / 2:
        st.warning(
            "超过一半入选论文的相关性得分为 1.0，评分可能饱和；"
            "当前排序更多依赖主题配额、期刊等级和年份，应在后续人工金标准中校准。"
        )

    st.dataframe(
        [
            {
                "主题": item.theme_id,
                "年份": item.year,
                "地大等级": item.cug_tier or "未分级",
                "挪威2025等级": (
                    f"Level {item.norwegian_level}"
                    if item.norwegian_level is not None
                    else "未分级"
                ),
                "挪威匹配依据": item.norwegian_match_basis or "未启用",
                "相关性": item.relevance_score,
                "题名": item.title,
                "DOI": item.doi,
                "选择理由": "；".join(item.selection_reasons),
            }
            for item in selection.selected
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "分级解释：地大2023是本地课程偏好；挪威目录2025是独立补充证据，"
        "Level 2 为较高层级、Level 1 为获认可的基础层级、Level 0 为未获认可。"
        "两套等级不做伪精确换算。"
    )

    with st.expander("查看分主题检索诊断"):
        st.dataframe(
            [
                {
                    "主题": item["theme_id"],
                    "扫描": item["scanned_count"],
                    "候选": item["eligible_count"],
                    "预筛排除": item["excluded_count"],
                    "达到候选目标": item["target_reached"],
                    "排除原因": json.dumps(
                        item["exclusion_reason_counts"],
                        ensure_ascii=False,
                    ),
                }
                for item in payload["discovery"]
            ],
            width="stretch",
            hide_index=True,
        )
        if payload["verification_exclusion_reason_counts"]:
            st.markdown("**真实性验证排除原因**")
            st.json(payload["verification_exclusion_reason_counts"])
    st.caption(f"本地运行目录：{st.session_state['literature_run_dir']}")

    downloads = st.columns(3)
    downloads[0].download_button(
        "下载最终文献 JSON",
        st.session_state["literature_result_json"],
        file_name="verified_literature_selection.json",
        mime="application/json",
        type="primary",
        width="stretch",
    )
    downloads[1].download_button(
        "下载最终 RIS",
        st.session_state["literature_ris"],
        file_name="verified_literature.ris",
        mime="application/x-research-info-systems",
        width="stretch",
    )
    downloads[2].download_button(
        "下载真实性验证证据",
        st.session_state["literature_verification_json"],
        file_name="literature_verification_evidence.json",
        mime="application/json",
        width="stretch",
    )
