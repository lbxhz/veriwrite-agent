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
from veriwrite_agent.models.executable_policy import ExecutableRequirementPolicy
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.services.literature_blueprint_confirmation import (
    LiteratureBlueprintConfirmationService,
)
from veriwrite_agent.services.literature_shortage_recovery import (
    LiteratureShortageRecoveryService,
)
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.ui.literature_workbench import LiteratureWorkbench
from veriwrite_agent.ui.evidence_console import (
    PDF_STATE_KEYS,
    render_pdf_acquisition_console,
)
from veriwrite_agent.ui.workbench import project_root

LITERATURE_STATE_KEYS = (
    "literature_blueprint_json",
    "literature_blueprint_editor",
    "literature_confirmed_blueprint_json",
    "literature_result_json",
    "literature_ris",
    "literature_verification_json",
    "literature_run_dir",
    "literature_pool_multiplier",
    *PDF_STATE_KEYS,
    "mvp_final_matter_json",
    "mvp_final_paper_json",
    "mvp_ai_declaration",
    "mvp_final_repair_checkpoint_json",
    "mvp_final_semantic_review_attestation",
    "mvp_final_repair_auto_suppressed_id",
    "v04_selected_section",
    "v04_selected_section_request",
)

LITERATURE_DERIVED_STATE_KEYS = (
    "literature_result_json",
    "literature_ris",
    "literature_verification_json",
    "literature_run_dir",
    *PDF_STATE_KEYS,
    "mvp_final_matter_json",
    "mvp_final_paper_json",
    "mvp_ai_declaration",
    "mvp_final_semantic_review_attestation",
    "mvp_final_repair_auto_suppressed_id",
)


def clear_literature_state() -> None:
    for key in LITERATURE_STATE_KEYS:
        st.session_state.pop(key, None)


def _clear_literature_derived_state() -> None:
    for key in LITERATURE_DERIVED_STATE_KEYS:
        st.session_state.pop(key, None)


def render_literature_console(*, include_downstream: bool = True) -> None:
    """Render V0.2 only after V0.1 produced a confirmed requirement contract."""

    if "confirmed_json" not in st.session_state:
        return
    confirmed_requirement = ConfirmedRequirementSpec.model_validate_json(
        st.session_state["confirmed_json"]
    )

    st.divider()
    st.header("V0.2 查找并验证真实文献")
    st.caption(
        "先看检索方案，再一次完成 Crossref 检索、DOI/RIS 真实性验证和均衡选择。"
    )
    _render_requirement_handoff(confirmed_requirement)

    completed_payload = st.session_state.get("literature_result_json")
    if completed_payload:
        payload = json.loads(completed_payload)
        completed_selection = BalancedLiteratureSelection.model_validate(payload["selection"])
        if _selection_requires_admission_refresh(completed_selection):
            st.error(
                "这批文献来自旧版“真实性验证即准入”流程，尚未经过主题边界和用途审查，"
                "不能继续作为 V0.3/V0.4 的写作来源。"
            )
            st.caption(
                "重新执行会保留 runtime 中的原检索/PDF文件，但清除当前页面的下游派生状态；"
                "新流程会逐篇给出保留/排除、支撑论点、适用章节和使用边界。"
            )
            if st.button(
                "按立题卡重新执行文献准入",
                type="primary",
                width="stretch",
                key="refresh_literature_admission",
            ):
                clear_literature_state()
                st.session_state["mvp_flash"] = (
                    "旧文献池未被删除；正在按立题卡重新生成检索方案和文献准入表。"
                )
                st.rerun()
            return
        if completed_selection.target_reached and not completed_selection.policy_issues:
            _render_completed_literature_result(
                completed_selection,
                payload,
                include_downstream=include_downstream,
            )
            return

    if "literature_blueprint_json" not in st.session_state:
        st.subheader("生成检索方案")
        st.info(
            "DeepSeek 会把课程要求拆成主题、研究问题、英文检索词和文献配额；"
            "这一步不会联网搜索论文。"
        )
        if st.button(
            "生成检索方案",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("正在生成临时检索蓝图…"):
                    blueprint = LiteratureWorkbench.live().plan(confirmed_requirement)
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
    st.subheader("检索方案")
    _render_blueprint_summary(draft)

    if "literature_confirmed_blueprint_json" not in st.session_state:
        st.caption("如果主题和配额合理，直接开始；只有确实需要时才编辑底层 JSON。")
        st.session_state.setdefault(
            "literature_blueprint_editor",
            draft.model_dump_json(indent=2),
        )
        with st.expander("高级：编辑检索 JSON"):
            edited_json = st.text_area(
                "可编辑蓝图 JSON",
                key="literature_blueprint_editor",
                height=440,
                help=(
                    "重点检查主题、研究问题、search_queries、target_count、"
                    "年份和 max_candidates。各主题 target_count 之和必须等于 target_total。"
                ),
            )
        submitted = st.button(
            "采用此方案并开始检索",
            type="primary",
            width="stretch",
        )
        if submitted:
            try:
                edited = LiteratureSearchBlueprint.model_validate_json(edited_json)
                confirmed_blueprint = LiteratureBlueprintConfirmationService().confirm(
                    edited,
                    confirmed_by=confirmed_requirement.confirmed_by,
                    expected_policy=draft.requirement_policy,
                )
            except (ValidationError, ValueError) as exc:
                st.error(f"检索方案无效：{exc}")
            else:
                st.session_state["literature_blueprint_json"] = edited.model_dump_json(indent=2)
                st.session_state["literature_confirmed_blueprint_json"] = (
                    confirmed_blueprint.model_dump_json(indent=2)
                )
                _clear_literature_derived_state()
                st.session_state["literature_auto_run_requested"] = True
                st.rerun()
        return

    confirmed_blueprint = ConfirmedLiteratureSearchBlueprint.model_validate_json(
        st.session_state["literature_confirmed_blueprint_json"]
    )
    st.success("检索方案已锁定；检索只会使用这组主题、配额和年份限制。")
    pool_multiplier = int(st.session_state.get("literature_pool_multiplier", 2))
    st.caption(
        f"当前每个主题获取最终配额约 {pool_multiplier} 倍的候选；"
        "运行结果自动缓存，中断后可继续。"
    )
    with st.expander("方案导出与高级检索设置"):
        controls = st.columns([1, 1])
        controls[0].download_button(
            "下载检索方案",
            st.session_state["literature_confirmed_blueprint_json"],
            file_name="confirmed_literature_search_blueprint.json",
            mime="application/json",
            width="stretch",
        )
        if controls[1].button("修改主题或检索词", width="stretch"):
            st.session_state.pop("literature_confirmed_blueprint_json", None)
            _clear_literature_derived_state()
            st.rerun()
        _render_retrieval_adjustment_controls(confirmed_blueprint)

    st.subheader("检索真实文献")
    auto_run_requested = st.session_state.pop("literature_auto_run_requested", False)
    if st.button(
        "开始或继续检索",
        type="primary",
        width="stretch",
    ) or auto_run_requested:
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
            status_box.caption(f"阶段进度 {current}/{total} · 每一步均写入本地缓存")

        try:
            workbench = LiteratureWorkbench.live(
                pool_multiplier=pool_multiplier,
                doi_max_attempts=3,
            )
            result = workbench.run(
                confirmed_blueprint,
                cache_root=project_root() / "runtime" / "literature_console",
                progress=update_progress,
            )
        except Exception as exc:
            st.error(f"V0.2 运行中断，已完成阶段仍保存在本地；修复问题后可继续。错误：{exc}")
        else:
            st.session_state["literature_result_json"] = result.result_json()
            st.session_state["literature_ris"] = result.ris_text
            st.session_state["literature_verification_json"] = result.verifications.model_dump_json(
                indent=2
            )
            st.session_state["literature_run_dir"] = str(result.run_dir)
            st.rerun()

    if "literature_result_json" in st.session_state:
        _render_literature_result(include_downstream=include_downstream)


def _render_requirement_handoff(
    confirmed: ConfirmedRequirementSpec,
) -> None:
    requirement = confirmed.requirement
    recovered_policy_json = st.session_state.get("recovered_executable_policy_json")
    policy = (
        ExecutableRequirementPolicy.model_validate_json(recovered_policy_json)
        if recovered_policy_json
        else RequirementPolicyCompiler().compile(confirmed)
    )
    with st.expander("查看 V0.1 → V0.2 交接内容", expanded=False):
        if recovered_policy_json:
            st.warning(
                "本页由本地V0.2运行产物恢复。下游继续使用当时冻结的"
                "ExecutableRequirementPolicy；V0.1原始来源证据未被伪造。"
            )
        columns = st.columns(4)
        columns[0].metric("研究主题", requirement.topic or "未确认")
        columns[1].metric(
            "目标文献",
            requirement.references.target_total or requirement.references.minimum_total or 50,
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
        st.markdown("**下游实际执行的 ExecutableRequirementPolicy**")
        st.json(json.loads(policy.model_dump_json()))
        if policy.unresolved_requirements:
            st.warning(
                "以下要求已传到下游，但当前能力不能完全自动执行；"
                "最终交付前必须补充数据或由用户明确处理。"
            )
            st.code("\n".join(policy.unresolved_requirements))


def _render_completed_literature_result(
    selection: BalancedLiteratureSelection,
    payload: dict,
    *,
    include_downstream: bool,
) -> None:
    st.success("文献数量、主题配额和 V0.1 硬性规则均已满足。")
    metrics = st.columns(4)
    metrics[0].metric("真实性通过", payload["verified_count"])
    metrics[1].metric("最终入选", len(selection.selected))
    metrics[2].metric("覆盖主题", len({item.theme_id for item in selection.selected}))
    metrics[3].metric("外文文献", sum(item.is_foreign for item in selection.selected))

    if not include_downstream:
        if st.button("继续获取核心论文全文", type="primary", width="stretch"):
            st.session_state["mvp_navigation_request"] = "evidence"
            st.rerun()

    with st.expander("查看文献清单与验证记录"):
        st.dataframe(
            [
                {
                    "主题": item.theme_id,
                    "年份": item.year,
                    "相关性": item.relevance_score,
                    "期刊": item.journal or "—",
                    "题名": item.title,
                    "DOI": item.doi,
                }
                for item in selection.selected
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(f"本地运行目录：{st.session_state['literature_run_dir']}")
        downloads = st.columns(3)
        downloads[0].download_button(
            "文献 JSON",
            st.session_state["literature_result_json"],
            file_name="verified_literature_selection.json",
            mime="application/json",
            width="stretch",
        )
        downloads[1].download_button(
            "文献 RIS",
            st.session_state["literature_ris"],
            file_name="verified_literature.ris",
            mime="application/x-research-info-systems",
            width="stretch",
        )
        downloads[2].download_button(
            "真实性证据",
            st.session_state["literature_verification_json"],
            file_name="literature_verification_evidence.json",
            mime="application/json",
            width="stretch",
        )
        st.divider()
        restart_accepted = st.checkbox(
            "我知道重新检索会清除 V0.2 之后的页面状态",
            key="literature_restart_accepted",
        )
        if st.button(
            "重新生成方案并检索",
            disabled=not restart_accepted,
            width="stretch",
        ):
            clear_literature_state()
            st.rerun()

    if include_downstream:
        render_pdf_acquisition_console(selection)


def _render_blueprint_summary(
    blueprint: LiteratureSearchBlueprint,
) -> None:
    metrics = st.columns(4)
    metrics[0].metric("主题数", len(blueprint.themes))
    metrics[1].metric("最终目标", blueprint.target_total)
    metrics[2].metric("候选上限", blueprint.max_candidates)
    metrics[3].metric(
        "年份",
        (f"{blueprint.year_from or '不限'}–{blueprint.year_to or '不限'}"),
    )
    boundary = blueprint.topic_boundary
    st.markdown("**立题卡与主题边界**")
    st.dataframe(
        [
            {
                "中心问题": boundary.central_question or "待补充",
                "纳入对象": "；".join(boundary.included_objects) or "待补充",
                "明确排除": "；".join(boundary.excluded_objects) or "无明确项",
                "仅可作支撑": "；".join(boundary.contextual_only_topics) or "无",
                "边界来源": boundary.origin,
            }
        ],
        width="stretch",
        hide_index=True,
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


def _selection_requires_admission_refresh(
    selection: BalancedLiteratureSelection,
) -> bool:
    if not selection.blueprint.topic_boundary.is_actionable:
        return True
    return any(
        item.admission_status != "admit"
        or item.centrality not in {"central", "supporting"}
        or not item.supported_claim
        or not item.suitable_section_id
        for item in selection.selected
    )


def _render_literature_result(*, include_downstream: bool = True) -> None:
    payload = json.loads(st.session_state["literature_result_json"])
    selection = BalancedLiteratureSelection.model_validate(payload["selection"])
    st.subheader("文献选择结果")
    metrics = st.columns(4)
    metrics[0].metric("预筛候选", payload["prefiltered_count"])
    metrics[1].metric("真实性通过", payload["verified_count"])
    metrics[2].metric("最终入选", len(selection.selected))
    metrics[3].metric("主题缺口", sum(selection.shortages.values()))

    if selection.target_reached:
        st.success("最终文献数量和各主题配额均已达到。")
    else:
        st.warning(
            "当前没有达到全部主题配额。系统不会用其他主题论文静默凑数。"
            "可在上方高级检索设置中增加候选扫描上限或候选池倍率后补搜，"
            "也可以返回蓝图修改关键词和限制。"
        )
        theme_targets = {
            theme.theme_id: theme.target_count for theme in selection.blueprint.themes
        }
        selected_counts = {theme_id: 0 for theme_id in theme_targets}
        for item in selection.selected:
            selected_counts[item.theme_id] += 1
        st.dataframe(
            [
                {
                    "缺口主题": theme_id,
                    "目标": theme_targets[theme_id],
                    "已入选": selected_counts[theme_id],
                    "还缺": shortage,
                }
                for theme_id, shortage in selection.shortages.items()
            ],
            width="stretch",
            hide_index=True,
        )
        minimum_total = None
        policy = selection.blueprint.requirement_policy
        if policy is not None:
            minimum_total = policy.references.minimum_total
        if minimum_total is not None and len(selection.selected) < minimum_total:
            st.error(
                f"V0.1 规定至少 {minimum_total} 篇；当前只有 {len(selection.selected)} 篇，"
                "这是硬性要求，不能通过‘接受当前结果’绕过。"
            )
        actions = st.columns(2)
        if actions[0].button(
            "扩大候选池并自动补搜",
            type="primary",
            width="stretch",
            help="提高检索池容量，但不降低主题配额、相关性阈值或V0.1硬性要求。",
        ):
            try:
                confirmed = ConfirmedLiteratureSearchBlueprint.model_validate_json(
                    st.session_state["literature_confirmed_blueprint_json"]
                )
                recovery = LiteratureShortageRecoveryService().expand_candidate_pool(
                    confirmed,
                    current_pool_multiplier=int(
                        st.session_state.get("literature_pool_multiplier", 2)
                    ),
                    shortages=selection.shortages,
                )
            except ValueError as exc:
                st.error(f"无法继续扩大候选池：{exc}")
            else:
                expanded = recovery.confirmed_blueprint
                serialized_blueprint = expanded.blueprint.model_dump_json(indent=2)
                st.session_state["literature_blueprint_json"] = serialized_blueprint
                st.session_state["literature_blueprint_editor"] = serialized_blueprint
                st.session_state["literature_confirmed_blueprint_json"] = (
                    expanded.model_dump_json(indent=2)
                )
                st.session_state["literature_pool_multiplier"] = recovery.pool_multiplier
                _clear_literature_derived_state()
                st.session_state["literature_auto_run_requested"] = True
                st.rerun()
        if actions[1].button(
            "返回修改关键词或限制",
            width="stretch",
        ):
            st.session_state.pop("literature_confirmed_blueprint_json", None)
            _clear_literature_derived_state()
            st.rerun()

    if selection.policy_issues:
        st.error("文献数量可能已达到，但 V0.1 策略仍未满足，因此不能进入 V0.3。")
        st.code("\n".join(selection.policy_issues))

    if selection.admission_exclusions:
        with st.expander("查看文献准入闸门的排除统计"):
            st.dataframe(
                [
                    {"排除原因": reason, "篇数": count}
                    for reason, count in sorted(
                        selection.admission_exclusions.items()
                    )
                ],
                width="stretch",
                hide_index=True,
            )

    cug_unranked = sum(item.cug_tier is None for item in selection.selected)
    norwegian_fallback = sum(
        item.cug_tier is None and item.norwegian_level is not None for item in selection.selected
    )
    dual_unranked = sum(
        item.cug_tier is None and item.norwegian_level is None for item in selection.selected
    )
    if cug_unranked:
        st.info(
            f"有 {cug_unranked} 篇在所选地大学科目录中未取得唯一等级；"
            f"其中 {norwegian_fallback} 篇由挪威国家目录2025提供补充分级，"
            f"仍有 {dual_unranked} 篇在两个目录中均未取得唯一等级。"
        )
    norwegian_level_zero = sum(item.norwegian_level == 0 for item in selection.selected)
    if norwegian_level_zero:
        st.warning(
            f"有 {norwegian_level_zero} 篇对应挪威目录 Level 0（2025年未获认可）。"
            "这不推翻 DOI 真实性，但属于较低的期刊质量偏好。"
        )
    score_counts: dict[float, int] = {}
    for item in selection.selected:
        score_counts[item.relevance_score] = score_counts.get(item.relevance_score, 0) + 1
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
                "中心性": item.centrality,
                "具体支撑论点": item.supported_claim or "待人工复核",
                "适用章节": item.suitable_section_id or item.theme_id,
                "使用边界": item.use_boundary or "仅按准入结论使用",
                "外文": item.is_foreign,
                "期刊": item.journal or "—",
                "出版社": item.publisher or "—",
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
    with st.expander("导出文献与验证记录"):
        st.caption(f"本地运行目录：{st.session_state['literature_run_dir']}")
        downloads = st.columns(3)
        downloads[0].download_button(
            "文献 JSON",
            st.session_state["literature_result_json"],
            file_name="verified_literature_selection.json",
            mime="application/json",
            width="stretch",
        )
        downloads[1].download_button(
            "文献 RIS",
            st.session_state["literature_ris"],
            file_name="verified_literature.ris",
            mime="application/x-research-info-systems",
            width="stretch",
        )
        downloads[2].download_button(
            "真实性证据",
            st.session_state["literature_verification_json"],
            file_name="literature_verification_evidence.json",
            mime="application/json",
            width="stretch",
        )
    if selection.target_reached and not selection.policy_issues and not include_downstream:
        if st.button(
            "继续获取核心论文全文",
            type="primary",
            width="stretch",
        ):
            st.session_state["mvp_navigation_request"] = "evidence"
            st.rerun()
    if include_downstream:
        render_pdf_acquisition_console(selection)


def _render_retrieval_adjustment_controls(
    confirmed: ConfirmedLiteratureSearchBlueprint,
) -> None:
    """Always expose target and retrieval-scale controls before V0.2 execution."""

    service = LiteratureShortageRecoveryService()
    minimum_target = service.minimum_allowed_target(confirmed)
    blueprint = confirmed.blueprint
    current_multiplier = int(st.session_state.get("literature_pool_multiplier", 2))

    st.markdown("#### 调整检索规模")
    st.caption(
        "提高最终目标会按原比例重算各主题配额；候选扫描上限控制 Crossref "
        "最多检查多少条；候选池倍率控制每个主题为每个最终名额准备多少候选。"
        "实际候选量受后两者共同限制。"
    )
    st.info(
        f"V0.1 的硬性下限是 {minimum_target} 篇，不能在此处降低；"
        "保存后会保留主题和检索词，清除旧的 V0.2 下游结果并自动重新检索。"
    )
    form_key = (
        "literature_retrieval_adjustment_"
        f"{confirmed.confirmed_at.isoformat()}_{blueprint.max_candidates}"
    )
    with st.form(form_key):
        fields = st.columns(3)
        target_total = int(
            fields[0].number_input(
                "最终目标文献数",
                min_value=minimum_target,
                max_value=100,
                value=max(minimum_target, blueprint.target_total),
                step=1,
                help="这是最终要入选的论文篇数，不是期刊种类数。",
            )
        )
        max_candidates = int(
            fields[1].number_input(
                "Crossref 候选扫描上限",
                min_value=20,
                max_value=1000,
                value=blueprint.max_candidates,
                step=50,
                help="增大后会扫描更多候选，但网络请求、验证和评分耗时也会增加。",
            )
        )
        pool_multiplier = int(
            fields[2].number_input(
                "每主题候选池倍率",
                min_value=1,
                max_value=10,
                value=current_multiplier,
                step=1,
                help="例如 4 表示每个主题尽量为每个最终名额准备约 4 篇候选。",
            )
        )
        submitted = st.form_submit_button(
            "保存检索规模并重新检索",
            type="primary",
            width="stretch",
        )

    if submitted:
        try:
            adjustment = service.adjust_retrieval(
                confirmed,
                target_total=target_total,
                max_candidates=max_candidates,
                current_pool_multiplier=current_multiplier,
                pool_multiplier=pool_multiplier,
            )
        except ValueError as exc:
            st.error(f"检索参数修改失败：{exc}")
        else:
            adjusted = adjustment.confirmed_blueprint
            serialized = adjusted.blueprint.model_dump_json(indent=2)
            st.session_state["literature_blueprint_json"] = serialized
            st.session_state["literature_blueprint_editor"] = serialized
            st.session_state["literature_confirmed_blueprint_json"] = (
                adjusted.model_dump_json(indent=2)
            )
            st.session_state["literature_pool_multiplier"] = adjustment.pool_multiplier
            _clear_literature_derived_state()
            st.session_state["literature_auto_run_requested"] = True
            st.rerun()
