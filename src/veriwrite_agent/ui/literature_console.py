"""Streamlit views that connect the confirmed V0.1 hand-off to V0.2."""

from __future__ import annotations

import json
from pathlib import Path

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
from veriwrite_agent.services.requirement_policy import RequirementPolicyCompiler
from veriwrite_agent.ui.literature_workbench import (
    LITERATURE_RESULT_SCHEMA_VERSION,
    LiteratureWorkbench,
)
from veriwrite_agent.ui.evidence_console import (
    PDF_STATE_KEYS,
    render_pdf_acquisition_console,
)
from veriwrite_agent.ui.workbench import project_root

LITERATURE_AUTO_ADVANCE_KEY = "literature_auto_advance_requested"

LITERATURE_STATE_KEYS = (
    "literature_blueprint_json",
    "literature_blueprint_editor",
    "literature_confirmed_blueprint_json",
    "literature_result_json",
    "literature_ris",
    "literature_verification_json",
    "literature_run_dir",
    "literature_pool_multiplier",
    "literature_recovery_seed_run_dir",
    "literature_auto_run_requested",
    LITERATURE_AUTO_ADVANCE_KEY,
    *PDF_STATE_KEYS,
    "mvp_final_matter_json",
    "mvp_final_paper_json",
    "mvp_ai_declaration",
    "mvp_final_repair_checkpoint_json",
    "mvp_final_semantic_review_attestation",
    "mvp_final_repair_auto_suppressed_id",
    "mvp_external_writing_evaluation_json",
    "mvp_external_writing_baseline_json",
    "mvp_external_writing_failure_json",
    "v04_selected_section",
    "v04_selected_section_request",
)

LITERATURE_DERIVED_STATE_KEYS = (
    "literature_result_json",
    "literature_ris",
    "literature_verification_json",
    "literature_run_dir",
    "literature_recovery_seed_run_dir",
    *PDF_STATE_KEYS,
    "mvp_final_matter_json",
    "mvp_final_paper_json",
    "mvp_ai_declaration",
    "mvp_final_semantic_review_attestation",
    "mvp_final_repair_auto_suppressed_id",
    "mvp_external_writing_evaluation_json",
    "mvp_external_writing_baseline_json",
    "mvp_external_writing_failure_json",
)


def clear_literature_state() -> None:
    for key in LITERATURE_STATE_KEYS:
        st.session_state.pop(key, None)


def _clear_literature_derived_state() -> None:
    for key in LITERATURE_DERIVED_STATE_KEYS:
        st.session_state.pop(key, None)


def render_literature_console(
    *,
    include_downstream: bool = True,
    agent_embedded: bool = False,
) -> None:
    """Render V0.2 only after V0.1 produced a confirmed requirement contract."""

    if "confirmed_json" not in st.session_state:
        return
    confirmed_requirement = ConfirmedRequirementSpec.model_validate_json(
        st.session_state["confirmed_json"]
    )

    if not agent_embedded:
        st.divider()
        st.header("V0.2 查找并验证真实文献")
        st.caption(
            "点击一次后，系统自动完成检索规划、Crossref 检索、真实性验证、"
            "相关性筛选和缺口补搜。"
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
            if _route_completed_evidence_recovery():
                st.rerun()
            _render_completed_literature_result(
                completed_selection,
                payload,
                include_downstream=include_downstream,
            )
            return
        result_uses_previous_engine = (
            payload.get("schema_version") != LITERATURE_RESULT_SCHEMA_VERSION
        )
        if result_uses_previous_engine or not payload.get(
            "automatic_search_exhausted", False
        ):
            # Legacy/in-progress results may have stopped at the old fixed 300-candidate
            # ceiling or before a recovery bug was fixed. Resume automatically under
            # the confirmed topic boundary and reuse every valid cached stage.
            st.session_state["literature_auto_run_requested"] = True
            st.session_state[LITERATURE_AUTO_ADVANCE_KEY] = True

    if "literature_blueprint_json" not in st.session_state:
        st.subheader("检索真实文献")
        st.info(
            "系统会根据已确认的课程要求自动生成检索主题和边界；"
            "数量不足时会自行加深检索并改写缺口主题的检索式。"
        )
        if st.button(
            "确认并获取文献",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("正在规划检索并锁定主题边界…"):
                    blueprint = LiteratureWorkbench.live().plan(confirmed_requirement)
                    confirmed_blueprint = LiteratureBlueprintConfirmationService().confirm(
                        blueprint,
                        confirmed_by=confirmed_requirement.confirmed_by,
                        note="系统依据已确认的V0.1要求自动冻结检索计划。",
                        expected_policy=blueprint.requirement_policy,
                    )
            except Exception as exc:
                st.error(f"检索准备失败：{exc}")
            else:
                serialized = blueprint.model_dump_json(indent=2)
                st.session_state["literature_blueprint_json"] = serialized
                st.session_state["literature_blueprint_editor"] = serialized
                st.session_state["literature_confirmed_blueprint_json"] = (
                    confirmed_blueprint.model_dump_json(indent=2)
                )
                _clear_literature_derived_state()
                st.session_state["literature_auto_run_requested"] = True
                st.session_state[LITERATURE_AUTO_ADVANCE_KEY] = True
                st.rerun()
        return

    draft = LiteratureSearchBlueprint.model_validate_json(
        st.session_state["literature_blueprint_json"]
    )
    if not agent_embedded:
        st.subheader("检索方案")
        _render_blueprint_summary(draft)

    if "literature_confirmed_blueprint_json" not in st.session_state:
        st.caption("系统会自动采用该方案；底层 JSON 仅供高级检查，不是必做步骤。")
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
            "确认并获取文献",
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
                st.session_state[LITERATURE_AUTO_ADVANCE_KEY] = True
                st.rerun()
        return

    confirmed_blueprint = ConfirmedLiteratureSearchBlueprint.model_validate_json(
        st.session_state["literature_confirmed_blueprint_json"]
    )
    if not agent_embedded:
        st.success("检索计划已准备完成。")
    pool_multiplier = int(st.session_state.get("literature_pool_multiplier", 2))
    if agent_embedded:
        st.caption("Agent 正在复用已确认边界，只补搜当前缺失论点；成功结果逐批缓存。")
    else:
        st.caption(
            f"首轮按每个主题最终配额的约 {pool_multiplier} 倍检索；"
            "真实性验证和相关性筛选后若仍有缺口，系统只扩展缺口主题，"
            "并从上次位置继续。每轮结果都会缓存。"
        )
        with st.expander("查看或导出检索计划"):
            st.download_button(
                "下载检索方案",
                st.session_state["literature_confirmed_blueprint_json"],
                file_name="confirmed_literature_search_blueprint.json",
                mime="application/json",
                width="stretch",
            )
        st.subheader("检索真实文献")
    auto_run_requested = bool(
        st.session_state.pop("literature_auto_run_requested", False)
        or "literature_result_json" not in st.session_state
    )
    if auto_run_requested:
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
                seed_run_dir=(
                    Path(seed_run_dir)
                    if (
                        seed_run_dir := st.session_state.get(
                            "literature_recovery_seed_run_dir"
                        )
                    )
                    else None
                ),
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
            st.session_state.pop("literature_recovery_seed_run_dir", None)
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
        if st.session_state.pop(LITERATURE_AUTO_ADVANCE_KEY, False):
            st.session_state["mvp_navigation_request"] = "evidence"
            st.session_state["mvp_flash"] = (
                "文献数量、主题配额与真实性门禁均已满足；系统已自动进入 V0.3，"
                "开始检查可获取的核心全文。"
            )
            st.rerun()
        if st.button("进入核心论文全文获取", type="primary", width="stretch"):
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


def _route_completed_evidence_recovery() -> bool:
    """Continue an internal V0.4 recovery without exposing a redundant user gate."""

    raw = st.session_state.get("v04_evidence_recovery_json")
    if not raw:
        return False
    try:
        request = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False
    if request.get("status") != "pending_search":
        return False
    request["status"] = "pending_full_text"
    st.session_state["v04_evidence_recovery_json"] = json.dumps(
        request,
        ensure_ascii=False,
        indent=2,
    )
    st.session_state["mvp_navigation_request"] = (
        "writing"
        if st.session_state.get("v04_autopilot_requested")
        else "evidence"
    )
    st.session_state["mvp_flash"] = (
        "定向补搜已满足文献数量与主题要求；系统正在自动检查本地全文、"
        "更新证据库并恢复未完成章节。"
    )
    return True


def _render_blueprint_summary(
    blueprint: LiteratureSearchBlueprint,
) -> None:
    metrics = st.columns(4)
    metrics[0].metric("主题数", len(blueprint.themes))
    metrics[1].metric("最终目标", blueprint.target_total)
    metrics[2].metric(
        "自动候选上限",
        min(1000, blueprint.target_total * 10),
    )
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
        or not item.use_boundary
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
        stop_labels = {
            "candidate_capacity_exhausted": "已达到本项目的候选文献安全容量",
            "search_stagnated": "连续多轮新检索式未发现新的合格 DOI",
            "automatic_round_limit_reached": "已达到自动检索与查询改写轮次上限",
        }
        stop_reason = payload.get("stop_reason")
        st.warning(
            "系统已自动执行固定深度扩展、缺口主题查询改写、去重、真实性验证和"
            "相关性筛选，但仍无法在已确认的主题边界与年份限制内补足配额。"
            f"停止原因：{stop_labels.get(stop_reason, '自动检索空间已经穷尽')}。"
            "系统不会用不相关论文凑数，也不需要重复点击检索。"
        )
        st.caption(
            f"自动运行 {payload.get('automatic_rounds', '—')} 轮；"
            f"候选安全容量 {payload.get('candidate_capacity', '—')}。"
            "如硬性数量仍无法满足，需要回到 V0.1 修改年份、主题配额或研究边界，"
            "而不是继续重复相同搜索。"
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
