"""Deterministic V0.1 parser for Chinese course requirements."""

from __future__ import annotations

import re
from dataclasses import dataclass

from veriwrite_agent.models.requirements import (
    AIUsagePolicy,
    FormattingRequirement,
    LengthRequirement,
    PolicyRule,
    ReferenceRequirement,
    RequirementProfile,
    RequirementSpec,
    SelectionPolicy,
    SourceEvidence,
    StructureRequirement,
    SubmissionRequirement,
)


@dataclass(frozen=True)
class LengthMention:
    value: int
    qualifier: str
    source_text: str


class RuleBasedRequirementParser:
    """Extract measurable and operational constraints without an LLM."""

    _section_keywords = (
        "标题",
        "作者",
        "单位",
        "摘要",
        "关键词",
        "引言",
        "研究背景与意义",
        "国内外研究现状",
        "方法比较",
        "现状分析",
        "当前研究问题",
        "未来工作与解决思路",
        "存在问题",
        "技术发展趋势",
        "小标题",
        "结语",
        "参考文献",
        "规范格式的引用",
    )
    _profile_header = re.compile(
        r"(?m)^\s*(?P<teacher>[\u4e00-\u9fff]{2,8})老师"
        r"(?:\s*[（(](?P<track>[^）)\n]{1,20})[）)])?\s*[：:]"
    )
    _mobile_noise = {
        "0",
        "目",
        "品",
        "编辑",
        "分页视图",
        "大纲",
        "AI",
        "操作",
        "148",
        "三",
        "C",
        "O",
    }

    def parse(self, text: str) -> RequirementSpec:
        normalized = self._normalize(text)
        compact = self._compact(normalized)
        evidence: list[SourceEvidence] = []
        ambiguities: list[str] = []
        profiles = self._parse_profiles(normalized, evidence)

        shared_text = self._shared_text(normalized) if profiles else normalized
        length = self._parse_length(shared_text, evidence, ambiguities)
        references = self._parse_references(shared_text, evidence)
        structure = self._parse_structure(shared_text)

        topic = None if profiles else self._parse_single_topic(normalized, evidence)
        if len(profiles) == 1:
            topic = profiles[0].topic

        workflow_conditions: list[str] = []
        if "审核未通过" in compact and "修改说明" in compact:
            workflow_conditions.append("学院审核未通过后再次提交时必须提供修改说明")

        deliverables = self._parse_deliverables(compact)
        submission = self._parse_submission(compact)
        if "paper" in submission.required_media:
            self._append_unique(deliverables, "纸质版")
        if "electronic" in submission.required_media:
            self._append_unique(deliverables, "电子版")

        themes = self._present_items(
            compact,
            ("人工智能", "新一代信息技术", "专业领域", "多学科交叉"),
        )
        selection_policy = self._parse_selection_policy(compact)
        ai_policy = self._parse_ai_policy(compact)
        policy_rules = self._parse_global_policy_rules(compact)

        output_language = "pending_confirmation"
        if profiles and all(profile.output_language == "English" for profile in profiles):
            output_language = "English"
        elif "英文综述" in compact:
            output_language = "English"

        return RequirementSpec(
            document_type="research_direction_literature_review",
            institution="中国地质大学" if "中国地质大学" in compact else None,
            school_or_department=("未来技术学院" if "未来技术学院" in compact else None),
            course_name=(
                "研究方向文献综述与论文写作（硕士）"
                if "研究方向文献综述与论文写作" in compact
                else None
            ),
            output_language=output_language,
            topic=topic,
            topic_source="explicit" if topic else "user_confirmation_required",
            required_theme_elements=themes,
            deliverables=deliverables,
            length=length,
            structure=structure,
            references=references,
            formatting=FormattingRequirement(
                paper_size="A4" if "A4" in compact else None,
                body_font="宋体" if "宋体" in compact else None,
                body_font_size="小四" if "小四" in compact else None,
                line_spacing=1.5 if "1.5倍行距" in compact else None,
            ),
            workflow_conditions=workflow_conditions,
            policy_rules=policy_rules,
            profiles=profiles,
            selection_policy=selection_policy,
            submission=submission,
            ai_policy=ai_policy,
            ambiguities=ambiguities,
            source_evidence=evidence,
        )

    def _parse_profiles(
        self,
        text: str,
        evidence: list[SourceEvidence],
    ) -> list[RequirementProfile]:
        matches = list(self._profile_header.finditer(text))
        if not matches:
            return []

        candidates: dict[str, tuple[int, str | None, str]] = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            unified = re.search(r"(?m)^\s*统一要求\s*1\s*[：:]", text[match.end() : end])
            if unified:
                end = match.end() + unified.start()
            teacher = match.group("teacher")
            block = text[match.start() : end].strip()
            previous = candidates.get(teacher)
            if previous is None or len(block) > len(previous[2]):
                candidates[teacher] = (
                    match.start(),
                    (match.group("track") or "").strip() or None,
                    block,
                )

        profiles: list[RequirementProfile] = []
        ordered = sorted(candidates.items(), key=lambda item: item[1][0])
        for option_index, (teacher, (_, track, block)) in enumerate(ordered, 1):
            profile = self._parse_profile(
                profile_id=f"option_{option_index}",
                teacher=teacher,
                track=track,
                block=block,
            )
            profiles.append(profile)
            evidence.append(
                SourceEvidence(
                    field=f"profiles.{option_index - 1}",
                    source_text=self._compact(block)[:500],
                    note=f"识别为可选要求：{teacher}老师",
                )
            )
        return profiles

    def _parse_profile(
        self,
        *,
        profile_id: str,
        teacher: str,
        track: str | None,
        block: str,
    ) -> RequirementProfile:
        compact = self._compact(block)
        topic = self._parse_profile_topic(compact)
        themes = self._profile_theme_elements(compact)
        references = self._parse_references(block, [])
        policy_rules = list(references.restriction_rules)

        if any(marker in compact for marker in ("禁止抄袭", "禁止剽窃", "禁止截图", "不规范引用")):
            policy_rules.append(
                PolicyRule(
                    category="academic_integrity",
                    description="禁止抄袭、剽窃、截图及不规范引用等学术不端行为",
                    consequence="课程总评直接不及格",
                )
            )
        if "直接使用AI工具生成的论文" in compact:
            policy_rules.append(
                PolicyRule(
                    category="ai_usage",
                    description="禁止直接使用 AI 工具生成论文",
                    consequence="判定为0分",
                    score=0,
                )
            )

        return RequirementProfile(
            profile_id=profile_id,
            teacher=teacher,
            track=track,
            output_language="English" if "英文综述" in compact else "pending_confirmation",
            topic=topic,
            required_theme_elements=themes,
            deliverables=["英文文献综述"] if "英文综述" in compact else [],
            length=self._parse_length(block, [], []),
            structure=self._parse_structure(block),
            references=references,
            workflow_conditions=self._profile_workflow_conditions(compact),
            policy_rules=self._dedupe_models(policy_rules),
        )

    @staticmethod
    def _parse_profile_topic(text: str) -> str | None:
        if "AI在大地测量或卫星导航定位" in text:
            return "AI 在大地测量或卫星导航定位中的发展与应用"
        if "卫星遥感温室气体" in text:
            return "卫星遥感温室气体研究"
        if "SAR/InSAR" in text or "SARInSAR" in text:
            return "SAR/InSAR 与遥感相关理论、方法或应用"
        if "地理智能GeoAI在GIS" in text:
            return "地理智能 GeoAI 在 GIS 及相关方向中的发展与应用"
        match = re.search(r"介绍(.{2,180}?)(?:。|并进行总结)", text)
        return match.group(1).strip("，。") if match else None

    @staticmethod
    def _profile_theme_elements(text: str) -> list[str]:
        candidates = (
            ("历史发展", "历史发展"),
            ("最新的研究方向", "最新研究方向"),
            ("最新理论", "最新理论"),
            ("方法", "研究方法"),
            ("应用场景", "应用场景"),
            ("应用进展", "应用进展"),
            ("不足和问题", "当前不足与问题"),
            ("批判性意见", "批判性分析"),
            ("未来可以开展的工作", "未来工作与解决思路"),
        )
        result: list[str] = []
        for marker, value in candidates:
            if marker in text and value not in result:
                result.append(value)
        return result

    def _parse_length(
        self,
        text: str,
        evidence: list[SourceEvidence],
        ambiguities: list[str],
    ) -> LengthRequirement:
        compact = self._compact(text)
        length_mentions = self._extract_length_mentions(compact)
        minimum_chars = self._max_by_qualifier(
            length_mentions,
            {"以上", "至少", "不少于"},
        )
        target_chars = self._max_by_qualifier(length_mentions, {"左右"})
        if minimum_chars is not None and target_chars == minimum_chars:
            ambiguities.append(
                "同一文件同时使用“至少/以上”和“左右”描述字数，默认采用更严格的最低字数。"
            )
        for mention in length_mentions:
            evidence.append(
                SourceEvidence(
                    field="length",
                    source_text=mention.source_text,
                    note=f"解析为 {mention.value} 字，限定词为“{mention.qualifier}”",
                )
            )

        minimum_words = maximum_words = target_words = None
        range_match = re.search(
            r"(\d{3,6})\s*[-—~～至]\s*(\d{3,6})\s*(?:个)?(?:英文)?单词",
            compact,
        )
        if range_match:
            minimum_words = int(range_match.group(1))
            maximum_words = int(range_match.group(2))
            evidence.append(
                SourceEvidence(
                    field="length.minimum_words",
                    source_text=range_match.group(0),
                )
            )
            evidence.append(
                SourceEvidence(
                    field="length.maximum_words",
                    source_text=range_match.group(0),
                )
            )
        else:
            target_match = re.search(r"(\d{3,6})\s*单词\s*左右", compact)
            if target_match:
                target_words = int(target_match.group(1))

        excluded: list[str] = []
        exclusion_markers = (
            ("参考文献不计算", "参考文献"),
            ("参考文献不计入", "参考文献"),
            ("摘要不计入", "摘要"),
            ("封面文档不计入", "封面"),
            ("AI声明不计入", "AI声明"),
        )
        for marker, component in exclusion_markers:
            if marker in compact and component not in excluded:
                excluded.append(component)

        return LengthRequirement(
            minimum_chars=minimum_chars,
            target_chars=target_chars,
            minimum_words=minimum_words,
            maximum_words=maximum_words,
            target_words=target_words,
            figures_excluded="图件不计字数" in compact,
            excluded_components=excluded,
            counting_policy="words" if minimum_words or target_words else "pending_confirmation",
        )

    def _parse_references(
        self,
        text: str,
        evidence: list[SourceEvidence],
    ) -> ReferenceRequirement:
        compact = self._compact(text)
        minimum_total = self._extract_int(
            compact,
            r"参考文献[^。；]{0,45}?(?:不少于|至少|应)\s*(\d+)\s*(?:篇|本)",
            "references.minimum_total",
            evidence,
        )
        if minimum_total is None:
            minimum_total = self._extract_int(
                compact,
                r"引用[^。；]{0,25}?参考文献[^。；]{0,25}?(\d+)\s*(?:篇|本)",
                "references.minimum_total",
                evidence,
            )

        target_total = None
        target_match = re.search(r"(\d+)\s*篇\s*左右(?:期刊)?文献", compact)
        if target_match:
            target_total = int(target_match.group(1))
            evidence.append(
                SourceEvidence(
                    field="references.target_total",
                    source_text=target_match.group(0),
                    note="“左右”表示目标数量，不误写为最低数量。",
                )
            )

        foreign_ratio = None
        foreign_match = re.search(r"外文文献[^。；]{0,30}?三分之一", compact)
        if foreign_match:
            foreign_ratio = 1 / 3
            evidence.append(
                SourceEvidence(
                    field="references.minimum_foreign_ratio",
                    source_text=foreign_match.group(0),
                )
            )

        recent_match = re.search(r"近\s*(\d+)\s*年", compact)
        recent_window = int(recent_match.group(1)) if recent_match else None
        if recent_match:
            evidence.append(
                SourceEvidence(
                    field="references.recent_year_window",
                    source_text=recent_match.group(0),
                )
            )
        recent_rule_strength = self._recency_rule_strength(
            compact,
            recent_match,
        )
        max_cluster = self._extract_int(
            compact,
            r"一次性引用[^。；]{0,20}?不能超过\s*(\d+)\s*篇",
            "references.max_references_per_citation_cluster",
            evidence,
        )

        tools = [tool for tool in ("Mendeley", "Endnote") if tool.casefold() in compact.casefold()]
        styles = [
            style
            for marker, style in (
                ("RemoteSensingofEnvironment", "Remote Sensing of Environment"),
                ("IJGIS", "IJGIS"),
            )
            if marker.casefold() in compact.casefold()
        ]
        bibliography_style = (
            " 或 ".join(f"{style} 期刊格式" for style in styles)
            if styles
            else "pending_confirmation"
        )

        restriction_rules: list[PolicyRule] = []
        if any(
            marker in compact for marker in ("MDPI旗下所有", "Frontiersin系列所有", "IEEEAccess")
        ):
            restriction_rules.append(
                PolicyRule(
                    category="source_restriction",
                    description=(
                        "禁止阅读和引用 MDPI 旗下期刊、Frontiers in 系列、"
                        "IEEE Access 及年发文量超过1000的 OA 期刊"
                    ),
                    consequence="作业判60分",
                    score=60,
                )
            )
        if "禁止引用本硕博毕业论文" in compact or "普通中文期刊论文" in compact:
            restriction_rules.append(
                PolicyRule(
                    category="source_restriction",
                    description="禁止引用本硕博毕业论文及普通中文期刊论文（非一级学报）",
                    consequence="作业判60分",
                    score=60,
                )
            )

        return ReferenceRequirement(
            minimum_total=minimum_total,
            target_total=target_total,
            target_is_approximate=target_total is not None,
            minimum_foreign_ratio=foreign_ratio,
            recent_year_window=recent_window,
            recent_year_rule_strength=recent_rule_strength,
            preferred_source_types=self._present_items(
                compact,
                ("重要学术期刊论文", "专著", "硕博学位论文"),
            ),
            discouraged_source_types=(
                ["会议论文", "报告"] if "少引用会议论文和报告" in compact else []
            ),
            citation_order="first_appearance" if "按顺序" in compact else "unspecified",
            in_text_style="numeric_superscript" if "上角标" in compact else "unspecified",
            max_references_per_citation_cluster=max_cluster,
            bibliography_style=bibliography_style,
            style_examples=styles,
            required_management_tools=tools,
            restriction_rules=restriction_rules,
            all_bibliography_items_must_be_cited_and_discussed=(
                "必须在正文中" in compact and "引用和评述" in compact
            ),
        )

    def _parse_structure(self, text: str) -> StructureRequirement:
        compact = self._compact(text)
        sections = [item for item in self._section_keywords if item in compact]
        if "规范格式的参考文献和引用" in compact:
            self._append_unique(sections, "参考文献")
            self._append_unique(sections, "规范格式的引用")
        return StructureRequirement(
            required_or_recommended_sections=sections,
            must_include_original_analysis=(
                "自己的观点" in compact
                or "独立的见解" in compact
                or "批判性意见" in compact
                or "归纳、总结、评述" in compact
            ),
            must_not_list_titles_or_abstracts_only=("不能单纯把文献的题目或摘要罗列" in compact),
        )

    @staticmethod
    def _profile_workflow_conditions(text: str) -> list[str]:
        conditions: list[str] = []
        if "利用文献检索方法" in text:
            conditions.append("使用文献检索方法从常用数据库查找、下载并阅读相关文献")
        if "Mendeley/Endnote" in text:
            conditions.append("使用 Mendeley 或 Endnote 管理并插入参考文献")
        return conditions

    @staticmethod
    def _parse_selection_policy(text: str) -> SelectionPolicy:
        options_total = required_choices = None
        choice_match = re.search(r"(\d+)\s*选\s*(\d+)", text)
        if choice_match:
            options_total = int(choice_match.group(1))
            required_choices = int(choice_match.group(2))
        rules: list[str] = []
        candidates = (
            (
                "请勿跨组、跨实验室选题",
                "本院或 GIS 中心学生不得跨组、跨实验室选题，应选择组内、"
                "大实验室内或研究方向相近教师的题目",
            ),
            (
                "非本院、非GIS中心同学",
                "非本院、非 GIS 中心学生可任选与指导教师要求对应的方向",
            ),
            (
                "选题和选导有偏差",
                "选题与指导教师有偏差时将转交相应教师评分，且可能要求重做",
            ),
            (
                "其他学院的同学可以按照自己研究方向选题",
                "其他学院学生可按自身研究方向选题",
            ),
        )
        for marker, rule in candidates:
            if marker in text:
                rules.append(rule)
        return SelectionPolicy(
            options_total=options_total,
            required_choices=required_choices,
            rules=rules,
            fallback_teacher="姚尧" if "可以选姚尧老师" in text else None,
        )

    @staticmethod
    def _parse_submission(text: str) -> SubmissionRequirement:
        media: list[str] = []
        if "纸质版" in text:
            media.append("paper")
        if "电子版" in text:
            media.append("electronic")
        deadline = re.search(
            r"(\d{1,2})月(\d{1,2})日(?:晚上|晚)?(\d{1,2})点前",
            text,
        )
        deadline_hour = int(deadline.group(3)) if deadline else None
        if (
            deadline
            and ("晚上" in deadline.group(0) or "晚" in deadline.group(0))
            and deadline_hour is not None
            and deadline_hour < 12
        ):
            deadline_hour += 12
        return SubmissionRequirement(
            required_media=media,
            deadline_text=deadline.group(0) if deadline else None,
            deadline_month=int(deadline.group(1)) if deadline else None,
            deadline_day=int(deadline.group(2)) if deadline else None,
            deadline_hour=deadline_hour,
        )

    @staticmethod
    def _parse_ai_policy(text: str) -> AIUsagePolicy:
        if "统一要求3" not in text and "AI声明" not in text:
            return AIUsagePolicy()
        prohibited: list[str] = []
        for marker, description in (
            ("AI无中生有", "使用 AI 无中生有"),
            ("AI生成任何句子或段落内容", "使用 AI 生成任何句子或段落"),
            ("AI洗稿", "使用 AI 洗稿论文或在线资料"),
        ):
            if marker in text:
                prohibited.append(description)
        permitted = []
        if "允许AI用于" in text:
            permitted = [use for use in ("翻译", "润色") if use in text]
        declaration_fields = [
            label
            for marker, label in (
                ("AI工具", "AI工具名称"),
                ("版本号", "版本号"),
                ("修改位置", "修改位置"),
                ("使用目的", "使用目的"),
            )
            if marker in text
        ]
        no_ai_statement = None
        normalized_statement = text.replace(".", "").replace("Al", "AI").replace("al", "ai")
        if "NoAItoolswereusedinthewritingofthisreport" in normalized_statement:
            no_ai_statement = "No AI tools were used in the writing of this report."
        monitoring = []
        if "编辑时间、次数过短或过长" in text:
            monitoring.append("编辑时间或编辑次数过短或过长将引起重视，甚至调查")
        return AIUsagePolicy(
            prohibited_uses=prohibited,
            permitted_uses=permitted,
            declaration_required="AI声明" in text,
            declaration_location=("参考文献之前" if "参考文献之前添加AI声明" in text else None),
            declaration_fields=declaration_fields,
            no_ai_statement=no_ai_statement,
            declaration_excluded_from_word_count="AI声明不计入总字数" in text,
            monitoring_conditions=monitoring,
            violation_consequence=(
                "违规使用 AI 撰写报告，0分" if "违规使用AI撰写报告，0分" in text else None
            ),
            missing_declaration_consequence=(
                "无 AI 声明作业不及格" if "无AI声明作业不及格" in text else None
            ),
        )

    @staticmethod
    def _parse_global_policy_rules(text: str) -> list[PolicyRule]:
        rules: list[PolicyRule] = []
        if "一次无故未到" in text and "平时分0分" in text:
            rules.append(
                PolicyRule(
                    category="attendance",
                    description="平时课程一次无故未到",
                    consequence="平时分0分",
                    score=0,
                )
            )
        return rules

    @staticmethod
    def _shared_text(text: str) -> str:
        match = re.search(r"(?m)^\s*统一要求\s*1\s*[：:]", text)
        return text[match.start() :] if match else text

    @staticmethod
    def _parse_single_topic(
        text: str,
        evidence: list[SourceEvidence],
    ) -> str | None:
        match = re.search(
            r"(?:研究主题|论文题目|综述题目)\s*[：:]\s*([^\n。]{2,100})",
            text,
        )
        if not match:
            return None
        evidence.append(SourceEvidence(field="topic", source_text=match.group(0)))
        return match.group(1).strip()

    @staticmethod
    def _parse_deliverables(text: str) -> list[str]:
        candidates = (
            "课程论文封面",
            "考试成绩登记表",
            "文献综述正文",
            "参考文献",
        )
        result = [item for item in candidates if item in text]
        if "研究方向文献综述说明" in text or "《研究方向文献综述》说明" in text:
            result.insert(0, "研究方向文献综述说明")
        return result

    @classmethod
    def _normalize(cls, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("（", "(").replace("）", ")")
        text = re.sub(r"\bSAR\s*InSAR\b", "SAR/InSAR", text, flags=re.IGNORECASE)
        text = text.replace("SARInSAR", "SAR/InSAR")
        text = text.replace("Al Tool Usage Declaration", "AI Tool Usage Declaration")
        text = text.replace("Al Tool Name", "AI Tool Name")
        text = text.replace("no Al tools", "no AI tools")
        text = text.replace("No Al tools", "No AI tools")
        cleaned: list[str] = []
        for raw_line in text.splitlines():
            line = " ".join(raw_line.split())
            if not line or line in cls._mobile_noise:
                continue
            if re.fullmatch(r"\d{1,2}:\d{2}[0.…]*", line):
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?\s*(?:KB/s)?", line, re.IGNORECASE):
                continue
            if line.startswith(("[OCR_IMAGE]", "[SOURCE_IMAGE_")):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    @staticmethod
    def _compact(text: str) -> str:
        return re.sub(r"\s+", "", text)

    @staticmethod
    def _present_items(text: str, candidates: tuple[str, ...]) -> list[str]:
        return [item for item in candidates if item in text]

    @staticmethod
    def _extract_length_mentions(text: str) -> list[LengthMention]:
        mentions: list[LengthMention] = []
        patterns = (
            re.compile(r"(\d+(?:\.\d+)?)\s*万\s*字?\s*(以上|左右|至少|不少于)"),
            re.compile(r"(\d{4,})\s*字\s*(以上|左右|至少|不少于)"),
        )
        for pattern in patterns:
            for match in pattern.finditer(text):
                number = float(match.group(1))
                value = round(number * 10000) if "万" in match.group(0) else int(number)
                mention = LengthMention(value, match.group(2), match.group(0))
                if mention not in mentions:
                    mentions.append(mention)
        return mentions

    @staticmethod
    def _max_by_qualifier(
        mentions: list[LengthMention],
        qualifiers: set[str],
    ) -> int | None:
        values = [mention.value for mention in mentions if mention.qualifier in qualifiers]
        return max(values) if values else None

    @staticmethod
    def _extract_int(
        text: str,
        pattern: str,
        field: str,
        evidence: list[SourceEvidence],
    ) -> int | None:
        match = re.search(pattern, text)
        if not match:
            return None
        evidence.append(SourceEvidence(field=field, source_text=match.group(0)))
        return int(match.group(1))

    @staticmethod
    def _recency_rule_strength(
        text: str,
        match: re.Match[str] | None,
    ) -> str:
        if match is None:
            return "unspecified"
        clause_start = max(
            text.rfind("。", 0, match.start()),
            text.rfind("；", 0, match.start()),
            text.rfind(";", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        )
        following_boundaries = [
            position
            for delimiter in ("。", "；", ";", "\n")
            if (position := text.find(delimiter, match.end())) >= 0
        ]
        clause_end = min(following_boundaries, default=len(text))
        clause = text[clause_start + 1 : clause_end]
        if any(marker in clause for marker in ("尽量", "优先", "建议")):
            return "soft_preference"
        if any(marker in clause for marker in ("限定", "必须", "仅限", "只限", "不得早于")):
            return "hard"
        return "unspecified"

    @staticmethod
    def _append_unique(values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)

    @staticmethod
    def _dedupe_models(values: list[PolicyRule]) -> list[PolicyRule]:
        result: list[PolicyRule] = []
        for value in values:
            if value not in result:
                result.append(value)
        return result
