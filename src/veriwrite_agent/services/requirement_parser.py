"""Deterministic V0.1 parser for common Chinese course requirements."""

from __future__ import annotations

import re
from dataclasses import dataclass

from veriwrite_agent.models.requirements import (
    FormattingRequirement,
    LengthRequirement,
    ReferenceRequirement,
    RequirementSpec,
    SourceEvidence,
    StructureRequirement,
)


@dataclass(frozen=True)
class LengthMention:
    value: int
    qualifier: str
    source_text: str


class RuleBasedRequirementParser:
    """Parse measurable constraints without calling an LLM.

    This is intentionally narrow. Later LLM-based parsers must emit the same
    RequirementSpec contract and are evaluated against the same fixtures.
    """

    _section_keywords = (
        "摘要",
        "关键词",
        "引言",
        "国内外研究现状",
        "方法比较",
        "现状分析",
        "存在问题",
        "技术发展趋势",
        "结语",
        "参考文献",
    )

    def parse(self, text: str) -> RequirementSpec:
        normalized = self._normalize(text)
        evidence: list[SourceEvidence] = []
        ambiguities: list[str] = []

        length_mentions = self._extract_length_mentions(normalized)
        minimum_chars = self._max_by_qualifier(length_mentions, {"以上", "至少", "不少于"})
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

        minimum_total = self._extract_int(
            normalized,
            r"参考文献[^。；\n]{0,45}?(?:不少于|至少|应)\s*(\d+)\s*(?:篇|本)",
            "references.minimum_total",
            evidence,
        )
        if minimum_total is None:
            minimum_total = self._extract_int(
                normalized,
                r"引用[^。；\n]{0,25}?参考文献[^。；\n]{0,25}?(\d+)\s*(?:篇|本)",
                "references.minimum_total",
                evidence,
            )

        foreign_ratio = None
        foreign_match = re.search(r"外文文献[^。；\n]{0,30}?三分之一", normalized)
        if foreign_match:
            foreign_ratio = 1 / 3
            evidence.append(
                SourceEvidence(
                    field="references.minimum_foreign_ratio",
                    source_text=foreign_match.group(0),
                )
            )

        recent_window = self._extract_int(
            normalized,
            r"近\s*(\d+)\s*年",
            "references.recent_year_window",
            evidence,
        )
        max_cluster = self._extract_int(
            normalized,
            r"一次性引用[^。；\n]{0,20}?不能超过\s*(\d+)\s*篇",
            "references.max_references_per_citation_cluster",
            evidence,
        )

        topic_match = re.search(
            r"(?:研究主题|论文题目|综述题目)\s*[：:]\s*([^\n。]{2,100})", normalized
        )
        topic = topic_match.group(1).strip() if topic_match else None
        if topic_match:
            evidence.append(
                SourceEvidence(field="topic", source_text=topic_match.group(0))
            )

        workflow_conditions: list[str] = []
        if "审核未通过" in normalized and "修改说明" in normalized:
            workflow_conditions.append("学院审核未通过后再次提交时必须提供修改说明")

        deliverable_candidates = (
            "研究方向文献综述说明",
            "课程论文封面",
            "考试成绩登记表",
            "文献综述正文",
            "参考文献",
        )
        deliverables = [item for item in deliverable_candidates if item in normalized]

        theme_candidates = ("人工智能", "新一代信息技术", "专业领域", "多学科交叉")
        themes = [item for item in theme_candidates if item in normalized]
        sections = [item for item in self._section_keywords if item in normalized]

        references = ReferenceRequirement(
            minimum_total=minimum_total,
            minimum_foreign_ratio=foreign_ratio,
            recent_year_window=recent_window,
            recent_year_rule_strength=(
                "soft_preference" if recent_window and "尽量" in normalized else "unspecified"
            ),
            preferred_source_types=self._present_items(
                normalized, ("重要学术期刊论文", "专著", "硕博学位论文")
            ),
            discouraged_source_types=self._present_items(
                normalized, ("会议论文", "报告")
            ),
            citation_order="first_appearance" if "按顺序" in normalized else "unspecified",
            in_text_style="numeric_superscript" if "上角标" in normalized else "unspecified",
            max_references_per_citation_cluster=max_cluster,
            all_bibliography_items_must_be_cited_and_discussed=(
                "必须在正文中" in normalized and "引用和评述" in normalized
            ),
        )

        return RequirementSpec(
            document_type="research_direction_literature_review",
            institution="中国地质大学未来技术学院" if "未来技术学院" in normalized else None,
            course_name=(
                "研究方向文献综述与论文写作（硕士）"
                if "研究方向文献综述与论文写作" in normalized
                else None
            ),
            topic=topic,
            topic_source="explicit" if topic else "user_confirmation_required",
            required_theme_elements=themes,
            deliverables=deliverables,
            length=LengthRequirement(
                minimum_chars=minimum_chars,
                target_chars=target_chars,
                figures_excluded="图件不计字数" in normalized,
            ),
            structure=StructureRequirement(
                required_or_recommended_sections=sections,
                must_include_original_analysis=(
                    "自己的观点" in normalized or "独立的见解" in normalized
                ),
                must_not_list_titles_or_abstracts_only=(
                    "不能单纯把文献的题目或摘要罗列" in normalized
                ),
            ),
            references=references,
            formatting=FormattingRequirement(
                paper_size="A4" if "A4" in normalized else None,
                body_font="宋体" if "宋体" in normalized else None,
                body_font_size="小四" if "小四" in normalized else None,
                line_spacing=1.5 if "1.5倍行距" in normalized else None,
            ),
            workflow_conditions=workflow_conditions,
            ambiguities=ambiguities,
            source_evidence=evidence,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("（", "(").replace("）", ")")
        return re.sub(r"[ \t]+", " ", text)

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
        mentions: list[LengthMention], qualifiers: set[str]
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

