"""Constrain an LLM to semantic relevance scoring of already verified papers."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.literature_selection import (
    LiteratureRelevanceAssessment,
    LiteratureRelevanceAssessmentBatch,
    LiteratureSearchBlueprint,
)
from veriwrite_agent.models.literature_verification import (
    LiteratureVerificationResult,
)


class RelevanceScoringError(ValueError):
    """Raised when the LLM cannot score exactly the supplied papers and themes."""


class LLMLiteratureRelevanceScorer:
    """Let the LLM judge semantic fit, never DOI truth or journal level."""

    def __init__(self, client: LLMClient, *, batch_size: int = 20) -> None:
        if not 1 <= batch_size <= 50:
            raise ValueError("batch_size must be between 1 and 50")
        self._client = client
        self._batch_size = batch_size

    def score(
        self,
        blueprint: LiteratureSearchBlueprint,
        verifications: list[LiteratureVerificationResult],
    ) -> list[LiteratureRelevanceAssessment]:
        if any(result.status != "verified" for result in verifications):
            raise ValueError("relevance scoring accepts verified papers only")
        assessments: list[LiteratureRelevanceAssessment] = []
        for start in range(0, len(verifications), self._batch_size):
            batch = verifications[start : start + self._batch_size]
            assessments.extend(self._score_batch(blueprint, batch))
        return assessments

    def _score_batch(
        self,
        blueprint: LiteratureSearchBlueprint,
        batch: list[LiteratureVerificationResult],
    ) -> list[LiteratureRelevanceAssessment]:
        expected_dois = [result.candidate.doi for result in batch]
        expected_themes = [theme.theme_id for theme in blueprint.themes]
        source = json.dumps(
            {
                "overall_topic": blueprint.topic,
                "topic_boundary": blueprint.topic_boundary.model_dump(mode="json"),
                "themes": [
                    {
                        "theme_id": theme.theme_id,
                        "section_title": theme.section_title,
                        "section_purpose": theme.section_purpose,
                        "research_questions": theme.research_questions,
                    }
                    for theme in blueprint.themes
                ],
                "papers": [
                    {
                        "doi": result.candidate.doi,
                        "title": (
                            result.authority.metadata.title
                            if result.authority is not None
                            and result.authority.metadata is not None
                            else result.candidate.title
                        ),
                        "abstract": result.candidate.abstract,
                    }
                    for result in batch
                ],
            },
            ensure_ascii=False,
        )
        schema = json.dumps(
            LiteratureRelevanceAssessmentBatch.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是已验证文献的章节相关性评估器，只返回JSON。"
                    "你不能判断或修改DOI真实性、题名、作者、年份、期刊等级。"
                    "只能根据给定题名和摘要，给每篇论文对每个theme_id评0至1分。"
                    "0表示无关，0.5表示部分相关，1表示直接支撑该章节研究问题。"
                    "A real DOI is not enough for admission. For every paper, decide "
                    "admission_status, centrality, the exact supported_claim, the suitable "
                    "section, and its use boundary. Admit only when you can answer: if this "
                    "paper were removed, which concrete planned claim would lose support? "
                    "Reject papers matching excluded_objects. A contextual_only topic may "
                    "be admitted only as supporting context and must not define the section. "
                    "Use manual_review when the title and abstract are insufficient. "
                    "每篇必须覆盖全部且仅覆盖给定theme_id，best_theme_id必须是最高分主题。"
                    "不要添加、删除或修改任何DOI。"
                    f"输出必须符合以下JSON Schema：{schema}"
                ),
            },
            {"role": "user", "content": source},
        ]
        parsed = self._complete(messages)
        try:
            self._validate_scope(parsed, expected_dois, expected_themes)
            return parsed.assessments
        except RelevanceScoringError as first_error:
            repaired = self._complete(
                [
                    *messages,
                    {
                        "role": "assistant",
                        "content": parsed.model_dump_json(),
                    },
                    {
                        "role": "user",
                        "content": (
                            "上一个结果超出给定DOI或主题范围。只修复范围和JSON，"
                            f"不要改变论文事实。错误：{first_error}"
                        ),
                    },
                ]
            )
            self._validate_scope(repaired, expected_dois, expected_themes)
            return repaired.assessments

    def _complete(
        self,
        messages: list[dict[str, str]],
    ) -> LiteratureRelevanceAssessmentBatch:
        raw = self._client.complete(
            messages,
            response_format={"type": "json_object"},
        )
        try:
            payload = json.loads(raw)
            _repair_redundant_best_theme_ids(payload)
            return LiteratureRelevanceAssessmentBatch.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RelevanceScoringError(
                "LLM relevance output violates the data contract: "
                f"{_validation_details(exc)}"
            ) from exc

    @staticmethod
    def _validate_scope(
        batch: LiteratureRelevanceAssessmentBatch,
        expected_dois: list[str],
        expected_themes: list[str],
    ) -> None:
        actual_dois = [item.doi for item in batch.assessments]
        if sorted(actual_dois) != sorted(expected_dois):
            raise RelevanceScoringError(
                "relevance output must contain every supplied DOI exactly once"
            )
        expected_theme_set = set(expected_themes)
        for assessment in batch.assessments:
            if assessment.admission_status == "legacy_unreviewed":
                raise RelevanceScoringError(
                    f"admission decision is missing for DOI {assessment.doi}"
                )
            actual_themes = {
                score.theme_id for score in assessment.theme_scores
            }
            if actual_themes != expected_theme_set:
                raise RelevanceScoringError(
                    f"relevance output has wrong themes for DOI {assessment.doi}"
                )
            if (
                assessment.admission_status == "admit"
                and assessment.suitable_section_id not in expected_theme_set
            ):
                raise RelevanceScoringError(
                    "admission output has an unknown suitable section for DOI "
                    f"{assessment.doi}"
                )


def _repair_redundant_best_theme_ids(payload: Any) -> None:
    """Derive best_theme_id from scores instead of failing on an LLM arithmetic slip."""

    if not isinstance(payload, dict):
        return
    assessments = payload.get("assessments")
    if not isinstance(assessments, list):
        return
    for assessment in assessments:
        if not isinstance(assessment, dict):
            continue
        theme_scores = assessment.get("theme_scores")
        if not isinstance(theme_scores, list) or not theme_scores:
            continue
        scored_themes: list[tuple[str, float]] = []
        for item in theme_scores:
            if not isinstance(item, dict):
                continue
            theme_id = item.get("theme_id")
            score = item.get("score")
            if (
                isinstance(theme_id, str)
                and theme_id
                and isinstance(score, (int, float))
                and not isinstance(score, bool)
            ):
                scored_themes.append((theme_id, float(score)))
        if not scored_themes:
            continue
        maximum = max(score for _, score in scored_themes)
        highest = [theme_id for theme_id, score in scored_themes if score == maximum]
        if assessment.get("best_theme_id") not in highest:
            assessment["best_theme_id"] = highest[0]


def _validation_details(exc: json.JSONDecodeError | ValidationError) -> object:
    if isinstance(exc, ValidationError):
        return exc.errors(include_url=False)[:10]
    return str(exc)
