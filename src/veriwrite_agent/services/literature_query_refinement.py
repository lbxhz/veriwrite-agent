"""LLM-assisted query rewriting for unresolved V0.2 theme shortages."""

from __future__ import annotations

import json
import re

from pydantic import Field, ValidationError, field_validator

from veriwrite_agent.llm.base import LLMClient
from veriwrite_agent.models.literature_selection import LiteratureSearchBlueprint
from veriwrite_agent.models.requirements import StrictModel


class LiteratureQueryRefinementError(ValueError):
    """Raised when shortage query rewriting cannot produce a safe contract."""


class ThemeQueryRefinement(StrictModel):
    """New Crossref phrases for one shortage theme."""

    theme_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,39}$")
    search_queries: list[str] = Field(min_length=2, max_length=4)

    @field_validator("search_queries", mode="after")
    @classmethod
    def normalize_queries(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = " ".join(value.split())
            fingerprint = clean.casefold()
            if not clean or fingerprint in seen:
                continue
            if re.search(r"\b(?:AND|OR|NOT)\b", clean):
                raise ValueError(
                    "Crossref search phrases cannot use uppercase Boolean operators"
                )
            normalized.append(clean)
            seen.add(fingerprint)
        if len(normalized) < 2:
            raise ValueError("each shortage theme requires at least two distinct queries")
        return normalized


class LiteratureQueryRefinementBatch(StrictModel):
    """Auditable semantic recovery plan for all current shortage themes."""

    schema_version: str = Field(default="0.2-query-refinement.1")
    themes: list[ThemeQueryRefinement] = Field(min_length=1)


class LiteratureShortageQueryRefiner:
    """Rewrite only shortage-theme queries without weakening the topic boundary."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def refine(
        self,
        blueprint: LiteratureSearchBlueprint,
        shortages: dict[str, int],
        *,
        previous_recovery_queries: dict[str, list[str]] | None = None,
    ) -> LiteratureQueryRefinementBatch:
        shortage_ids = {
            theme_id for theme_id, count in shortages.items() if count > 0
        }
        themes_by_id = {theme.theme_id: theme for theme in blueprint.themes}
        if not shortage_ids or not shortage_ids <= set(themes_by_id):
            raise ValueError("shortages must contain known themes with positive counts")

        previous_recovery_queries = previous_recovery_queries or {}
        source = {
            "topic": blueprint.topic,
            "topic_boundary": blueprint.topic_boundary.model_dump(mode="json"),
            "shortage_themes": [
                {
                    "theme_id": theme_id,
                    "shortage": shortages[theme_id],
                    "section_title": themes_by_id[theme_id].section_title,
                    "section_purpose": themes_by_id[theme_id].section_purpose,
                    "research_questions": themes_by_id[theme_id].research_questions,
                    "primary_keywords": themes_by_id[theme_id].primary_keywords,
                    "related_keywords": themes_by_id[theme_id].related_keywords,
                    "already_used_queries": [
                        *themes_by_id[theme_id].search_queries,
                        *previous_recovery_queries.get(theme_id, []),
                    ],
                }
                for theme_id in sorted(shortage_ids)
            ],
        }
        schema = json.dumps(
            LiteratureQueryRefinementBatch.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是学术文献检索式重写器，只返回JSON。"
                    "当前检索深度已增加但仍有主题缺口。为每个缺口主题生成2至4条新的"
                    "英文Crossref自由文本检索短语。短语应优先寻找直接研究、综述和"
                    "方法比较，而不是泛化背景材料。不得改变topic_boundary，不得纳入"
                    "明确排除对象，不得重复already_used_queries，不得使用大写布尔运算符"
                    "或字段语法。必须且只能返回输入中的每个theme_id一次。"
                    f"输出必须符合以下JSON Schema：{schema}"
                ),
            },
            {"role": "user", "content": json.dumps(source, ensure_ascii=False)},
        ]
        raw = self._client.complete(
            messages,
            response_format={"type": "json_object"},
        )
        try:
            return self._validate_scope(
                LiteratureQueryRefinementBatch.model_validate_json(raw),
                shortage_ids=shortage_ids,
                source=source,
            )
        except (ValidationError, LiteratureQueryRefinementError) as first_error:
            repaired = self._client.complete(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "上一个输出未通过检索式合同。只返回修复后的完整JSON。"
                            f"错误：{first_error}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            try:
                return self._validate_scope(
                    LiteratureQueryRefinementBatch.model_validate_json(repaired),
                    shortage_ids=shortage_ids,
                    source=source,
                )
            except (ValidationError, LiteratureQueryRefinementError):
                # A malformed refinement is not a reason to abort the whole V0.2
                # recovery loop.  The confirmed blueprint already contains enough
                # bounded semantic material to construct conservative Crossref
                # phrases without asking the user to repair model JSON.
                return self._fallback_batch(
                    blueprint,
                    shortage_ids=shortage_ids,
                    previous_recovery_queries=previous_recovery_queries,
                )

    @staticmethod
    def _fallback_batch(
        blueprint: LiteratureSearchBlueprint,
        *,
        shortage_ids: set[str],
        previous_recovery_queries: dict[str, list[str]],
    ) -> LiteratureQueryRefinementBatch:
        """Build boundary-preserving queries when both LLM attempts are unusable."""

        themes_by_id = {theme.theme_id: theme for theme in blueprint.themes}
        refinements: list[ThemeQueryRefinement] = []
        for theme_id in sorted(shortage_ids):
            theme = themes_by_id[theme_id]
            used = {
                query.casefold()
                for query in [
                    *theme.search_queries,
                    *previous_recovery_queries.get(theme_id, []),
                ]
            }
            candidates = LiteratureShortageQueryRefiner._fallback_candidates(
                blueprint,
                theme_id=theme_id,
            )
            queries: list[str] = []
            seen: set[str] = set()
            for candidate in candidates:
                clean = " ".join(candidate.split())
                fingerprint = clean.casefold()
                if (
                    not clean
                    or fingerprint in used
                    or fingerprint in seen
                    or re.search(r"\b(?:AND|OR|NOT)\b", clean)
                ):
                    continue
                queries.append(clean)
                seen.add(fingerprint)
                if len(queries) == 4:
                    break
            if len(queries) < 2:
                raise LiteratureQueryRefinementError(
                    f"deterministic fallback cannot create two queries for {theme_id}"
                )
            refinements.append(
                ThemeQueryRefinement(theme_id=theme_id, search_queries=queries)
            )
        return LiteratureQueryRefinementBatch(
            schema_version="0.2-query-refinement.fallback.1",
            themes=refinements,
        )

    @staticmethod
    def _fallback_candidates(
        blueprint: LiteratureSearchBlueprint,
        *,
        theme_id: str,
    ) -> list[str]:
        """Create a deep, finite search space from confirmed boundary-safe phrases.

        Recovery can run for twelve rounds.  A six-query fallback merely moves the
        failure to a later round, so this generator deliberately provides at least
        48 meaningful variants even when a theme has only one original query.
        """

        theme = {item.theme_id: item for item in blueprint.themes}[theme_id]
        original_bases = list(theme.search_queries)
        mostly_ascii = [
            query for query in original_bases if _ascii_ratio(query) >= 0.75
        ]
        bases = mostly_ascii or original_bases
        if not bases:
            bases = [f"{blueprint.topic} {theme.section_title}"]
        evidence_foci = (
            "data quality and uncertainty",
            "interpretability and explainability",
            "physics informed constraints",
            "multi sensor data fusion",
            "operational validation",
            "computational efficiency",
            "benchmark datasets",
            "generalization and transferability",
        )
        study_intents = (
            "systematic review",
            "comparative analysis",
            "recent advances",
            "research challenges",
            "future directions",
            "methods evaluation",
        )
        return [
            f"{base} {focus} {intent}"
            for focus in evidence_foci
            for intent in study_intents
            for base in bases
        ]

    @staticmethod
    def _validate_scope(
        batch: LiteratureQueryRefinementBatch,
        *,
        shortage_ids: set[str],
        source: dict[str, object],
    ) -> LiteratureQueryRefinementBatch:
        actual_ids = [theme.theme_id for theme in batch.themes]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != shortage_ids:
            raise LiteratureQueryRefinementError(
                "query refinement must contain every shortage theme exactly once"
            )
        source_themes = source["shortage_themes"]
        if not isinstance(source_themes, list):
            raise TypeError("shortage theme source must be a list")
        used_by_theme = {
            str(item["theme_id"]): {
                str(query).casefold() for query in item["already_used_queries"]
            }
            for item in source_themes
            if isinstance(item, dict)
            and isinstance(item.get("already_used_queries"), list)
        }
        validated_themes: list[ThemeQueryRefinement] = []
        for theme in batch.themes:
            new_queries = [
                query
                for query in theme.search_queries
                if query.casefold() not in used_by_theme.get(theme.theme_id, set())
            ]
            if len(new_queries) < 2:
                raise LiteratureQueryRefinementError(
                    "query refinement must retain at least two new queries for "
                    f"{theme.theme_id}"
                )
            validated_themes.append(
                ThemeQueryRefinement(
                    theme_id=theme.theme_id,
                    search_queries=new_queries,
                )
            )
        return LiteratureQueryRefinementBatch(
            schema_version=batch.schema_version,
            themes=validated_themes,
        )


def _ascii_ratio(value: str) -> float:
    visible = [character for character in value if not character.isspace()]
    if not visible:
        return 0.0
    return sum(ord(character) < 128 for character in visible) / len(visible)
