"""Run the V0.2.1 Agent-vs-direct-DeepSeek literature authenticity experiment."""

from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field, ValidationError

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.literature.crossref import CrossrefSearchProvider
from veriwrite_agent.literature.cug_catalog import CugJournalRankingProvider
from veriwrite_agent.literature.doi import DoiOrgResolver, DoiRisMetadataProvider
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.literature_discovery import (
    LiteratureCandidate,
    canonicalize_doi,
)
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import (
    ReferenceRequirement,
    RequirementSpec,
    StrictModel,
)
from veriwrite_agent.services.literature_discovery import LiteratureDiscoveryService
from veriwrite_agent.services.literature_identity_verification import (
    LiteratureIdentityVerificationService,
)
from veriwrite_agent.services.literature_keyword_planner import LiteratureKeywordPlanner

OUTPUT_DIR = Path(__file__).resolve().parent
YEAR_FROM = 2022
YEAR_TO = 2026
TARGET_COUNT = 20
TOPIC = (
    "大气遥感：利用卫星、激光雷达和光谱遥感研究气溶胶、云、"
    "温室气体与空气质量"
)


class DirectPaperClaim(StrictModel):
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    year: int = Field(ge=1900, le=2100)
    journal: str = Field(min_length=1)
    doi: str = Field(min_length=1)


class DirectPaperList(StrictModel):
    papers: list[DirectPaperClaim] = Field(min_length=1, max_length=20)


def main() -> None:
    settings = LLMSettings()
    llm = DeepSeekClient(settings)
    ranking = CugJournalRankingProvider.from_default_catalog()
    resolver = DoiOrgResolver()
    ris_provider = DoiRisMetadataProvider()
    verifier = LiteratureIdentityVerificationService(resolver, ris_provider)

    confirmed = ConfirmedRequirementSpec(
        confirmed_by="controlled-experiment",
        requirement=RequirementSpec(
            document_type="research_direction_literature_review",
            topic=TOPIC,
            topic_source="explicit",
            references=ReferenceRequirement(
                minimum_total=TARGET_COUNT,
                recent_year_window=5,
                recent_year_rule_strength="hard",
            ),
        ),
    )
    plan = LiteratureKeywordPlanner(
        llm,
        ranking.available_disciplines,
        current_year=YEAR_TO,
    ).plan(confirmed)
    _write_json("system_search_plan.json", plan.model_dump(mode="json"))

    discovery = LiteratureDiscoveryService(
        CrossrefSearchProvider(),
        ranking,
    ).discover(plan)
    system_rows: list[dict[str, object]] = []
    system_ris: list[str] = []
    for decision in discovery.eligible_records:
        verification = verifier.verify(decision.candidate)
        if verification.status != "verified" or verification.authority is None:
            continue
        metadata = verification.authority.metadata
        if metadata is None or metadata.year is None:
            continue
        if not YEAR_FROM <= metadata.year <= YEAR_TO:
            continue
        system_rows.append(
            {
                "doi": metadata.doi,
                "title": metadata.title,
                "authors": metadata.authors,
                "year": metadata.year,
                "journal": metadata.journal_title,
                "publisher": metadata.publisher,
                "final_url": (
                    verification.resolution.final_url
                    if verification.resolution is not None
                    else None
                ),
                "landing_warning": verification.warning_codes,
                "cug_discipline": decision.ranking.discipline,
                "cug_tier": decision.ranking.resolved_tier,
                "title_topic_signal": _title_topic_signal(metadata.title or ""),
            }
        )
        system_ris.append(verification.authority.raw_ris or "")
        if len(system_rows) >= TARGET_COUNT:
            break

    _write_json(
        "system_results.json",
        {
            "search_diagnostics": {
                "scanned_count": discovery.scanned_count,
                "duplicate_count": discovery.duplicate_count,
                "prefilter_eligible_count": len(discovery.eligible_records),
                "prefilter_excluded_count": len(discovery.excluded_records),
            },
            "papers": system_rows,
        },
    )
    (OUTPUT_DIR / "system_verified.ris").write_text(
        "\n".join(system_ris),
        encoding="utf-8",
    )

    direct_prompt = _direct_prompt()
    (OUTPUT_DIR / "deepseek_prompt.txt").write_text(
        direct_prompt,
        encoding="utf-8",
    )
    raw = llm.complete(
        [
            {
                "role": "system",
                "content": (
                    "你是普通大模型文献助手。本次不使用搜索、浏览器、数据库或外部工具。"
                    "只根据模型已有知识回答，并严格返回JSON对象。"
                ),
            },
            {"role": "user", "content": direct_prompt},
        ],
        response_format={"type": "json_object"},
    )
    (OUTPUT_DIR / "deepseek_raw_response.txt").write_text(raw, encoding="utf-8")
    claims, repair_used = _parse_or_repair_claims(llm, raw)
    deepseek_rows = [
        _evaluate_direct_claim(claim, verifier) for claim in claims.papers
    ]
    _write_json(
        "deepseek_results.json",
        {
            "repair_used": repair_used,
            "papers": deepseek_rows,
        },
    )

    summary = _build_summary(
        settings,
        plan.discipline,
        discovery,
        system_rows,
        deepseek_rows,
        repair_used,
    )
    _write_json("experiment_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _direct_prompt() -> str:
    schema = json.dumps(
        DirectPaperList.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "请直接列出20篇2022年至2026年发表的、与大气遥感密切相关的真实英文期刊论文。"
        "主题包括卫星大气探测、气溶胶、云、温室气体或空气质量遥感。"
        "每篇必须给出完整题名、全部作者、年份、期刊名和DOI。"
        "不要解释，不要使用Markdown，只返回JSON对象。"
        f"输出必须符合以下JSON Schema：{schema}"
    )


def _parse_or_repair_claims(
    llm: DeepSeekClient,
    raw: str,
) -> tuple[DirectPaperList, bool]:
    try:
        return DirectPaperList.model_validate_json(raw), False
    except ValidationError as error:
        repaired = llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "只修复JSON结构，不搜索、不增加新论文、不改变任何文献事实。"
                    ),
                },
                {"role": "user", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "请按给定Schema返回JSON。验证错误："
                        f"{error.errors(include_url=False)}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        (OUTPUT_DIR / "deepseek_repaired_response.txt").write_text(
            repaired,
            encoding="utf-8",
        )
        return DirectPaperList.model_validate_json(repaired), True


def _evaluate_direct_claim(
    claim: DirectPaperClaim,
    verifier: LiteratureIdentityVerificationService,
) -> dict[str, object]:
    result: dict[str, object] = {
        "claim": claim.model_dump(mode="json"),
        "canonical_doi": None,
        "doi_verified": False,
        "verification_status": "not_run",
        "verification_reasons": [],
        "authority": None,
        "field_matches": None,
        "identity_hallucination": False,
        "metadata_error": False,
        "fully_correct": False,
        "within_year_range": False,
        "title_topic_signal": False,
    }
    try:
        doi = canonicalize_doi(claim.doi)
    except ValueError:
        result["verification_status"] = "invalid_doi_syntax"
        return result

    result["canonical_doi"] = doi
    candidate = LiteratureCandidate(
        doi=doi,
        title=claim.title,
        authors=claim.authors,
        year=claim.year,
        journal_title=claim.journal,
        source_provider="deepseek_direct",
        source_url=f"https://doi.org/{doi}",
    )
    verification = verifier.verify(candidate)
    result["verification_status"] = verification.status
    result["verification_reasons"] = verification.reason_codes
    if verification.status != "verified" or verification.authority is None:
        return result

    metadata = verification.authority.metadata
    if metadata is None:
        return result
    matches = {
        "title": _normalize_text(claim.title) == _normalize_text(metadata.title or ""),
        "authors": {
            _normalize_person(value) for value in claim.authors
        }
        == {_normalize_person(value) for value in metadata.authors},
        "year": claim.year == metadata.year,
        "journal": _normalize_text(claim.journal)
        == _normalize_text(metadata.journal_title or ""),
    }
    identity_hallucination = not matches["title"]
    metadata_error = matches["title"] and not all(matches.values())
    result.update(
        {
            "doi_verified": True,
            "authority": metadata.model_dump(mode="json"),
            "field_matches": matches,
            "identity_hallucination": identity_hallucination,
            "metadata_error": metadata_error,
            "fully_correct": all(matches.values()),
            "within_year_range": (
                metadata.year is not None and YEAR_FROM <= metadata.year <= YEAR_TO
            ),
            "title_topic_signal": _title_topic_signal(metadata.title or ""),
        }
    )
    return result


def _build_summary(
    settings: LLMSettings,
    discipline: str,
    discovery: object,
    system_rows: list[dict[str, object]],
    deepseek_rows: list[dict[str, object]],
    repair_used: bool,
) -> dict[str, object]:
    from veriwrite_agent.models.literature_discovery import LiteratureDiscoveryResult

    if not isinstance(discovery, LiteratureDiscoveryResult):
        raise TypeError("discovery must be a LiteratureDiscoveryResult")
    direct_counts = Counter(
        (
            "invalid_doi_syntax"
            if row["verification_status"] == "invalid_doi_syntax"
            else "doi_not_verified"
            if not row["doi_verified"]
            else "identity_hallucination"
            if row["identity_hallucination"]
            else "metadata_error"
            if row["metadata_error"]
            else "fully_correct"
        )
        for row in deepseek_rows
    )
    unique_claim_dois = {
        row["canonical_doi"]
        for row in deepseek_rows
        if row["canonical_doi"] is not None
    }
    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "topic": TOPIC,
        "year_range": [YEAR_FROM, YEAR_TO],
        "target_count": TARGET_COUNT,
        "agent_system": {
            "discipline": discipline,
            "scanned_count": discovery.scanned_count,
            "verified_output_count": len(system_rows),
            "doi_verified_count": len(system_rows),
            "title_topic_signal_count": sum(
                bool(row["title_topic_signal"]) for row in system_rows
            ),
            "landing_warning_count": sum(
                bool(row["landing_warning"]) for row in system_rows
            ),
        },
        "direct_deepseek": {
            "claimed_count": len(deepseek_rows),
            "unique_syntactic_doi_count": len(unique_claim_dois),
            "format_repair_used": repair_used,
            "outcome_counts": dict(direct_counts),
            "doi_verified_count": sum(
                bool(row["doi_verified"]) for row in deepseek_rows
            ),
            "fully_correct_count": sum(
                bool(row["fully_correct"]) for row in deepseek_rows
            ),
            "within_year_range_count": sum(
                bool(row["within_year_range"]) for row in deepseek_rows
            ),
            "title_topic_signal_count": sum(
                bool(row["title_topic_signal"]) for row in deepseek_rows
            ),
        },
    }


def _normalize_text(value: str) -> str:
    text = html.unescape(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"\s*&\s*", " and ", text.casefold())
    text = re.sub(r"[^0-9a-z\u3400-\u9fff]+", " ", text)
    return " ".join(text.split())


def _normalize_person(value: str) -> str:
    return " ".join(sorted(_normalize_text(value).split()))


def _title_topic_signal(title: str) -> bool:
    normalized = _normalize_text(title)
    atmosphere_terms = (
        "atmospher",
        "aerosol",
        "cloud",
        "ozone",
        "air quality",
        "carbon dioxide",
        "methane",
        "greenhouse gas",
        "trace gas",
        "water vapor",
        "precipitation",
    )
    sensing_terms = (
        "remote sensing",
        "satellite",
        "retriev",
        "observation",
        "lidar",
        "radar",
        "spectrom",
        "sensor",
    )
    return any(term in normalized for term in atmosphere_terms) and any(
        term in normalized for term in sensing_terms
    )


def _write_json(filename: str, value: object) -> None:
    (OUTPUT_DIR / filename).write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
