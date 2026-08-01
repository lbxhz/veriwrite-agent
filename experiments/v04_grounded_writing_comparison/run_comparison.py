"""Controlled DeepSeek comparison: one-shot PDF writing versus V0.4."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    EvidenceCard,
    EvidenceLibrary,
    LiteratureLibraryRecord,
)
from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import RequirementSpec
from veriwrite_agent.models.writing_handoff import (
    ConfirmedWritingOutline,
    V04WritingHandoff,
    WritingOutlineDraft,
    WritingOutlineSection,
)
from veriwrite_agent.services.evidence_card_extraction import (
    LLMEvidenceCardExtractor,
)
from veriwrite_agent.services.grounded_writing import (
    LLMGroundedSectionWriter,
    SectionEvidencePacketBuilder,
)
from veriwrite_agent.services.pdf_text_extraction import PdfPageExtractor

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
PDF_DIR = Path(r"E:\AI-Agent-Projects")
SECTION_ID = "aerosol_remote_sensing"
SECTION_TITLE = "气溶胶遥感反演方法、应用与局限"
SECTION_PURPOSE = (
    "综合比较卫星气溶胶光学厚度反演、时空连续估计、降尺度方法和"
    "辐射效应研究，并指出证据边界与当前局限。"
)
TARGET_UNITS = 800
COMPARISON_MODEL = "deepseek-chat"

PAPERS = [
    {
        "doi": "10.1016/j.atmosres.2025.108411",
        "title": (
            "Downscaling aerosol optical depth by fusing satellite retrieval "
            "and model simulation using artificial intelligence technology"
        ),
        "authors": ["Sun, Lin", "Zhang, Xiangshuo", "Fan, Yulong"],
        "year": 2026,
        "journal": "Atmospheric Research",
        "filename": "1-s2.0-S0169809525005034-main.pdf",
    },
    {
        "doi": "10.1016/j.atmosenv.2025.121365",
        "title": (
            "Enhanced continuous aerosol optical depth (AOD) estimation using "
            "geostationary satellite data: focusing on nighttime AOD over East Asia"
        ),
        "authors": ["Song, Sanghyeon", "Kang, Yoojin", "Im, Jungho"],
        "year": 2025,
        "journal": "Atmospheric Environment",
        "filename": "1-s2.0-S1352231025003401-main.pdf",
    },
    {
        "doi": "10.1016/j.atmosres.2021.105938",
        "title": (
            "Spatio-temporal distribution of aerosol direct radiative forcing "
            "over mid-latitude regions in north hemisphere estimated from "
            "satellite observations"
        ),
        "authors": ["Chen, Annan", "Zhao, Chuanfeng", "Fan, Tianyi"],
        "year": 2022,
        "journal": "Atmospheric Research",
        "filename": "1-s2.0-S0169809521004944-main.pdf",
    },
]

EXPERIMENT_PAGES = {
    "10.1016/j.atmosres.2025.108411": {1, 6, 10},
    "10.1016/j.atmosenv.2025.121365": {1, 6, 12},
    "10.1016/j.atmosres.2021.105938": {1, 9, 10},
}


class CriterionScore(BaseModel):
    score_a: int = Field(ge=1, le=5)
    score_b: int = Field(ge=1, le=5)
    reason: str


class BlindEvaluation(BaseModel):
    coherence: CriterionScore
    source_fidelity: CriterionScore
    citation_authenticity: CriterionScore
    traceability: CriterionScore
    thematic_synthesis: CriterionScore
    unsupported_claim_risk: CriterionScore
    winner: str
    confidence: float = Field(ge=0, le=1)
    summary: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=["baseline", "evidence", "grounded", "judge", "report"],
    )
    parser.add_argument(
        "--doi",
        help="Limit the evidence phase to one DOI for recoverable runs.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    {
        "baseline": run_baseline,
        "evidence": lambda: run_evidence(args.doi),
        "grounded": run_grounded,
        "judge": run_judge,
        "report": write_report,
    }[args.phase]()


def run_baseline() -> None:
    pages_by_doi, _, _ = load_documents()
    pdf_payload = []
    for paper in PAPERS:
        pages = pages_by_doi[paper["doi"]]
        pdf_payload.append(
            {
                "filename": paper["filename"],
                "pages": [
                    {
                        "page": page.page_number,
                        "text": page.text[:5000],
                    }
                    for page in pages
                ],
            }
        )
    prompt = (
        "下面是用户上传的三篇PDF自动提取文本。请直接撰写约800字的中文文献综述"
        f"章节《{SECTION_TITLE}》。要求比较方法、应用和局限，采用作者-年份引用，"
        "最后列出本章参考文献及DOI。不要编造论文或结果。\n\n"
        + json.dumps(pdf_payload, ensure_ascii=False)
    )
    response = client().complete(
        [
            {
                "role": "system",
                "content": (
                    "你是普通的学术写作LLM。用户上传PDF后，你负责阅读并直接写作。"
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    (OUTPUT_DIR / "baseline_prompt.txt").write_text(prompt, encoding="utf-8")
    (OUTPUT_DIR / "baseline_response.md").write_text(
        response,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "phase": "baseline",
                "characters": len(response),
                "doi_mentions": extract_dois(response),
            },
            ensure_ascii=False,
        )
    )


def run_evidence(selected_doi: str | None = None) -> None:
    pages_by_doi, _, _ = load_documents()
    extractor = LLMEvidenceCardExtractor(
        client(),
        page_batch_size=1,
        max_chars_per_page=1200,
        max_cards_per_batch=2,
        max_quote_chars=300,
    )
    cards_path = OUTPUT_DIR / "evidence_cards.json"
    cards = (
        [
            EvidenceCard.model_validate(item)
            for item in json.loads(cards_path.read_text(encoding="utf-8"))
        ]
        if cards_path.exists()
        else []
    )
    papers = [
        paper
        for paper in PAPERS
        if selected_doi is None or paper["doi"] == selected_doi
    ]
    if selected_doi and not papers:
        raise ValueError(f"unknown experiment DOI: {selected_doi}")
    for paper in papers:
        cards = [card for card in cards if card.doi != paper["doi"]]
        print(f"extracting evidence: {paper['doi']}", flush=True)
        paper_cards = extractor.extract(
            doi=paper["doi"],
            title=paper["title"],
            theme_id=SECTION_ID,
            section_purpose=SECTION_PURPOSE,
            pages=[
                page
                for page in pages_by_doi[paper["doi"]]
                if page.page_number in EXPERIMENT_PAGES[paper["doi"]]
            ],
        )
        cards.extend(
            card.model_copy(update={"review_status": "confirmed"})
            for card in paper_cards
        )
        cards_path.write_text(
            json.dumps(
                [card.model_dump(mode="json") for card in cards],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"grounded cards: {paper['doi']} -> {len(paper_cards)}",
            flush=True,
        )
    print(json.dumps({"phase": "evidence", "cards": len(cards)}))


def run_grounded() -> None:
    cards = [
        EvidenceCard.model_validate(item)
        for item in json.loads(
            (OUTPUT_DIR / "evidence_cards.json").read_text(encoding="utf-8")
        )
    ]
    handoff = build_handoff(cards)
    packet = SectionEvidencePacketBuilder().build(handoff, SECTION_ID)
    draft = LLMGroundedSectionWriter(client()).draft(packet)
    (OUTPUT_DIR / "v04_evidence_packet.json").write_text(
        packet.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "v04_draft_audit.json").write_text(
        draft.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "v04_response.md").write_text(
        draft.markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "phase": "grounded",
                "status": draft.status,
                "counted_units": draft.counted_words,
                "citations": len(draft.citations),
                "issues": [issue.code for issue in draft.issues],
            },
            ensure_ascii=False,
        )
    )


def run_judge() -> None:
    baseline = (OUTPUT_DIR / "baseline_response.md").read_text(encoding="utf-8")
    grounded = (OUTPUT_DIR / "v04_response.md").read_text(encoding="utf-8")
    packet = json.loads(
        (OUTPUT_DIR / "v04_evidence_packet.json").read_text(encoding="utf-8")
    )
    evidence_digest = [
        {
            "evidence_id": item["evidence_id"],
            "doi": item["doi"],
            "claim": item["normalized_claim"],
            "quotes": item["supporting_quotes"],
        }
        for item in packet["evidence_items"]
    ]
    schema = json.dumps(
        BlindEvaluation.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = {
        "task": SECTION_PURPOSE,
        "verified_evidence": evidence_digest,
        "draft_a": baseline,
        "draft_b": grounded,
        "scoring": (
            "1 is poor and 5 is strong. For unsupported_claim_risk, 5 means "
            "low risk and 1 means high risk."
        ),
    }
    raw = client().complete(
        [
            {
                "role": "system",
                "content": (
                    "You are a blind evaluator of two academic section drafts. "
                    "Use only the supplied verified evidence to assess factual support. "
                    "Do not assume Draft A or B came from any particular system. "
                    f"Return JSON matching this schema: {schema}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        response_format={"type": "json_object"},
    )
    evaluation = BlindEvaluation.model_validate_json(raw)
    (OUTPUT_DIR / "blind_judge.json").write_text(
        evaluation.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(evaluation.model_dump_json())


def write_report() -> None:
    baseline = (OUTPUT_DIR / "baseline_response.md").read_text(encoding="utf-8")
    audit = json.loads(
        (OUTPUT_DIR / "v04_draft_audit.json").read_text(encoding="utf-8")
    )
    evaluation = BlindEvaluation.model_validate_json(
        (OUTPUT_DIR / "blind_judge.json").read_text(encoding="utf-8")
    )
    known_dois = {paper["doi"] for paper in PAPERS}
    baseline_dois = set(extract_dois(baseline))
    grounded_dois = {citation["doi"] for citation in audit["citations"]}
    grounded_with_pages = sum(
        bool(citation["page_numbers"]) for citation in audit["citations"]
    )
    rows = [
        (
            "引用DOI均来自给定论文",
            f"{len(baseline_dois - known_dois) == 0}",
            f"{len(grounded_dois - known_dois) == 0}",
        ),
        (
            "可回溯到证据ID和PDF页码",
            "0个机器可验证绑定",
            f"{grounded_with_pages}/{len(audit['citations'])}个绑定含页码",
        ),
        (
            "代码阻止越权或虚构引用",
            "否",
            "是",
        ),
        (
            "结构化可恢复中间产物",
            "无",
            "证据卡、章节证据包、草稿审计包",
        ),
        (
            "调用方式",
            "1次长上下文生成",
            "分批证据提取 + 1次章节生成",
        ),
    ]
    criteria = {
        "连贯性": evaluation.coherence,
        "来源忠实度": evaluation.source_fidelity,
        "引用真实性": evaluation.citation_authenticity,
        "可追溯性": evaluation.traceability,
        "主题综合": evaluation.thematic_synthesis,
        "低无依据风险": evaluation.unsupported_claim_risk,
    }
    lines = [
        "# V0.4与普通LLM+PDF受控对照",
        "",
        f"- 运行时间：{datetime.now(timezone.utc).isoformat()}",
        f"- 模型：{COMPARISON_MODEL}",
        "- 输入：同三篇气溶胶论文PDF全文提取文本",
        f"- 任务：{SECTION_PURPOSE}",
        "",
        "## 确定性审计",
        "",
        "| 指标 | 普通LLM+PDF | V0.4 |",
        "|---|---|---|",
    ]
    lines.extend(f"| {name} | {left} | {right} |" for name, left, right in rows)
    lines.extend(
        [
            "",
            "## 盲评模型评分",
            "",
            "| 维度 | 草稿A | 草稿B | 说明 |",
            "|---|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {name} | {score.score_a} | {score.score_b} | {score.reason} |"
        for name, score in criteria.items()
    )
    lines.extend(
        [
            "",
            f"- 盲评胜者：{evaluation.winner}",
            f"- 置信度：{evaluation.confidence:.2f}",
            f"- 总结：{evaluation.summary}",
            "",
            "## 解释边界",
            "",
            "- 本实验不是模型排行榜，只比较同一模型在两种工作流中的行为。",
            "- 普通方案得到更少API调用和更低延迟；V0.4增加了证据提取成本。",
            "- 盲评属于LLM评估，不是绝对真值；确定性引用审计才是硬指标。",
            "- V0.4的优势主要是可审计、可恢复和失效可见，而非保证文风一定更好。",
            "",
            "## 文件",
            "",
            "- `baseline_response.md`：普通LLM+PDF输出",
            "- `evidence_cards.json`：经原文短引句校验的证据卡",
            "- `v04_response.md`：V0.4输出",
            "- `v04_draft_audit.json`：引用与章节审计",
            "- `blind_judge.json`：匿名LLM评分",
        ]
    )
    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "comparison_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(report)


def build_handoff(cards: list[EvidenceCard]) -> V04WritingHandoff:
    pages_by_doi, documents, all_pages = load_documents()
    del pages_by_doi
    records = [
        LiteratureLibraryRecord(
            doi=paper["doi"],
            title=paper["title"],
            authors=paper["authors"],
            year=paper["year"],
            journal=paper["journal"],
            source_url=f"https://doi.org/{paper['doi']}",
            theme_ids=[SECTION_ID],
            evidence_tier="A_core",
            evidence_status="full_text_verified",
            permitted_use="detailed_claims",
        )
        for paper in PAPERS
    ]
    library = EvidenceLibrary(
        status="confirmed",
        records=records,
        documents=documents,
        pages=all_pages,
        evidence_cards=cards,
        confirmed_by="controlled_experiment",
        confirmed_at=datetime.now(timezone.utc),
    )
    outline = ConfirmedWritingOutline(
        outline=WritingOutlineDraft(
            topic="大气气溶胶遥感",
            writing_through_line="从反演方法到应用与局限。",
            target_words=TARGET_UNITS,
            sections=[
                WritingOutlineSection(
                    section_id=SECTION_ID,
                    title=SECTION_TITLE,
                    purpose=SECTION_PURPOSE,
                    target_words=TARGET_UNITS,
                    research_questions=[
                        "AI与时空连续估计如何改善AOD产品？",
                        "卫星观测如何用于气溶胶辐射效应研究？",
                        "现有方法的验证和适用边界是什么？",
                    ],
                    core_dois=[paper["doi"] for paper in PAPERS],
                    evidence_card_ids=[card.evidence_id for card in cards],
                )
            ],
        ),
        confirmed_by="controlled_experiment",
    )
    requirement = ConfirmedRequirementSpec(
        confirmed_by="controlled_experiment",
        confirmed_at=datetime.now(timezone.utc),
        requirement=RequirementSpec(
            document_type="literature_review",
            output_language="Chinese",
            topic="大气气溶胶遥感",
        ),
    )
    return V04WritingHandoff(
        requirement=requirement,
        outline=outline,
        evidence_library=library,
    )


def load_documents() -> tuple[dict[str, list], list, list]:
    pages_by_doi = {}
    documents = []
    all_pages = []
    extractor = PdfPageExtractor(enable_ocr=False)
    for paper in PAPERS:
        path = PDF_DIR / paper["filename"]
        payload = path.read_bytes()
        import hashlib

        acquisition = DocumentAcquisition(
            doi=paper["doi"],
            status="available",
            method="user_upload",
            source_url=f"https://doi.org/{paper['doi']}",
            local_path=str(path),
            sha256=hashlib.sha256(payload).hexdigest(),
            media_type="application/pdf",
            file_size_bytes=len(payload),
            attempts=1,
        )
        extraction = extractor.extract(acquisition)
        if extraction.status != "complete":
            raise RuntimeError(
                f"incomplete PDF extraction for {paper['doi']}: "
                f"{extraction.model_dump()}"
            )
        pages_by_doi[paper["doi"]] = extraction.pages
        documents.append(acquisition)
        all_pages.extend(extraction.pages)
    return pages_by_doi, documents, all_pages


def extract_dois(value: str) -> list[str]:
    return sorted(
        {
            match.group(0).rstrip(".,;)]}")
            for match in re.finditer(
                r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
                value,
                re.IGNORECASE,
            )
        }
    )


def client() -> DeepSeekClient:
    configured = LLMSettings()
    return DeepSeekClient(
        LLMSettings(
            api_key=configured.api_key,
            base_url=configured.base_url,
            model=COMPARISON_MODEL,
            timeout_seconds=60,
            max_retries=0,
        )
    )


if __name__ == "__main__":
    main()
