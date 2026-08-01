"""Reliable V0.3 PDF extraction, page retrieval, and resumable caches."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from veriwrite_agent.models.evidence import (
    DocumentAcquisition,
    DocumentExtractionResult,
    DocumentPage,
    EvidenceCard,
    EvidencePageSelection,
)


class EvidencePageRetriever:
    """Select relevant pages by deterministic lexical scoring after full extraction."""

    def __init__(self, *, max_pages: int = 12) -> None:
        if not 2 <= max_pages <= 30:
            raise ValueError("max_pages must be between 2 and 30")
        self._max_pages = max_pages

    def select(
        self,
        *,
        doi: str,
        theme_id: str,
        query_text: str,
        pages: list[DocumentPage],
    ) -> tuple[EvidencePageSelection, list[DocumentPage]]:
        if not pages:
            raise ValueError("page retrieval requires extracted PDF text")
        ordered = sorted(pages, key=lambda page: page.page_number)
        if any(page.doi != doi for page in ordered):
            raise ValueError("page retrieval cannot mix DOI values")
        terms = _query_terms(query_text)
        scores = {page.page_number: _page_score(page.text, terms) for page in ordered}
        selected_numbers = {ordered[0].page_number}
        ranked = sorted(
            ordered,
            key=lambda page: (-scores[page.page_number], page.page_number),
        )
        for page in ranked:
            if len(selected_numbers) >= min(self._max_pages, len(ordered)):
                break
            selected_numbers.add(page.page_number)

        selected = [page for page in ordered if page.page_number in selected_numbers]
        audit = EvidencePageSelection(
            doi=doi,
            theme_id=theme_id,
            query_text=query_text,
            total_extracted_pages=len(ordered),
            selected_page_numbers=[page.page_number for page in selected],
            page_scores={page.page_number: round(scores[page.page_number], 6) for page in selected},
        )
        return audit, selected


class EvidenceRuntimeCache:
    """Persist extraction and LLM card checkpoints under a policy-bound run key."""

    def __init__(self, root: Path, *, policy_fingerprint: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", policy_fingerprint):
            raise ValueError("policy_fingerprint must be a SHA-256 hex digest")
        self._root = root / policy_fingerprint[:16]

    def load_extraction(
        self,
        acquisition: DocumentAcquisition,
    ) -> DocumentExtractionResult | None:
        path = self._document_dir(acquisition) / "extraction.json"
        if not path.is_file():
            return None
        try:
            result = DocumentExtractionResult.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if result.doi != acquisition.doi or result.document_sha256 != acquisition.sha256:
            return None
        return result

    def save_extraction(self, result: DocumentExtractionResult) -> None:
        directory = self._root / result.document_sha256
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(directory / "extraction.json", result.model_dump_json(indent=2))

    def load_cards(
        self,
        acquisition: DocumentAcquisition,
        *,
        title: str,
        selection: EvidencePageSelection,
    ) -> list[EvidenceCard] | None:
        path = self._document_dir(acquisition) / "cards.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != _card_signature(title, selection):
                return None
            return [EvidenceCard.model_validate(item) for item in payload["cards"]]
        except Exception:
            return None

    def save_cards(
        self,
        acquisition: DocumentAcquisition,
        *,
        title: str,
        selection: EvidencePageSelection,
        cards: list[EvidenceCard],
    ) -> None:
        directory = self._document_dir(acquisition)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            directory / "cards.json",
            json.dumps(
                {
                    "schema_version": "0.3.1-cache",
                    "signature": _card_signature(title, selection),
                    "selection": selection.model_dump(mode="json"),
                    "cards": [card.model_dump(mode="json") for card in cards],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    def _document_dir(self, acquisition: DocumentAcquisition) -> Path:
        if not acquisition.sha256:
            raise ValueError("evidence cache requires an available hashed PDF")
        return self._root / acquisition.sha256


def _query_terms(value: str) -> list[str]:
    latin = [item.casefold() for item in re.findall(r"[A-Za-z0-9]{3,}", value)]
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", value)
    chinese = [
        chunk[index : index + 2]
        for chunk in chinese_chunks
        for index in range(max(1, len(chunk) - 1))
    ]
    return list(dict.fromkeys([*latin, *chinese]))


def _page_score(text: str, terms: list[str]) -> float:
    normalized = " ".join(text.casefold().split())
    term_score = sum(normalized.count(term) for term in terms)
    section_bonus = sum(
        marker in normalized
        for marker in (
            "abstract",
            "introduction",
            "method",
            "result",
            "discussion",
            "conclusion",
            "摘要",
            "方法",
            "结果",
            "结论",
        )
    )
    return float(term_score + section_bonus * 0.25)


def _card_signature(title: str, selection: EvidencePageSelection) -> str:
    canonical = json.dumps(
        {
            "pipeline_version": "evidence-passage-selection-v2",
            "title": title,
            "selection": selection.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
