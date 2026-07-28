"""Local OCR boundary used by image and scanned-PDF requirement inputs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol


class OCRUnavailableError(RuntimeError):
    """Raised when the optional local OCR runtime is not installed."""


class OCRNoTextError(ValueError):
    """Raised when OCR completes but finds no usable text."""


class OCREngine(Protocol):
    def __call__(self, image: Any) -> Any:
        """Return an object exposing txts and scores."""


@dataclass(frozen=True)
class OCRTextResult:
    text: str
    average_confidence: float
    line_count: int


def extract_image_text(
    image: Any,
    *,
    engine: OCREngine | None = None,
) -> OCRTextResult:
    """Run local OCR and return text plus a transparent quality signal."""

    active_engine = engine or _get_engine()
    output = active_engine(image)
    raw_texts = tuple(getattr(output, "txts", ()) or ())
    raw_scores = tuple(getattr(output, "scores", ()) or ())

    lines: list[str] = []
    scores: list[float] = []
    for index, raw_text in enumerate(raw_texts):
        text = " ".join(str(raw_text).split())
        if not text:
            continue
        lines.append(text)
        if index < len(raw_scores) and raw_scores[index] is not None:
            scores.append(float(raw_scores[index]))

    if not lines:
        raise OCRNoTextError("OCR 没有识别到可用文本。")

    average = sum(scores) / len(scores) if scores else 0.0
    return OCRTextResult(
        text="\n".join(lines),
        average_confidence=average,
        line_count=len(lines),
    )


@lru_cache(maxsize=1)
def _get_engine() -> OCREngine:
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise OCRUnavailableError(
            "本地 OCR 尚未安装；请执行 pip install -e \".[ocr]\"。"
        ) from exc
    return RapidOCR()
