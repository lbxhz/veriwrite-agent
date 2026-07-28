from types import SimpleNamespace

import pytest

from veriwrite_agent.services.ocr import OCRNoTextError, extract_image_text


class FakeEngine:
    def __init__(self, texts, scores) -> None:
        self._output = SimpleNamespace(txts=texts, scores=scores)

    def __call__(self, image):
        return self._output


def test_local_ocr_normalizes_lines_and_reports_average_confidence() -> None:
    result = extract_image_text(
        object(),
        engine=FakeEngine(
            (" 课程要求 ", "参考文献  不少于60篇"),
            (0.9, 0.8),
        ),
    )

    assert result.text == "课程要求\n参考文献 不少于60篇"
    assert result.average_confidence == pytest.approx(0.85)
    assert result.line_count == 2


def test_local_ocr_rejects_empty_output() -> None:
    with pytest.raises(OCRNoTextError, match="没有识别到"):
        extract_image_text(
            object(),
            engine=FakeEngine((), ()),
        )
