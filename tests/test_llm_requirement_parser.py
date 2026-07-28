import pytest

from veriwrite_agent.llm.fake_client import FakeLLMClient
from veriwrite_agent.models.requirements import (
    LengthRequirement,
    ReferenceRequirement,
    RequirementSpec,
)
from veriwrite_agent.services.llm_requirement_parser import (
    LLMOutputValidationError,
    LLMRequirementParser,
)


def test_llm_parser_validates_fake_json_without_api_call() -> None:
    expected = RequirementSpec(
        document_type="research_direction_literature_review",
        length=LengthRequirement(minimum_chars=15000),
        references=ReferenceRequirement(
            minimum_total=60,
            minimum_foreign_ratio=1 / 3,
        ),
    )
    fake = FakeLLMClient(expected.model_dump_json())

    result = LLMRequirementParser(fake).parse("课程要求原文")

    assert result.length.minimum_chars == 15000
    assert result.references.minimum_foreign_count == 20
    assert len(fake.calls) == 1
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_llm_parser_rejects_invalid_output() -> None:
    fake = FakeLLMClient('{"document_type": 123}')

    with pytest.raises(LLMOutputValidationError):
        LLMRequirementParser(fake).parse("课程要求原文")

    assert len(fake.calls) == 2


def test_llm_parser_repairs_one_invalid_json_response() -> None:
    valid = RequirementSpec(document_type="review").model_dump_json()

    class SequencedClient:
        def __init__(self):
            self.responses = iter(['{"document_type": 123}', valid])
            self.calls = []

        def complete(self, messages, *, response_format=None):
            self.calls.append(list(messages))
            return next(self.responses)

    client = SequencedClient()

    result = LLMRequirementParser(client).parse("课程要求原文")

    assert result.document_type == "review"
    assert len(client.calls) == 2
    assert "字段错误" in client.calls[1][-1]["content"]
