import pytest
from pydantic import ValidationError

from veriwrite_agent.models.requirements import LengthRequirement, ReferenceRequirement


def test_foreign_reference_count_rounds_up() -> None:
    requirement = ReferenceRequirement(minimum_total=61, minimum_foreign_ratio=1 / 3)
    assert requirement.minimum_foreign_count == 21


def test_invalid_foreign_ratio_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ReferenceRequirement(minimum_total=60, minimum_foreign_ratio=1.2)


def test_target_cannot_be_below_minimum() -> None:
    with pytest.raises(ValidationError):
        LengthRequirement(minimum_chars=15000, target_chars=12000)

