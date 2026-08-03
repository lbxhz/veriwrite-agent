from veriwrite_agent.models.requirement_workflow import ConfirmedRequirementSpec
from veriwrite_agent.models.requirements import (
    LengthRequirement,
    PolicyRule,
    ReferenceRequirement,
    RequirementSpec,
)
from veriwrite_agent.services.requirement_policy import (
    RequirementPolicyCompiler,
    candidate_source_restriction_reasons,
)
from veriwrite_agent.models.literature_discovery import LiteratureCandidate


def test_compiles_one_policy_and_enforces_source_restrictions() -> None:
    confirmed = ConfirmedRequirementSpec(
        confirmed_by="student",
        requirement=RequirementSpec(
            document_type="research_direction_literature_review",
            output_language="English",
            topic="Atmospheric remote sensing",
            topic_source="explicit",
            length=LengthRequirement(
                minimum_words=600,
                maximum_words=800,
                counting_policy="words",
            ),
            references=ReferenceRequirement(
                minimum_total=6,
                minimum_foreign_ratio=1 / 3,
                recent_year_window=5,
                recent_year_rule_strength="hard",
                restriction_rules=[
                    PolicyRule(
                        category="source_restriction",
                        description="禁止引用 MDPI 旗下所有期刊和 IEEE Access",
                    )
                ],
                bibliography_style="Remote Sensing of Environment",
            ),
        ),
    )

    policy = RequirementPolicyCompiler(current_year=2026).compile(confirmed)

    assert policy.length.target_units == 800
    assert policy.references.minimum_foreign_count == 2
    assert (policy.references.year_from, policy.references.year_to) == (2022, 2026)
    assert {item.enforcement for item in policy.coverage} >= {"enforced", "audited"}
    candidate = LiteratureCandidate(
        doi="10.1000/mdpi.1",
        title="A real but prohibited source",
        journal_title="Remote Sensing",
        publisher="MDPI AG",
        source_provider="crossref",
    )
    assert candidate_source_restriction_reasons(policy, candidate) == ["source_restriction_rule_0"]


def test_resolved_length_ambiguity_is_not_reported_as_unresolved() -> None:
    confirmed = ConfirmedRequirementSpec(
        confirmed_by="student",
        requirement=RequirementSpec(
            document_type="research_direction_literature_review",
            output_language="Chinese",
            topic="大气遥感",
            length=LengthRequirement(
                minimum_chars=15000,
                target_chars=15000,
                counting_policy="chinese_chars_and_english_words",
            ),
            references=ReferenceRequirement(bibliography_style="GB/T 7714—2015"),
            ambiguities=[
                "同一文件同时使用“至少/以上”和“左右”描述字数，默认采用更严格的最低字数。"
            ],
        ),
    )

    policy = RequirementPolicyCompiler(current_year=2026).compile(confirmed)

    assert policy.unresolved_requirements == []
    assert any(
        "resolved by enforcing the stricter confirmed minimum" in note
        for note in policy.resolution_notes
    )
