"""Command-line interface for VeriWrite."""

from __future__ import annotations

import argparse
from pathlib import Path

from veriwrite_agent.config.settings import LLMSettings
from veriwrite_agent.llm.deepseek_client import DeepSeekClient
from veriwrite_agent.models.requirement_workflow import (
    RequirementConfirmation,
    RequirementReviewPackage,
)
from veriwrite_agent.services.llm_requirement_parser import LLMRequirementParser
from veriwrite_agent.services.requirement_confirmation import (
    RequirementConfirmationError,
    RequirementConfirmationService,
)
from veriwrite_agent.services.requirement_input import load_requirement_text
from veriwrite_agent.services.requirement_parser import RuleBasedRequirementParser
from veriwrite_agent.services.requirement_pipeline import RequirementReviewPipeline
from veriwrite_agent.services.requirement_review_renderer import (
    RequirementReviewRenderer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veriwrite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_requirements = subparsers.add_parser(
        "parse-requirements",
        help="Legacy rule-only parse that writes one RequirementSpec.",
    )
    parse_requirements.add_argument("--input", type=Path, required=True)
    parse_requirements.add_argument("--output", type=Path, required=True)

    prepare_requirements = subparsers.add_parser(
        "prepare-requirements",
        help="Create a review package using rule-only or dual parsing.",
    )
    prepare_requirements.add_argument("--input", type=Path, required=True)
    prepare_requirements.add_argument("--output", type=Path, required=True)
    prepare_requirements.add_argument(
        "--summary-output",
        type=Path,
        help="Markdown confirmation form; defaults beside --output.",
    )
    prepare_requirements.add_argument(
        "--mode",
        choices=("rule", "dual"),
        default="rule",
        help="Use --mode dual to call the configured LLM as the second parser.",
    )

    confirm_requirements = subparsers.add_parser(
        "confirm-requirements",
        help="Apply user answers and write a confirmed requirement hand-off.",
    )
    confirm_requirements.add_argument("--review", type=Path, required=True)
    confirm_requirements.add_argument("--answers", type=Path, required=True)
    confirm_requirements.add_argument("--output", type=Path, required=True)

    subparsers.add_parser(
        "check-config", help="Validate local LLM configuration without making an API call."
    )
    return parser


def run_parse_requirements(input_path: Path, output_path: Path) -> int:
    text = load_requirement_text(input_path)
    spec = RuleBasedRequirementParser().parse(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    print(f"Parsed: {input_path}")
    print(f"Output: {output_path}")
    print(f"Minimum characters: {spec.length.minimum_chars}")
    print(f"Minimum references: {spec.references.minimum_total}")
    print(f"Minimum foreign references: {spec.references.minimum_foreign_count}")
    print(f"Ambiguities: {len(spec.ambiguities)}")
    return 0


def run_prepare_requirements(
    input_path: Path,
    output_path: Path,
    *,
    mode: str,
    summary_output_path: Path | None = None,
) -> int:
    text = load_requirement_text(input_path)
    llm_parser = None
    if mode == "dual":
        settings = LLMSettings()
        llm_parser = LLMRequirementParser(DeepSeekClient(settings))

    review = RequirementReviewPipeline(
        RuleBasedRequirementParser(),
        llm_parser=llm_parser,
    ).prepare(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
    summary_path = summary_output_path or output_path.with_suffix(".md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        RequirementReviewRenderer().render_markdown(review),
        encoding="utf-8",
    )

    print(f"Prepared: {input_path}")
    print(f"Review package: {output_path}")
    print(f"Confirmation form: {summary_path}")
    print(f"Parser mode: {review.parser_mode}")
    print(f"Conflicts: {len(review.reconciliation.conflicts)}")
    print(f"Blocking issues: {review.completeness.blocking_count}")
    print(f"Warnings: {review.completeness.warning_count}")
    print(f"Status: {review.status}")
    return 0


def run_confirm_requirements(
    review_path: Path,
    answers_path: Path,
    output_path: Path,
) -> int:
    review = RequirementReviewPackage.model_validate_json(
        review_path.read_text(encoding="utf-8")
    )
    confirmation = RequirementConfirmation.model_validate_json(
        answers_path.read_text(encoding="utf-8")
    )
    confirmed = RequirementConfirmationService().confirm(review, confirmation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        confirmed.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(f"Confirmed requirement: {output_path}")
    print(f"Confirmed by: {confirmed.confirmed_by}")
    print(f"Remaining warnings: {len(confirmed.remaining_warnings)}")
    return 0


def run_check_config() -> int:
    summary = LLMSettings().public_summary()
    print("LLM configuration is valid.")
    print(f"API key configured: {summary['api_key_configured']}")
    print(f"Base URL: {summary['base_url']}")
    print(f"Model: {summary['model']}")
    print(f"Timeout seconds: {summary['timeout_seconds']}")
    print(f"Max retries: {summary['max_retries']}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "parse-requirements":
        return run_parse_requirements(args.input, args.output)
    if args.command == "prepare-requirements":
        return run_prepare_requirements(
            args.input,
            args.output,
            mode=args.mode,
            summary_output_path=args.summary_output,
        )
    if args.command == "confirm-requirements":
        try:
            return run_confirm_requirements(
                args.review,
                args.answers,
                args.output,
            )
        except RequirementConfirmationError as exc:
            parser.error(str(exc))
    if args.command == "check-config":
        return run_check_config()
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
