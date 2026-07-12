"""Command-line interface for VeriWrite."""

from __future__ import annotations

import argparse
from pathlib import Path

from veriwrite_agent.services.requirement_parser import RuleBasedRequirementParser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veriwrite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_requirements = subparsers.add_parser(
        "parse-requirements", help="Parse a UTF-8 course requirement text file."
    )
    parse_requirements.add_argument("--input", type=Path, required=True)
    parse_requirements.add_argument("--output", type=Path, required=True)
    return parser


def run_parse_requirements(input_path: Path, output_path: Path) -> int:
    text = input_path.read_text(encoding="utf-8")
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


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "parse-requirements":
        return run_parse_requirements(args.input, args.output)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

