"""Command-line entry point for veriwrite-evaluator.

``veriwrite-eval mcp`` starts the MCP stdio server (for the VeriWrite agent).
One-shot modes also work: evaluate a file, compare two files, list rubrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from veriwrite_evaluator import adapter
from veriwrite_evaluator.config import EvaluatorSettings


def _read_text(path: str) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="veriwrite-eval", description="论文写作质量评定工具（基于 hermes-rubric，支持 MCP）")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-config", help="校验 API 配置")
    sub.add_parser("list-rubrics", help="列出可用中文评分模板")

    p_eval = sub.add_parser("evaluate", help="评定一篇论文正文")
    p_eval.add_argument("--target", required=True, help="待评定文件路径")
    p_eval.add_argument("--rubric", default="academic_writing_zh_v1")
    p_eval.add_argument("--window-bytes", type=int, default=0)
    p_eval.add_argument("--no-batch", action="store_true")
    p_eval.add_argument("--out", default=None, help="输出 JSON 文件路径")

    p_pair = sub.add_parser("pairwise", help="同题 A/B 对比评定")
    p_pair.add_argument("--text-a", required=True)
    p_pair.add_argument("--text-b", required=True)
    p_pair.add_argument("--rubric", default="academic_writing_zh_v1")
    p_pair.add_argument("--out", default=None)

    sub.add_parser("mcp", help="启动 MCP stdio server（供 VeriWrite Agent 调用）")

    args = parser.parse_args()

    if args.command == "check-config":
        try:
            settings = EvaluatorSettings.from_env()
        except ValueError as error:
            print(f"配置错误: {error}", file=sys.stderr)
            sys.exit(1)
        print(f"OK: base_url={settings.base_url} model={settings.model} batch={settings.batch} window_bytes={settings.target_window_bytes}")
        return

    if args.command == "list-rubrics":
        for rubric in adapter.list_rubrics():
            print(f"{rubric['id']}  hash={rubric['rubric_hash']}  dims={len(rubric['dimensions'])}")
        return

    if args.command == "mcp":
        from veriwrite_evaluator.mcp_server import main as mcp_main

        mcp_main()
        return

    if args.command == "evaluate":
        window = None if args.window_bytes <= 0 else args.window_bytes
        result = adapter.evaluate_writing(
            _read_text(args.target),
            rubric_id=args.rubric,
            target_path=args.target,
            batch=not args.no_batch,
            target_window_bytes=window,
        )
        _emit(result, args.out)
        return

    if args.command == "pairwise":
        result = adapter.evaluate_pairwise(
            _read_text(args.text_a),
            _read_text(args.text_b),
            rubric_id=args.rubric,
        )
        _emit(result, args.out)
        return


def _emit(payload: dict, out: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if out:
        path = Path(out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"已写入 {path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
