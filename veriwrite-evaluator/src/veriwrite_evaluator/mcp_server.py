"""MCP server exposing the evaluator to the VeriWrite agent.

Runs over stdio by default (``veriwrite-eval mcp``). Tools:
- ``list_rubrics`` / ``get_rubric``
- ``evaluate_writing``
- ``evaluate_pairwise``
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from veriwrite_evaluator import adapter

server = FastMCP("veriwrite-evaluator")


@server.tool()
def list_rubrics() -> list[dict]:
    """列出可用中文评分模板（含锁定的 rubric_hash）。"""
    return adapter.list_rubrics()


@server.tool()
def get_rubric(rubric_id: str = "academic_writing_zh_v1") -> dict:
    """获取指定评分模板的完整定义（维度、权重、证据指引）。"""
    return adapter.load_rubric(rubric_id)


@server.tool()
def evaluate_writing(
    target_text: str,
    rubric_id: str = "academic_writing_zh_v1",
    target_window_bytes: int = 0,
    batch: bool = True,
) -> dict:
    """对一篇论文正文做写作质量评定。

    - ``target_text``: 待评定的论文正文文本（Markdown 或纯文本）。
    - ``rubric_id``: 评分模板 id，默认 academic_writing_zh_v1。
    - ``target_window_bytes``: 参与评定的正文窗口字节数（0 表示使用默认值；
      适配层始终扩展到完整正文，禁止静默截断）。
    - ``batch``: 是否把多维证据收集/打分合并为更少的 LLM 调用。
    返回：0-10 与 0-100 双刻度聚合分、各维度分、证据引用、hedge 提示、evaluation_method 指纹与可复现凭证。
    """
    window = None if target_window_bytes <= 0 else target_window_bytes
    return adapter.evaluate_writing(
        target_text,
        rubric_id=rubric_id,
        target_window_bytes=window,
        batch=batch,
    )


@server.tool()
def evaluate_pairwise(
    text_a: str,
    text_b: str,
    rubric_id: str = "academic_writing_zh_v1",
    target_window_bytes: int = 0,
    batch: bool = True,
) -> dict:
    """对同题两份文本做 A/B 对比评定（同一锁定模板、同一评判模型）。

    返回各维度差值、总体差值及偏好（A/B/tie），并附双方完整评定结果。
    """
    window = None if target_window_bytes <= 0 else target_window_bytes
    return adapter.evaluate_pairwise(
        text_a,
        text_b,
        rubric_id=rubric_id,
        target_window_bytes=window,
        batch=batch,
    )


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
