"""End-to-end MCP stdio integration: spawn the real server, list tools, call a no-LLM tool."""

from __future__ import annotations

import asyncio
import sys

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def test_list_and_call_tools():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "veriwrite_evaluator.mcp_server"],
    )

    async def scenario():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert {"list_rubrics", "get_rubric", "evaluate_writing", "evaluate_pairwise"} <= names

                result = await session.call_tool("list_rubrics", {})
                assert result.isError is False
                return names, result

    names, result = _run(scenario())
    # list_rubrics needs no API key; its content should mention the zh template.
    payload = " ".join(str(c.text) for c in result.content)
    assert "academic_writing_zh_v1" in payload
    assert "evaluate_writing" in names
