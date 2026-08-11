"""veriwrite-evaluator: MCP-callable academic writing quality evaluation.

Thin localization over the mature ``hermes-rubric`` engine:
- deterministic Chinese rubric templates (no LLM rubric synthesis);
- OpenAI-compatible DeepSeek backend via hermes-rubric's plugin registry;
- FastMCP stdio server so the VeriWrite writing agent can call it over MCP.
"""

__version__ = "0.2.0"

from veriwrite_evaluator.adapter import evaluate_pairwise, evaluate_writing, list_rubrics, load_rubric
from veriwrite_evaluator.config import EvaluatorSettings

__all__ = [
    "EvaluatorSettings",
    "evaluate_pairwise",
    "evaluate_writing",
    "list_rubrics",
    "load_rubric",
    "__version__",
]
