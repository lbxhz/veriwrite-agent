# veriwrite-evaluator

论文写作质量评定工具：以成熟开源项目 [hermes-rubric](https://github.com/hermes-labs-ai/hermes-rubric)（MIT）为引擎做薄本地化，并通过 **MCP（stdio）** 供 VeriWrite 写作 Agent 调用。

## 为什么薄封装而不是自研

hermes-rubric 已内置三阶段证据锚定评分协议：

1. 合成 rubric（我们用**确定性中文模板**替代 LLM 合成，解决"不固定"）
2. 收集证据引用（每维度输出引用原文的证据；引用未逐字命中会被剔除）
3. 只按证据打分（0–10；证据薄时钳制到 [3,7] 并标记 hedge；评分 prompt 内置"不奖励表面流畅"对抗门禁）

并输出 `rubric_hash`（钉住测量标准）与可复现 `receipt`（输入哈希/后端/时间戳）。

本仓库只做三处本地化：

| 改动 | 文件 |
|---|---|
| 中文学术评分模板 | `src/veriwrite_evaluator/templates/academic_writing_zh_v1.yaml` |
| OpenAI 兼容（DeepSeek）后端插件 | `src/veriwrite_evaluator/deepseek_backend.py` |
| MCP 适配层 | `src/veriwrite_evaluator/adapter.py`、`mcp_server.py` |

## 安装

```bash
pip install -e ".[mcp,dev]"   # 或只装运行依赖: pip install -e .
```

## 配置（环境变量）

| 变量 | 默认 |
|---|---|
| `VW_EVAL_API_KEY` / `LLM_API_KEY` | 必填 |
| `VW_EVAL_BASE_URL` / `LLM_BASE_URL` | `https://api.deepseek.com` |
| `VW_EVAL_MODEL` | `deepseek-chat` |
| `VW_EVAL_TEMPERATURE` | `0.0`（固定以保证可复现） |
| `VW_EVAL_TARGET_WINDOW_BYTES` | `120000`（适配层会自动扩展到完整正文） |
| `VW_EVAL_BATCH` | `1` |

## 用法

```bash
# MCP stdio server（VeriWrite 通过 MCP 客户端调用）
veriwrite-eval mcp

# 单次评定
veriwrite-eval evaluate --target paper.md --out result.json

# 同题 A/B 对比
veriwrite-eval pairwise --text-a v1.md --text-b v2.md --out cmp.json

# 查看模板
veriwrite-eval list-rubrics
veriwrite-eval check-config
```

MCP 工具：`list_rubrics` / `get_rubric` / `evaluate_writing` / `evaluate_pairwise`。

## 与 VeriWrite 的约定

- 输出含 `evaluation_method` 指纹（`hermes-rubric-v2|scope=full-document|rubric=...|rubric_hash=...|backend=...`），与 VeriWrite `compare()` 的版本隔离约定兼容，跨评估方法或局部/全文口径不宣称可比较。
- `aggregate` 为 0–10，`aggregate_100` 为 0–100 双刻度，便于接入 VeriWrite 六维评分卡的展示。
