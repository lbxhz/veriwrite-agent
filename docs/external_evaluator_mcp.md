# VeriWrite 外部写作评分 MCP 集成

## 角色边界

VeriWrite 同时保留两套互补评价：

- 确定性六维评分卡负责需求、文献、证据、引用和交付门禁。
- `veriwrite-evaluator` 基于 `hermes-rubric 1.0.2`，负责结构、论证、衔接、去重、学术语域和反 AI 套话。

外部评分不能证明事实或引文正确，也不能覆盖确定性阻塞项。

## MCP 客户端

客户端位于 `src/veriwrite_agent/services/external_writing_evaluator.py`，通过受控的
Python 子进程启动 `veriwrite_evaluator.mcp_server`。只允许调用：

- `list_rubrics`
- `get_rubric`
- `evaluate_writing`
- `evaluate_pairwise`

每份结果都经过 Pydantic 合同校验，并额外检查：

1. 新评测的 `evaluation_method` 必须使用
   `hermes-rubric-v2|scope=full-document` 和 `deepseek-openai`；历史 v1 结果只可读取，
   不可复用或与全文结果比较；
2. 指纹中的 `rubric_hash` 必须与 receipt 中的阶段一哈希一致；
3. receipt 的输入哈希必须命中实际送评文本；
4. 两份结果只有在完整 `evaluation_method` 相同时才允许比较。

## 安装与配置

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\veriwrite-evaluator[mcp,dev]"
```

客户端优先读取 `VW_EVAL_API_KEY`，否则复用 `LLM_API_KEY`。评分模型始终锁定为
`deepseek-chat`，温度锁定为 `0.0`；`VW_EVAL_BATCH`、
`VW_EVAL_TARGET_WINDOW_BYTES` 和 `VW_EVAL_TIMEOUT_SECONDS` 可以调整。学术论文评测
使用 `hermes-rubric-v2|scope=full-document` 指纹；适配层会把窗口扩展到完整正文，
并在 receipt 中记录 UTF-8 字节覆盖率。超过
`VW_EVAL_MAX_FULL_DOCUMENT_BYTES` 的文档会明确失败，不会用正文前缀冒充全文评分。

## V0.5 行为

论文通过硬性审计、进入可确认或已确认状态后，V0.5 自动运行外部评分。评分按论文
输入哈希缓存；正文没有变化时不会重复调用。服务不可用时只显示降级说明，不阻塞
Markdown、DOCX 和审计包交付。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_external_writing_evaluator.py veriwrite-evaluator\tests -q
.\.venv\Scripts\python.exe -m ruff check src tests veriwrite-evaluator
```
