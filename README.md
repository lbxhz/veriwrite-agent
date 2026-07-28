# VeriWrite Agent

VeriWrite 是一个“可验证、可追溯、分阶段协作”的课程研究与写作 Agent 项目。

当前版本为 **V0.1：课程要求获取与确认**。它把自然语言课程要求转换成经过 Pydantic 校验、双路比对和用户确认的 JSON，为后续文献检索、全文证据提取、分章节写作和审计提供稳定状态。

## 当前数据流

```text
课程要求文本或 DOCX
    -> RuleBasedRequirementParser ─┐
                                  ├-> RequirementReconciler
    -> LLMRequirementParser ──────┘
    -> RequirementCompletenessChecker
    -> requirement_review.json
    -> 用户确认与字段修正
    -> Pydantic 再校验
    -> confirmed_requirement_spec.json
    -> 后续的检索、证据和写作模块
```

规则解析器用于建立可重复测试的基线；双路模式会让规则与 LLM 分别输出同一个 `RequirementSpec` 数据合同。合并器只自动接受一致值或安全的单边值，不会让 LLM 静默覆盖冲突。

## 快速开始（Windows PowerShell）

```powershell
cd C:\Users\17811\Documents\Codex\2026-07-12\new-chat\outputs\veriwrite-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

旧版规则解析命令仍然可用：

```powershell
veriwrite parse-requirements `
  --input tests\fixtures\course_requirement.txt `
  --output runtime\requirement_spec.json
```

准备用户审查包。`rule` 模式不联网，`dual` 模式会调用 `.env` 中配置的 LLM：

```powershell
veriwrite prepare-requirements `
  --input tests\fixtures\course_requirement.txt `
  --output runtime\requirement_review.json `
  --mode dual
```

命令会同时在相邻位置生成 `requirement_review.md`，其中包含两条解析路径的候选结果、冲突和待确认事项。

根据审查包填写确认答案，可以参考
`examples/requirement_confirmation.example.json`，然后生成 V0.2 可直接消费的确认版本：

```powershell
veriwrite confirm-requirements `
  --review runtime\requirement_review.json `
  --answers examples\requirement_confirmation.example.json `
  --output runtime\confirmed_requirement_spec.json
```

## 项目结构

```text
veriwrite-agent/
├── docs/                       # PRD 与学习笔记
├── src/veriwrite_agent/
│   ├── models/                 # 系统承认的数据结构
│   ├── services/               # 解析、合并、完整性检查和确认
│   ├── llm/                    # 统一 LLM 接口和供应商适配
│   ├── config/                 # API Key 等运行配置
│   └── cli.py                  # 命令行入口
├── tests/
│   ├── fixtures/               # 固定测试输入
│   └── test_*.py               # 自动化验收标准
├── examples/                   # 用户确认等输入示例
├── .env.example                # 密钥字段示例，不存真实密钥
└── pyproject.toml              # 项目、依赖与工具配置
```

## V0.1 验收能力

- 提取至少 15000 字；
- 提取至少 60 篇参考文献；
- 将外文三分之一换算为至少 20 篇；
- 识别近 5 年是软偏好；
- 识别单处引用最多 4 篇；
- 发现“15000 字以上”与“1.5 万字左右”的表述差异；
- 不把模板示例题目误判为用户真实题目；
- 记录原文证据，便于人工复核。
- 支持 TXT、Markdown 和 DOCX 要求文件；
- 支持规则/LLM 双路解析和显式冲突记录；
- 阻止未确认主题或未解决冲突进入下一阶段；
- 保存确认人、确认时间、用户修改和仍然存在的警告。

概念与方案说明见
[`docs/v0.1_concepts_and_workflow.md`](docs/v0.1_concepts_and_workflow.md)。

## 路线图

- V0.1：课程要求获取与确认（当前）
- V0.2：文献检索、身份验证与 RIS 导出
- V0.3：PDF 全文、文献矩阵与证据卡
- V0.4：大纲、分章节长文与持久化状态
- V0.5：引用、要求、结构与语言审计
- V1.0：Web UI、Word 输出与完整评测报告
