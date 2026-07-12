# VeriWrite Agent

VeriWrite 是一个“可验证、可追溯、分阶段协作”的课程研究与写作 Agent 项目。

当前版本为 **V0.1：课程要求解析器**。它把自然语言课程要求转换成经过 Pydantic 校验的 JSON，为后续文献检索、全文证据提取、分章节写作和审计提供稳定状态。

## 当前数据流

```text
课程要求文本
    -> RuleBasedRequirementParser
    -> RequirementSpec (Pydantic)
    -> requirement_spec.json
    -> 后续的检索、证据和写作模块
```

V0.1 故意不调用 LLM。规则解析器用于建立可重复测试的基线；后续增加 LLM 解析器时，两者必须输出同一个 `RequirementSpec`，并用同一批测试数据比较。

## 快速开始（Windows PowerShell）

```powershell
cd C:\Users\17811\Documents\Codex\2026-07-12\new-chat\outputs\veriwrite-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

运行解析器：

```powershell
veriwrite parse-requirements `
  --input tests\fixtures\course_requirement.txt `
  --output runtime\requirement_spec.json
```

也可以不安装命令行入口，直接运行模块：

```powershell
python -m veriwrite_agent.cli parse-requirements `
  --input tests\fixtures\course_requirement.txt `
  --output runtime\requirement_spec.json
```

## 项目结构

```text
veriwrite-agent/
├── docs/                       # PRD 与学习笔记
├── src/veriwrite_agent/
│   ├── models/                 # 系统承认的数据结构
│   ├── services/               # 解析等业务能力
│   └── cli.py                  # 命令行入口
├── tests/
│   ├── fixtures/               # 固定测试输入
│   └── test_*.py               # 自动化验收标准
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

## 路线图

- V0.1：课程要求解析器（当前）
- V0.2：文献检索、身份验证与 RIS 导出
- V0.3：PDF 全文、文献矩阵与证据卡
- V0.4：大纲、分章节长文与持久化状态
- V0.5：引用、要求、结构与语言审计
- V1.0：Web UI、Word 输出与完整评测报告

