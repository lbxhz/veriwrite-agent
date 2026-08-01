# VeriWrite Agent

VeriWrite 是一个“可验证、可追溯、分阶段协作”的课程研究与写作 Agent 项目。

当前开发版本为 **MVP 1.0：要求贯通、证据约束写作与最终论文交付**。V0.1.2 负责把自然语言课程要求
转换成经过 Pydantic 校验、双路比对和用户确认的 JSON；V0.2 根据确认需求生成临时检索
蓝图，通过 Crossref RIS 与 DOI.org 验证文献身份，再按主题相关性、地大2023等级、
挪威国家目录2025等级和年份进行可解释选择。

V0.3在此基础上建立核心论文人机协作下载队列：用户只处理出版社登录、验证码和下载
按钮，Agent自动扫描下载目录、核对DOI/题名、检查PDF完整性并保留文件哈希与页码。
随后将LLM归纳结果约束为带短原文的`EvidenceCard`，再生成每个单元格都能追溯到
证据卡的`LiteratureMatrix`。非核心论文只保留已验证元数据，不冒充全文已验证。
V0.4按确认大纲为每章建立独立证据包。DeepSeek只负责组织段落并声明使用了哪些
证据编号，最终引用键、DOI和PDF页码由程序绑定。章节必须逐一确认，课程禁止AI代写时
系统会在调用模型前停止，并改为导出人工写作证据包。

## 当前数据流

```text
课程要求文件（TXT / Markdown / DOCX / DOC / PDF / 多张图片）
    -> 原生文本或本地 OCR -> 连续截图去重 -> 人工校对
    -> RuleBasedRequirementParser ─┐
                                  ├-> RequirementReconciler
    -> LLMRequirementParser ──────┘
    -> RequirementCompletenessChecker
    -> requirement_review.json
    -> 用户确认与字段修正
    -> Pydantic 再校验
    -> confirmed_requirement_spec.json
    -> V0.2检索、验证与均衡选择
    -> V0.3核心PDF + 非核心元数据双层文献库
    -> 确认证据库与最终写作大纲
    -> v04_writing_handoff.json
```

规则解析器用于建立可重复测试的基线；双路模式会让规则与 LLM 分别输出同一个 `RequirementSpec` 数据合同。合并器只自动接受一致值或安全的单边值，不会让 LLM 静默覆盖冲突。

## 快速开始（Windows PowerShell）

```powershell
cd C:\Users\17811\Documents\Codex\2026-07-12\new-chat\outputs\veriwrite-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,ui,ocr]"
pytest
```

启动完整 MVP 本地工作台：

```powershell
streamlit run streamlit_app.py
```

工作台左侧按 `V0.1 需求确认 → V0.2 文献检索 → V0.3 全文证据 →
V0.4 逐章写作 → 最终交付` 导航。总览页显示各阶段的完成状态、阻塞原因和下一步，
尚未满足前置条件的阶段不会静默消失。可随时导出 `veriwrite_mvp_project.json`
项目检查点，之后恢复已提取文本和各阶段数据合同；检查点不会包含 `.env` 密钥或
原始上传文件。

主路径只要求用户处理会改变结果的决定：需求冲突、检索前授权、章节接受和最终交付。
确认人、重复勾选、技术 JSON、候选池参数和阶段审计产物不会长期占据主界面；它们由
上游身份与确定性代码记录，或收纳在高级区。完整交互取舍见
[`docs/ux_simplification.md`](docs/ux_simplification.md)。

工作台内置 5 份基础样例和 1 份复杂多选金标准，也可以上传自己的 TXT、
Markdown、DOCX、旧版 DOC、PDF 或图片。旧版 DOC 会调用本机 Microsoft Word
进行只读转换；PNG、JPG、TIFF 等图片以及 PDF 中的扫描页会通过
RapidOCR + ONNX Runtime 在本地识别。图片不会直接发送给 DeepSeek，
只有 OCR 后的文本会进入双路解析。多张连续截图可按阅读顺序一次上传，
系统会清除常见手机界面噪声并合并重叠段落；OCR 文本可在界面中人工校正后重跑。
界面会显示规则与 DeepSeek 的字段级对照、实质冲突、完整性问题和原文证据，
并允许用户逐项裁决后下载最终数据合同。

V0.1 最终需求确认后，同一个控制台会继续生成临时检索蓝图。蓝图经用户集中检查和确认后，
才能执行 Crossref 分主题检索、RIS/DOI 验证、DeepSeek 受限相关性评分与均衡选文。
真实运行按蓝图指纹缓存在 `runtime/literature_console/`，中断后可以继续，并可下载最终
文献 JSON、RIS 与逐篇真实性证据。完整操作说明见
[`docs/integrated_v0.1_v0.2_console.md`](docs/integrated_v0.1_v0.2_console.md)。

V0.2选文完成后，控制台继续提供核心论文下载队列。用户处理出版社登录、验证码和下载
按钮，系统扫描下载目录并识别正确PDF、错误网页、重复文件和缺失项。通过检查的PDF会
按页提取文本，由代码生成带ID的原文片段；DeepSeek只选择片段ID并归纳主张，原文引句
由代码回填并校验页码与PDF哈希，避免模型抄写时改动原句。无阻塞项且用户确认最终大纲
后，可下载`v04_writing_handoff.json`。V0.4逐章确认完成后，最终交付页负责生成并
审计标题、摘要、关键词、结论、参考文献与AI声明，最终可下载Markdown、DOCX和完整
合规审计包。

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
│   ├── ui/                     # 本地验证工作台的应用层与界面
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
- 支持 TXT、Markdown、DOCX、旧版 DOC、PDF 和常见图片；
- 对图片与 PDF 扫描页执行本地中英文 OCR，并显示平均置信度；
- 提供 5 份内置样例和本地上传验证工作台；
- 提供复杂多教师案例金标准，将 4 个可选方向与统一要求分层保存；
- 支持多张滚动截图按序合并、重叠去重、手机界面噪声清理和 OCR 文本校对；
- 区分“约 30 篇”目标数量与最低数量，原样保存 4000–5000 单词范围；
- 保存选题约束、提交方式与截止时间、文献禁用规则、处罚和 AI 声明要求；
- 用户选择教师/方向后，将该档案与统一要求物化为 V0.2 可消费的最终合同；
- LLM JSON 校验失败时自动按字段错误修复一次，仍失败则显示字段级原因；
- 支持规则/LLM 双路解析和显式冲突记录；
- 阻止未确认主题或未解决冲突进入下一阶段；
- 保存确认人、确认时间、用户修改和仍然存在的警告。

概念与方案说明见
[`docs/v0.1_concepts_and_workflow.md`](docs/v0.1_concepts_and_workflow.md)。

## V0.2.3 文献选择能力

- 从 V0.1 的确认需求生成 2—8 个需要文献支撑的临时检索主题；
- 临时检索蓝图必须经用户检查和确认，草案不能直接触发 Crossref 检索；
- 为不同主题分配明确文献配额，避免单一检索词占满最终结果；
- 同一主题的多条 Crossref 查询采用公平轮询；
- 只有通过权威 RIS 与 DOI 解析验证的文献才能进入相关性评分；
- LLM 只能判断真实题名/摘要与主题的贴切度，不能新增 DOI 或修改元数据；
- 最终顺序固定为“相关性 > 地大期刊等级 > 挪威2025等级 > 年份”；
- Crossref 返回的 ISSN 优先用于挪威目录匹配，规范化期刊题名只作兜底；
- 默认把两种等级都作为软偏好，未分级不等于虚假；
- 地大和挪威等级独立展示，不把 Level 2 伪装成地大 T2 或其他分区；
- 某主题不足时明确输出缺口，不用无关论文静默凑数；
- 网络验证与 LLM 评分按阶段缓存，中断后可恢复；
- 真实大气遥感回归实现四主题各 5 篇，最终 20/20 通过 RIS 与 DOI 验证。

设计与回归说明见
[`docs/v0.2.2_outline_guided_balanced_selection.md`](docs/v0.2.2_outline_guided_balanced_selection.md)
和
[`docs/v0.2.3_dual_journal_ranking.md`](docs/v0.2.3_dual_journal_ranking.md)。

## V0.3.0 全文证据能力

- 为用户选择的核心论文建立DOI权威入口与可恢复下载队列；
- 自动识别正确PDF、HTML拦截页、重复文件、无关文件和缺失论文；
- DOI或题名身份得分低于0.8时不自动分配，避免相似题名误匹配；
- 保存PDF SHA-256、页数、文件大小、身份依据和检查问题；
- 逐页提取原生文本，扫描页可进入本地OCR降级路径；
- 保存完整逐页提取清单，代码检索相关页面后才发送给LLM，完整PDF文本不被截断丢弃；
- 提取结果和证据卡按`ExecutableRequirementPolicy`指纹与PDF哈希持久化缓存；
- LLM只能提出证据类型、规范化结论和给定页中的短原文；
- 代码固定DOI、主题、PDF哈希和证据ID，并验证引句确实存在；
- 将研究对象、数据、方法、结果、局限、背景和未来工作写入可追溯矩阵；
- 区分`A_core/full_text_verified`与仅元数据验证的B/C层文献；
- 未解决PDF、OCR、证据或章节覆盖问题会阻止V0.4交接；
- 输出确认需求、最终写作大纲和确认版证据库组成的V0.4交接合同；

## V0.4.0 证据约束写作能力

- 每个章节只接收确认大纲分配的文献和证据卡；
- A级全文、B级辅助和C级背景文献具有不同的可用权限；
- DeepSeek输出结构化段落和来源声明，不直接生成引用；
- 程序确定性生成Pandoc引用键，并保留DOI、证据卡和PDF页码轨迹；
- 越权来源、未知证据、模型自造DOI或引用标记会阻止章节确认；
- 用户逐章确认，全部正文章节确认后才允许汇总Markdown；
- 课程禁止AI生成正文时，服务层禁止调用模型并导出人工写作证据包。

## MVP 最终交付与要求贯通

- V0.1确认结果会编译为版本化`ExecutableRequirementPolicy`，并随V0.2蓝图、V0.3证据库和V0.4交接包向下游传播；
- 检索与选择实际执行文献数量、硬年份、外文比例、来源类型偏好和禁用来源规则；
- 正文完成后才生成标题、摘要、关键词和结论，程序负责引用样式和最终参考文献；
- 最终发布审计再次检查字数口径、文献数、外文数、硬年份、来源禁限、章节、AI声明和引用簇上限；
- 只有审计无阻塞且用户最终确认后，才解锁完整Markdown、DOCX和审计JSON；
- 全链路金标准测试覆盖需求确认、策略编译、文献检索/真实性验证、真实PDF文本提取、证据卡、逐章写作、最终审计和DOCX打开验证；
- “逐句话是否真正被引文语义支持”的蕴含验证明确作为MVP后续优化项，不伪装成当前已完成能力。

实现边界、策略字段映射与验收说明见
[`docs/mvp_executable_policy_and_delivery.md`](docs/mvp_executable_policy_and_delivery.md)。

详细设计见[`docs/v0.4_grounded_writing_design.md`](docs/v0.4_grounded_writing_design.md)。
- 两篇真实Elsevier PDF烟雾测试分别完成12/12页和13/13页原生文本提取。

设计、边界与验收说明见
[`docs/v0.3_evidence_matrix_design.md`](docs/v0.3_evidence_matrix_design.md)。

## 路线图

- V0.1：课程要求获取与确认（已完成核心流程）
- V0.2：文献检索、身份验证与多主题均衡选择（已完成核心流程）
- V0.3：PDF全文、双层文献库、文献矩阵与证据卡（已补全提取审计与持久缓存）
- V0.4：按确认大纲与证据库逐章节写作（正文闭环已实现）
- MVP：要求策略贯通、最终Markdown/DOCX、发布审计与全链路金标准测试（已实现）
- 后续优化：逐句引用语义蕴含验证、更多引用样式和更大规模真实案例评测
