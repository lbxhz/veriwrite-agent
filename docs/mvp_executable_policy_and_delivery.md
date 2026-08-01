# MVP 要求贯通、PDF 可靠性与最终交付设计

## 1. 为什么需要 ExecutableRequirementPolicy

`RequirementSpec` 是用户确认的业务事实合同，回答“课程要求是什么”；
`ExecutableRequirementPolicy` 是运行时策略合同，回答“每个模块必须怎样执行和验收”。
策略由确定性代码从 `ConfirmedRequirementSpec` 编译，带有原始要求的 SHA-256 指纹，
不允许在 V0.2 临时检索蓝图中被 LLM 或用户编辑器静默修改。

| V0.1 要求 | 下游执行位置 | 失败行为 |
|---|---|---|
| 主题与主题元素 | V0.2 蓝图、V0.3 检索选页、V0.4 大纲 | 无主题则停止 |
| 字数与计数口径 | V0.4 章节预算、最终发布审计 | 低于/超过硬边界则阻塞 |
| 文献数、外文比例、年份 | V0.2 选择、最终引用文献审计 | 不足则不进入下一阶段 |
| 来源类型与禁用来源 | V0.2 元数据过滤、最终二次审计 | 记录规则编号并排除/阻塞 |
| 章节与交付物 | V0.4 大纲、最终论文组装 | 缺必需章节则阻塞 |
| AI 使用政策 | 调用 LLM 前、最终 AI 声明审计 | 禁止生成时不调用模型 |
| 字体、字号、纸张、行距 | DOCX 导出器 | 作为 Word 样式写入 |
| 提交与流程条件 | UI 用户门禁、最终策略清单 | 保留为用户确认事项 |

## 2. 贯通后的 MVP 数据流

```text
ConfirmedRequirementSpec
  -> RequirementPolicyCompiler
  -> ExecutableRequirementPolicy (immutable fingerprint)
  -> V0.2 confirmed blueprint / search plans / balanced selection
  -> V0.3 EvidenceLibrary (same policy fingerprint)
  -> V0.4 confirmed outline / section packets / body
  -> FinalPaperPackage
  -> final audit -> user confirmation -> Markdown + DOCX + audit JSON
```

如果证据库、写作大纲与 V0.1 的策略指纹不同，交接模型拒绝建立，避免旧缓存或其他项目的
中间结果被错误复用。

## 3. V0.3 真实运行可靠性

V0.3 先对 PDF 全部页面执行原生文本提取或本地 OCR，并保存 `DocumentExtractionResult`。
完整页面进入 `EvidenceLibrary.pages`；代码再使用主题、章节目的、研究问题和关键词进行页面
相关性检索，仅把选中的页面发送给 LLM。`EvidencePageSelection` 保存选页、得分和总页数，
因此“PDF 是否完整提取”和“哪些页面送给 LLM”是两个可分别审计的问题。

提取结果与证据卡使用以下键持久化：

```text
ExecutableRequirementPolicy fingerprint / PDF SHA-256 / stage artifact
```

同一要求和同一 PDF 在中断后可恢复；PDF 变化或要求变化会自动使用新缓存空间。

## 4. 最终论文交付门禁

只有全部正文章节确认后，LLM 才能基于已确认正文提出标题、摘要、关键词和结论。
LLM 无权创建 DOI、引用键或参考文献。程序按正文真实使用的 DOI 生成文内引用和参考文献，
并检查字数、文献数、外文比例、年份、禁用来源、引用簇、必需章节、AI 声明和未解决要求。

审计无阻塞后仍需用户最终确认，随后才能导出：

- 完整 Markdown；
- 按 `narrative_proposal` 基础预设和 `academic_course_paper` 命名覆盖生成的 DOCX；
- 包含策略指纹、统计数据、问题和确认记录的审计 JSON。

## 5. MVP 明确不承诺的能力

V0.4 当前证明引用来自哪个 DOI、证据卡和 PDF 页码，但不自动判断整段中的每一句话是否被
对应引文在语义上充分支持。`FinalPaperAudit.deferred_checks` 固定记录 `claim_entailment`，
该能力作为后续优化，不属于本次 MVP 完成标准。

## 6. 金标准测试

`tests/test_mvp_end_to_end_golden.py` 使用固定真实业务语义跑通：

1. 规则与 Fake LLM 双路需求解析、用户确认；
2. 策略编译与指纹传播；
3. 候选检索、DOI/RIS 真实性验证、相关性与均衡选择；
4. 两个真实可解析 PDF 文件的逐页文本提取；
5. 证据选页、证据卡、证据库、大纲和逐章确认；
6. 最终论文审计、确认、Markdown 与 DOCX 打开验证。

Fake LLM 只用于让模型输出可重复；检索、验证、PDF、策略、审计和交付均运行真实代码路径。
