# V0.4与普通LLM+PDF受控对照

- 运行时间：2026-07-31T06:48:24.818374+00:00
- 模型：deepseek-chat
- 输入：同三篇气溶胶论文PDF全文提取文本
- 任务：综合比较卫星气溶胶光学厚度反演、时空连续估计、降尺度方法和辐射效应研究，并指出证据边界与当前局限。

## 确定性审计

| 指标 | 普通LLM+PDF | V0.4 |
|---|---|---|
| 引用DOI均来自给定论文 | True | True |
| 可回溯到证据ID和PDF页码 | 0个机器可验证绑定 | 7/7个绑定含页码 |
| 代码阻止越权或虚构引用 | 否 | 是 |
| 结构化可恢复中间产物 | 无 | 证据卡、章节证据包、草稿审计包 |
| 调用方式 | 1次长上下文生成 | 分批证据提取 + 1次章节生成 |

## 盲评模型评分

| 维度 | 草稿A | 草稿B | 说明 |
|---|---:|---:|---|
| 连贯性 | 4 | 5 | Draft A is well-structured with clear logical flow, but Draft B is more coherent in integrating evidence and maintaining consistent narrative. |
| 来源忠实度 | 3 | 5 | Draft A includes specific numeric results not fully supported by the provided evidence, while Draft B sticks closely to the verified claims and quotes. |
| 引用真实性 | 5 | 5 | Both drafts provide plausible citations, but Draft B's citations are explicitly linked to evidence IDs and page numbers, enhancing authenticity. |
| 可追溯性 | 2 | 5 | Draft A lacks explicit page numbers and evidence IDs, making it harder to trace claims, whereas Draft B includes page numbers and evidence IDs. |
| 主题综合 | 4 | 5 | Both synthesize well, but Draft B more clearly integrates the three studies into a cohesive narrative on advancing aerosol remote sensing. |
| 低无依据风险 | 3 | 5 | Draft A contains several unsupported numeric details (e.g., specific R2 values for radiative fitting, AOD trends), while Draft B only uses numbers from verified evidence. |

- 盲评胜者：draft_b
- 置信度：0.90
- 总结：Draft B is superior in source fidelity, traceability, and low unsupported claim risk, as it closely adheres to the provided evidence with proper citations and page numbers. Draft A makes some claims not verifiable from the given evidence. Both are coherent, but B is more traceable and faithful.

## 解释边界

- 本实验不是模型排行榜，只比较同一模型在两种工作流中的行为。
- 普通方案得到更少API调用和更低延迟；V0.4增加了证据提取成本。
- 盲评属于LLM评估，不是绝对真值；确定性引用审计才是硬指标。
- V0.4的优势主要是可审计、可恢复和失效可见，而非保证文风一定更好。

## 文件

- `baseline_response.md`：普通LLM+PDF输出
- `evidence_cards.json`：经原文短引句校验的证据卡
- `v04_response.md`：V0.4输出
- `v04_draft_audit.json`：引用与章节审计
- `blind_judge.json`：匿名LLM评分
