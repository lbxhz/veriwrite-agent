# 大气遥感文献真实性对照实验

## 研究问题

在相同模型、相同主题、相同时间范围和相同目标数量下，对比：

1. Agent工作流：DeepSeek只生成检索计划，Crossref提供候选，Crossref RIS作为规范记录，
   DOI.org验证DOI能否解析。
2. 直接LLM：DeepSeek不使用检索工具，直接列出20篇论文及其DOI，再使用完全相同的
   Crossref RIS和DOI.org链路核验其声明。

## 固定条件

- 模型：项目`.env`配置的同一个DeepSeek模型。
- 主题：大气遥感，包括卫星大气探测、气溶胶、云、温室气体和空气质量遥感。
- 时间范围：2022—2026年（含首尾年份）。
- 文献类型：期刊论文。
- 目标数量：20篇。
- Agent分级数据：中国地质大学（武汉）2023版期刊分类目录，T1—T6均可。

## 真实性判据

Agent输出只有同时满足以下条件才进入最终20篇：

- Crossref能够返回单条RIS；
- RIS包含DOI、题名、作者、年份和期刊；
- RIS的`DO`字段等于请求的候选DOI；
- RIS DOI能够通过DOI.org完成解析；
- RIS年份在2022—2026年内。

出版社拒绝自动页面访问，但DOI已经完成重定向且RIS验证成功时，记录访问警告，不判为
虚假论文。

## DeepSeek幻觉判据

- `invalid_doi_syntax`：模型给出的DOI不符合DOI格式。
- `doi_not_verified`：Crossref无RIS、RIS不完整、DOI不匹配或DOI无法解析。
- `identity_hallucination`：DOI真实，但模型声称的题名与该DOI的RIS题名不同。
- `metadata_error`：题名相同，但作者、年份或期刊至少一项与RIS不同。
- `fully_correct`：DOI有效，题名、作者、年份、期刊均与RIS规范化一致。

作者比较要求模型给出的作者集合与RIS一致；题名和期刊允许大小写、标点、空格、重音符号
以及`&/and`的规范化差异。

## 相关性边界

本实验主要测量文献真实性和模型元数据幻觉。只额外记录基于题名关键词的主题信号，
不把题名信号当成完整相关性判断。摘要与全文相关性属于后续版本。

## 可复现产物

- `system_search_plan.json`
- `system_results.json`
- `system_verified.ris`
- `deepseek_prompt.txt`
- `deepseek_raw_response.txt`
- `deepseek_results.json`
- `experiment_summary.json`
- `comparison_report.md`
