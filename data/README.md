# 中国地质大学（武汉）2023版期刊分类索引

`cug_wuhan_journal_classification_2023.csv`由用户提供的官方附件
《中国地质大学（武汉）各学科期刊分类目录（2023版）》规范化生成。

- 评价体系：`CUG_WUHAN_TIER`
- 版本：2023
- 学科工作簿：38
- 原始有效记录：15,875
- 等级：T1–T6
- 原始ZIP SHA-256：
  `DE73E3A8A818C98C52FA681857A01E67766657D9BB45C74A100A7AD883D9F964`

该数据不是中科院期刊分区，不得在界面或报告中标记为中科院一区、二区等。
CSV保留源工作簿和源行号，便于解释每个匹配结果。

源目录中有18组同学科规范化重复记录，其中5组给出了不同等级。
适配器会将不同等级标记为`ambiguous`，不会自动选择较高或较低等级。

# 挪威国家学术出版渠道目录2025快照

`norwegian_register_journals_2025.csv`来自挪威高等教育与技能署
（HK-dir）维护的 Norwegian Register for Scientific Journals, Series and
Publishers 官方实时清单。

- 评价体系：`NORWEGIAN_REGISTER`
- 固定等级年份：2025
- 下载日期：2026-07-30
- 官方下载地址：
  `https://kanalregister.hkdir.no/publiseringskanaler/csvliste/tidsskrift?request_locale=en`
- 保留记录：36,265（仅保留2025年明确为0、1、2级的记录）
- 等级：Level 2（较高层级）、Level 1（获认可基础层级）、
  Level 0（该年度未获认可）
- CSV SHA-256：
  `A057402579FDC0ABB8111C824FC0103D5E0D4520FBA8106989303A04BFB5A637`
- 官方站点标示内容许可：CC BY 4.0 / NLOD

适配器优先用Print ISSN或Online ISSN匹配；仅在ISSN缺失或未命中时，
才使用与地大目录相同的保守题名规范化匹配。挪威等级是独立补充证据，
不会被伪装成中科院分区、JCR分区或地大T等级，也不在不同体系之间做
伪精确换算。
