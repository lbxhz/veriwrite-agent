# 大气遥感文献真实性对照实验报告

## 结论

在本次固定条件实验中，Agent工作流输出20篇文献，20篇均取得完整Crossref RIS且DOI能够
解析；直接DeepSeek输出20条外观完整的文献声明，但没有一条做到DOI、题名、作者、年份和
期刊全部正确。

这一结果只代表`deepseek-v4-flash`在本次提示、主题和运行中的表现，不能直接外推为所有
模型、所有提示或多次运行的固定幻觉率。

## 实验条件

- 运行日期：2026-07-29
- 时间范围：2022—2026年
- 主题：大气遥感，包括气溶胶、云、温室气体和空气质量
- 两组目标：各20篇期刊论文
- 两组模型：相同的`deepseek-v4-flash`
- 权威基准：Crossref RIS
- DOI检查：DOI.org解析
- Agent期刊目录：中国地质大学（武汉）2023版，大气科学，T1—T6

Agent组让DeepSeek只生成检索计划。直接组明确禁止使用外部工具，让DeepSeek直接列出
题名、全部作者、年份、期刊和DOI。两组使用完全相同的RIS和DOI验证标准。

## 定量结果

| 指标 | Agent工作流 | 直接DeepSeek |
|---|---:|---:|
| 最终输出/声明数量 | 20 | 20 |
| DOI格式正确且不重复 | 20 | 20 |
| Crossref RIS与DOI验证成功 | 20 | 7 |
| DOI真实但题名指向另一篇论文 | 0 | 7 |
| Crossref无对应RIS | 0 | 13 |
| 题名、作者、年份、期刊全部正确 | 20（以RIS为输出） | 0 |
| 基于题名关键词的主题信号 | 17 | 1（按真实DOI对应论文） |

直接DeepSeek的20个DOI全部具有很像真实DOI的格式，而且没有重复，但：

- 13个DOI无法从Crossref取得RIS，占65%；
- 7个DOI确实存在，但全部被配给了另一篇论文，占35%；
- 完整正确率为0%，本次“完整文献声明错误率”为100%。

7个真实DOI中，期刊字段7个正确、年份6个正确，但题名和作者均为0个正确。这表明模型能够
模仿期刊DOI编号、年份和卷期结构，却没有掌握这些DOI实际对应的论文身份。

## 真实DOI被错误套用的示例

| DOI | DeepSeek声称的题名 | RIS实际题名 |
|---|---|---|
| 10.5194/amt-16-2381-2023 | The OMNIA algorithm for retrieving aerosol optical depth from the Ozone Monitoring Instrument | Calibrating radar wind profiler reflectivity factor using surface disdrometer observations |
| 10.3390/rs14010123 | Retrieval of cloud droplet effective radius from the Advanced Himawari Imager | GMT-WGAN: An Adversarial Sample Expansion Method for Ground Moving Targets Classification |
| 10.1029/2023GL103456 | A new method for retrieving aerosol absorption from OMI and TROPOMI | Poynting Fluxes, Field-Aligned Current Densities, and the Efficiency of the Io-Jupiter Electrodynamic Interaction |

第三个例子尤其说明“DOI可以打开”不是充分条件：DOI真实、年份和期刊也像是正确的，但实际
论文主题完全不同。直接使用模型给出的题名和DOI会形成可解析但错误的引用。

## Agent组20篇规范记录

| 年份 | 地大等级 | 题名 | DOI |
|---:|:---:|---|---|
| 2022 | T2 | Diffuse light around cities: New perspectives in satellite remote sensing of nighttime aerosols | 10.1016/j.atmosres.2021.105969 |
| 2023 | T2 | Mineral dust aerosols over the Himalayas from polarization-resolved satellite lidar observations | 10.1016/j.atmosenv.2023.119584 |
| 2025 | T1 | Validation of Aerosol Optical Depth Retrieved From CALIPSO Lidar Ocean Surface Backscatter | 10.1029/2024JD042416 |
| 2026 | T2 | Retrieval of aerosol composition from spectral aerosol optical depth and optical properties using a machine learning approach | 10.5194/amt-19-421-2026 |
| 2026 | T2 | Aerosol optical depth in East Asia from VIIRS, MODIS, and MISR: Evaluation, variability, and meteorological associations | 10.1016/j.atmosenv.2025.121760 |
| 2024 | T3 | Variability of Aerosol Optical Depth and Altitude for Key Aerosol Types over Southern West Africa via CALIPSO/CALIOP Observations | 10.3390/atmos15040396 |
| 2022 | T3 | Deep Neural Networks for Aerosol Optical Depth Retrieval | 10.3390/atmos13010101 |
| 2022 | T3 | Spatiotemporal Analysis of MODIS Aerosol Optical Depth Data in the Philippines from 2010 to 2020 | 10.3390/atmos13060939 |
| 2024 | T3 | The Aerosol Optical Depth Retrieval from Wide-Swath Imaging of DaQi-1 over Beijing | 10.3390/atmos15121476 |
| 2026 | T2 | Retrieval of aerosol optical depth from INSAT-3DR for accurate geostationary monitoring of regional and temporal aerosol dynamics | 10.1016/j.atmosenv.2025.121730 |
| 2025 | T1 | TEMPO Aerosol Optical Depth and Aerosol Layer Height Retrieval Algorithm | 10.1029/2025JD044082 |
| 2022 | T2 | Evaluation of the MODIS Collection 6.1 3 km aerosol optical depth product over China | 10.1016/j.atmosenv.2022.118970 |
| 2023 | T2 | Evaluation and comparison of MODIS aerosol optical depth retrieval algorithms over Brazil | 10.1016/j.atmosenv.2023.120130 |
| 2023 | T3 | Aerosol Optical Depth Retrieval for Sentinel-2 Based on Convolutional Neural Network Method | 10.3390/atmos14091400 |
| 2023 | T3 | Retrieval of Aerosol Optical Depth and FMF over East Asia from Directional Intensity and Polarization Measurements of PARASOL | 10.3390/atmos15010006 |
| 2025 | T2 | Retrieval of aerosol properties from aerosol optical depth measurements with high temporal resolution and spectral range | 10.5194/amt-18-7651-2025 |
| 2026 | T2 | Vertical profiles of aerosol chemical species concentration retrieved through synergy of spaceborne lidar and polarimeter observations | 10.5194/amt-19-4669-2026 |
| 2024 | T1 | Assessments of Arctic Cloud Vertical Structure From AIRS Using Radar-Lidar Observations | 10.1029/2024JD040871 |
| 2024 | T2 | A case study of aircraft observations of ice cloud microphysical properties over the North China Plain: Vertical distribution, vertical airflow and aerosol-cloud effects | 10.1016/j.atmosres.2024.107488 |
| 2025 | T2 | A new technique to retrieve aerosol vertical profiles using micropulse lidar and ground-based aerosol measurements | 10.5194/amt-18-5841-2025 |

等级分布为T1三篇、T2十一篇、T3六篇；年份分布为2022—2026年每年四篇。三篇论文的
出版社页面限制自动访问，但DOI已完成重定向且RIS验证成功，因此保留访问警告而未误判为
虚假论文。

## 系统暴露的问题

Agent组保证了文献身份真实性，但主题覆盖不均衡。20篇主要集中于气溶胶光学厚度和云，
温室气体及空气质量分支不足。

本轮共扫描246篇候选，地大目录预筛得到50篇。当前搜索提供器按检索式顺序返回结果，
发现服务达到数量后立即停止，因此前两条气溶胶/云查询占用了大部分名额，后面的二氧化碳、
甲烷和空气质量查询没有获得公平配额。

下一步应把顺序消费改为检索式轮询或每个主题分支设置最低配额。这属于检索覆盖问题，
不是文献真实性问题。

## 能够支持的产品结论

本实验支持以下表述：

> 当LLM只负责生成检索策略，而文献身份由Crossref RIS和DOI.org确定时，系统能够稳定输出
> 可追溯的真实文献记录；直接让无检索工具的LLM生成完整参考文献，可能产生格式逼真、
> DOI可解析但论文身份错误的结果。

本实验不能证明：

- Agent组20篇都适合最终综述结构；
- 论文正文结论真实或能支持特定主张；
- DeepSeek在所有重复实验中的完整错误率恒为100%；
- 其他模型或联网检索型模型具有相同表现。

相关性、多主题覆盖、摘要筛选和全文证据仍需后续版本完成。
