# V0.1 + V0.2 本地控制台

## 启动

```powershell
cd C:\Users\17811\Documents\Codex\2026-07-12\new-chat\outputs\veriwrite-agent
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

浏览器打开 Streamlit 提供的本地地址后，可以从真实要求文件一直运行到最终验证文献。

## 完整操作顺序

1. 上传 TXT、Markdown、DOCX、旧 DOC、PDF 或多张连续截图；
2. 选择规则模式或“规则 + DeepSeek”双路模式；
3. 检查 OCR/原生文本、双路字段、冲突、阻塞项和原文证据；
4. 修改主题、处理冲突并确认 V0.1 最终 `RequirementSpec`；
5. 让 DeepSeek 根据确认需求生成临时检索蓝图；
6. 检查主题、研究问题、英文检索词、每主题配额、年份和候选上限；
7. 必要时直接编辑蓝图 JSON，再进行一次集中用户确认；
8. 启动 V0.2，依次执行分主题 Crossref 检索、RIS/DOI 验证、相关性评分和均衡选择；
9. 下载最终文献 JSON、RIS 和完整真实性验证证据。

临时检索蓝图未确认时，代码不允许开始 Crossref 检索。

## 真实案例测试时重点观察

- OCR 提取字符是否完整，校对文本是否正确传入双路解析；
- V0.1 最终主题、参考文献数量、年份和限制是否符合原文；
- 临时蓝图是否覆盖所有实际需要讨论的主题；
- 每个主题的配额是否合理，检索词是否过宽或过窄；
- `prefiltered_count`、`verified_count` 和真实性排除原因；
- 最终每个主题是否达到配额；
- 未分级论文是否只是地大目录未匹配，而不是 DOI/RIS 失败；
- 是否出现大量相关性 1.0 的评分饱和。

如果某主题不足，系统不会让其他主题静默占用名额。界面会建议由用户选择：增加关键词、
减少限制，或把 `max_candidates` 从 300 上调至 500。

## 中断与恢复

每个确认蓝图会计算独立指纹，缓存保存在：

```text
runtime/literature_console/<blueprint_run_id>/
```

缓存包含：

- `confirmed_blueprint.json`
- `discovery_cache.json`
- `verification_cache.json`
- `relevance_cache.json`
- `final_result.json`
- `selected.ris`

网络失败、关闭浏览器或模型调用中断后，使用同一确认蓝图再次点击“开始或继续”即可从未完成
阶段恢复。修改蓝图会生成新的指纹和独立目录，不会误用旧案例结果。

## 当前边界

- V0.2 验证的是文献身份和题名/摘要相关性，不是全文论点证据；
- 地大 2023 版期刊目录是本地质量偏好，不是论文真实性判据；
- DOI.org、Crossref 或出版社响应较慢时，真实运行可能持续数分钟；
- DeepSeek 相关性评分仍可能饱和，需要后续人工金标准校准；
- 文献矩阵、全文证据卡和最终写作大纲属于后续版本。
