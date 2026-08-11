# VeriWrite Agent 运行时数据合同

## 目的

现有 V0.1-V0.5 模型继续负责论文产物本身，例如需求、文献、证据库、
写作计划和最终论文。Agent 运行时合同只负责回答：

1. 当前系统处于什么状态；
2. Planner/Controller 申请执行什么行动；
3. Executor 实际执行出了什么结果；
4. Critic 发现了什么问题；
5. Controller 为什么继续、返修、回退或停止；
6. 失败后从哪个检查点恢复。

运行时状态只保存产物引用，不复制整篇论文或完整事件历史，避免
AgentState 无限制膨胀并污染模型上下文。

## 六个核心合同

| 合同 | 写入者 | 主要消费者 | 权限边界 |
|---|---|---|---|
| AgentState | Controller | 所有运行模块 | 当前事实的唯一来源，不保存模型私有思考 |
| AgentActionRequest | Planner 提议、Controller 批准 | Executor | 只能使用有限、带类型的 action payload |
| ToolObservation | Executor | Critic、Controller | 只描述实际结果，失败不能伪装成成功 |
| CriticReport | 确定性审计、独立 LLM、评分 MCP | Controller | 只读评价，无权修改正文或状态 |
| ControllerDecision | Controller | Runtime | 唯一可以推进、返修、回退和结束运行的合同 |
| AgentCheckpoint | Runtime | 恢复器 | 保存小型状态快照、父检查点和防篡改指纹 |

实现位置：src/veriwrite_agent/models/agent_runtime.py。

## 与现有阶段产物的映射

| 阶段 | 现有产物合同 | ArtifactReference.kind |
|---|---|---|
| V0.1 | RequirementSpec | requirement_spec |
| V0.1 | ExecutableRequirementPolicy | requirement_policy |
| V0.2 | ConfirmedLiteratureSearchBlueprint | literature_blueprint |
| V0.2 | LiteratureDiscoveryResult / selection result | literature_result |
| V0.2 | LiteratureVerificationBatch | literature_verification |
| V0.3 | EvidenceLibrary | evidence_library |
| V0.3→V0.4 | V04WritingHandoff | writing_handoff |
| V0.4 | GroundedWritingPlan | writing_plan |
| V0.4 | V04WritingProject | writing_project |
| V0.4 | BodyDraftPackage | body_draft |
| V0.5 | ManuscriptQualityReview | manuscript_review |
| V0.5 | FinalPaperPackage | final_package |
| 独立评分 | PaperQualityScorecard | quality_scorecard |

storage_key 指向本地项目存储中的真实产物，fingerprint 绑定具体版本。
运行时不能仅凭 UI 标签声称某项产物存在，必须能够解析并验证对应产物合同。

## 当前允许的行动

首版行动空间有意保持有限：

- refine_literature_search
- acquire_full_text
- rebuild_evidence
- revise_writing_plan
- write_or_revise_sections
- run_critic
- assemble_final_delivery
- request_user_input

每类行动都有独立 payload，而不是开放的 dict[str, Any]。例如定点返修必须
明确章节和段落；请求用户只能用于需求冲突、人工下载 PDF、改变策略或最终确认。

## 幂等性

AgentActionRequest.idempotency_key 由确定性代码根据输入产物 ID 和已验证的行动
payload 计算 SHA-256。模型不能自行填写任意幂等键。刷新或重试时，Executor
可以先查找相同幂等键的成功 ToolObservation，直接复用结果，避免重复检索、
重复下载或重复模型调用。

运行时存储为每个 run 建立以下唯一索引：

    (run_id, idempotency_key) -> observation_id

## 回退示例

写作 Critic 发现某段比较两种算法，但只有一篇完整 PDF：

1. CriticReport.outcome = rollback；
2. finding 的 responsibility_stage = evidence；
3. Controller 生成 decision_type = rollback；
4. current_stage = writing，target_stage = evidence；
5. next_action.payload.kind = rebuild_evidence；
6. Runtime 在行动前保存 AgentCheckpoint；
7. Executor 只补充受影响章节证据；
8. 成功后重新规划并只打开受影响章节。

合同会拒绝“从 writing 回退到 delivery”这类方向错误，也会拒绝没有下一行动的
活跃决策。

## 人机协同

request_user_input 是唯一必须设置 requires_user_approval=true 的行动。
当前允许的用户请求：

- V0.1 需求冲突；
- V0.3 无法自动获取的 PDF；
- 改变已确认策略；
- 最终论文确认。

普通关键词调整、候选扩充、局部返修和独立审稿不应增加用户按钮。

## 终止与预算

finish 只能发生在 delivery，且必须引用至少一份 Critic 报告。
AgentBudget 同时限制模型调用次数、总 token（配置时）和跨阶段恢复轮数。

预算耗尽时应生成 stop 决策和明确诊断，不能继续盲目重试。

## 已实现的持久化边界

`services/agent_artifacts.py` 负责把现有 Pydantic 阶段产物转换成紧凑的
ArtifactReference，并在同类新版本登记时把旧版本标记为 superseded。相同指纹的
产物重复登记是无操作，不会增加事件序号。

`services/agent_runtime_store.py` 使用独立文件保存 action、observation、critic、
decision 和 checkpoint：

    <run-root>/
      actions/<action_id>.json
      observations/<observation_id>.json
      critics/<report_id>.json
      decisions/<decision_id>.json
      checkpoints/<sequence>_<checkpoint_id>.json
      idempotency_index.json
      latest_checkpoint.json

所有写入均先写临时文件再原子替换。恢复器不依赖 latest_checkpoint.json 才能工作，
即使该指针在进程中断时损坏，也会扫描并验证父子检查点链，恢复最后一个连续有效
状态。重复 ID 只能对应完全相同的事件；同一幂等键不能绑定两个成功结果。

## 尚未接入的部分

本次仍没有替换现有 Streamlit 流程，也没有调用模型。下一步按以下顺序接入：

1. 将现有 V0.4 证据缺口回退转换为 Action→Observation→Critic→Decision；
2. 用 Controller 驱动全文 Critic 和定点返修；
3. 接入另一 Agent 提供的外部评分 MCP，但只作为 Critic 输入；
4. 用真实项目验证恢复和只重做失败节点；
5. 稳定后再减少旧 session_state 控制字段。

旧流程在新 Controller 通过真实案例验证前继续保留，避免一次性重写造成不可恢复
回归。
