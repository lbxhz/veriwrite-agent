# 项目目录指南：名词、动词和证据

不要用“输入端/输出端”理解目录。一个数据模型既可能来自输入，也可能成为另一个模块的输出。更稳定的划分方法是：

```text
models   = 名词：系统需要认识和保存什么
services = 动词：系统需要执行什么业务动作
tests    = 证据：动作是否符合预先定义的预期
```

## 1. models：系统的名词表和数据合同

`models` 描述业务概念，不负责检索或计算。例如：

- `RequirementSpec`：课程要求；
- `ReferenceRecord`：一篇参考文献；
- `EvidenceCard`：一条全文证据；
- `ConflictRecord`：两个解析器之间的冲突。

如果课程出现“至少一半为核心期刊文献”，而现有模型没有核心期刊比例字段，那么需要扩展 `models`。原因不是文献够不够，而是系统目前没有位置保存这个新概念。

## 2. services：系统的动词和用例

`services` 接收模型、执行一个业务动作、返回模型。例如：

- `RuleBasedRequirementParser.parse(text)`：解析要求；
- `RequirementReconciler.reconcile(candidates)`：合并候选结果；
- `ReferenceSearchService.search(query)`：检索文献；
- `ReferenceVerifier.verify(record)`：验证文献身份；
- `EvidenceExtractor.extract(pdf)`：从全文提取证据。

如果模型已经能保存外文最低比例，但解析器不认识“英文来源不得低于33%”，应该修改解析服务，而不是模型。

## 3. tests：可执行的验收标准

测试不是业务模块，也不只是“看看代码会不会运行”。测试声明我们预期系统具有什么行为：

```python
def test_parses_percentage_expression():
    result = parser.parse("英文来源不得低于33%")
    assert result.references.minimum_foreign_ratio == 0.33
```

如果新增一种表达方式，应先增加测试案例，再修改解析服务让测试通过。这样可以证明新增能力没有破坏已有能力。

## 4. config：运行环境和秘密的入口

`config` 负责读取模型名称、API 地址、超时和 Key。业务服务不应该自己到处读取 `.env`，否则测试困难、配置分散并容易泄露秘密。

`.env.example` 是可公开模板；`.env` 是本地秘密文件。

## 5. llm：外部模型供应商适配层

`llm` 隔离 DeepSeek/OpenAI 等供应商差异。上层服务只依赖统一的 `LLMClient`：

```text
LLMRequirementParser -> LLMClient -> DeepSeekClient -> DeepSeek API
                                -> FakeLLMClient -> 固定测试响应
```

这样测试不消耗额度，未来更换供应商也不需要重写课程解析业务。

## 6. cli：人和项目交互的入口

`cli.py` 解析命令行参数，然后调用 service。它不应该塞入大量业务逻辑。

## 7. RequirementReconciler 应该怎样工作

合并器主要使用确定性代码，而不是让 LLM 自由裁决：

1. 对每个候选结果做 Pydantic 验证；
2. 将安全等价值规范化，例如字符串 `"60"` 转为整数 `60`；
3. 相同值自动合并并保留全部来源；
4. 一方缺失时保留已知值并标记单一来源；
5. 不同值生成 `ConflictRecord`；
6. 关键冲突由用户确认；
7. 用户确认结果和原始候选一起保存，不能抹掉历史。

LLM 可以解释自然语言是否可能等价，但不能静默覆盖数字、比例、日期和格式等关键约束。

## 8. 判断应该修改哪个目录

先问三个问题：

1. 系统是否缺少一个需要保存的新概念？是：修改 `models`。
2. 数据结构足够，但处理方式不支持新表达或新流程？是：修改 `services`。
3. 是否需要证明新行为并防止回归？是：增加 `tests`。

大多数新功能会同时修改服务和测试；只有出现新业务概念时才必须修改模型。

