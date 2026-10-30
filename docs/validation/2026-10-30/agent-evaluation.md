# Agent 评测与合并检查（2026-10-30）

> 本文及对应 JSON 是 `4b295af` 保存的历史基线。后续代码已变更，当前修复与验收状态见 [evaluation-repairs.md](evaluation-repairs.md)。不能把这份基线当作当前源码的评测结果。

README 合并冲突已解决。原先失败的三个离线回归组在修复测试依赖后全部通过；当前 Agent 质量评测仍未达到现有门槛。

## 已修复的检查问题

- `f28fdf1`：合并 main 到 feature/agent-loop，保留新项目介绍和简化后的作者行。
- `52b868d`：补装 PDF 测试所需的 pypdf；core 环境缺少可选 LangGraph 时跳过对应用例，安装了 LangGraph 的 storage 环境仍执行该用例。
- 本地相关测试 35 项通过；[PR 的三个离线测试组](https://github.com/Yoimiya2627/email-rag-agent/actions/runs/40992045639)全部通过。模型评测与离线回归是两个不同的检查。

## 数据与运行范围

- 使用仓库中的 5,000 封固定合成邮件和 105 道固定 Agent 题目；正式批次每题执行一次，失败保留在分母中。准备阶段的单题试跑未混入正式结果。
- 调用当前源码的 `scripts/run_agent_eval.py` 和已配置的真实模型 API，没有替换模型响应、修改题目或降低评分门槛。
- 模型配置标识为 `deepseek-v4-flash`，嵌入模型为 `BAAI/bge-m3`，revision 为 `5617a9f61b028005a4858fdac845db406aefb181`。规划模型显式采用本地已有的生成模型配置；原配置未单独指定规划模型。
- 云端别名不代表永久固定的模型权重；旧 Flash 名称的当前路由关系见 [DeepSeek 官方说明](https://api-docs.deepseek.com/quick_start/pricing/)。
- 在独立目录恢复匹配的合成索引快照，程序验证得到 5,000 封邮件、5,001 个片段及完整语料摘要。评测使用 simulated 邮件 provider，独立审批和运行状态。
- 源码、题目、语料、索引摘要及实际运行配置均记录在[逐题 JSON 报告](../../../data/eval_results/agent_eval.json)的 `provenance` 中。

## 结果

| 指标 | 实测 | 现有门槛 |
| --- | --- | --- |
| 任务数量 | 105 | 至少 100 |
| Agent 任务成功 | 18/105，17.14% | 至少 80% |
| 预期工具覆盖 | 70/105，66.67% | 至少 80% |
| 禁止工具违规 | 0/105 | 0 |
| 达到步数上限 | 0/105 | 不高于 5% |

报告结构、逐题约束、执行结果和汇总一致性验证通过；质量检查因任务成功率和预期工具覆盖率不足而失败。预期工具覆盖表示调用了题目要求的工具，不等同于所有工具调用都正确。

## 当前限制

- 43 题运行状态为 `context_budget_exceeded`，15 题为 `needs_review`，42 题为 `success`，2 题为 `partial`，2 题为 `approval_required`，1 题为 `timeout`。
- 57 题评分状态为 `error`，48 题为 `ok`。评分失败计入失败分母；这些题不能被宣称为已经完成有效的模型质量判断。
- 运行失败和评分失败存在重叠，不能相加当作额外题目；17.14% 不能直接当作模型事实准确率，也不代表 RAG 检索指标。
- 后续需分别检查上下文预算耗尽、评分输入长度/输出完整性、工具选择和执行结果。当前检查脚本不应通过降低门槛来掩盖这些问题。

运行现有检查即可复核门槛结果：

```bash
python scripts/check_agent_eval_gate.py --input data/eval_results/agent_eval.json
```

该命令应报告两项质量门槛失败，不能把脚本运行完成解释为评测通过。
