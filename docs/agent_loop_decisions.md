# Agent Loop 升级 — 决策日志

记录 agent 化改造过程中靠实测/验证得出的关键决策，供后续步骤和文档收尾参考。

## Step 0 — 预检（已完成）

探针脚本：`scripts/probe_function_calling.py`（可重复运行复现）。

### 实测结果

| 模型 | function calling | 单次调用延迟 |
|------|:----------------:|------------:|
| `deepseek-v4-flash`（推理模型） | ✅ 返回 `tool_calls` | ~2.3s |
| `deepseek-chat`（非推理） | ✅ 返回 `tool_calls` | ~1.3s |

运行时 `LLM_TIMEOUT` 实测为 **60s**（`.env` 覆盖了 `settings.py` 的默认 15）。

### 决策

1. **agent loop 用原生 function calling**——两个模型都稳定返回 `tool_calls`，不需要手写 JSON 协议兜底。这清除了改造的头号风险。
2. **planning / 选工具调用用 `deepseek-chat`**：更快（~1s vs ~2s）、无推理开销、function calling 已验证。循环中每一步都省时间，累积收益明显。
   → 新增配置项 `AGENT_PLANNER_MODEL`，默认 `deepseek-chat`。
3. **最终答案生成保留 `deepseek-v4-flash`**：需要一点推理质量的地方继续用推理模型。
4. **`LLM_TIMEOUT` 无需改**：实测单次调用 1~2.3s，60s 余量充足。Bug #1（"15s 超时 vs 推理模型"）经实测**不是运行时 bug**——只是 `settings.py` 默认值 15 偏低 + 注释过时。

### 留给 Step 7 的文档/默认值清理

- `settings.py:52` `LLM_TIMEOUT` 默认值 `15` → 改成 `60`，与 `.env`/architecture 一致。
- `core/generator.py:110` 注释称推理模型 `reasoning_content` 要 30~70s——实测仅 ~105 字符 / 2.3s，注释已过时，需更正。
- 三处 `LLM_TIMEOUT` 文档不一致：README 配置表写 15、architecture §9 写 60、settings 默认 15。统一为 60。
- architecture.md §3 "链路总耗时 ~30~50秒" 与 README 评测 "V2 mean 8.7s" 不一致，需对齐。

## Step 1 — 抽 core/pipeline.py（已完成）

新增 `core/pipeline.py`，统一检索链路 `retrieve()`：rewrite → extract_filters →
hybrid_search → apply_post_filters → rerank。

### 改动
- `agents/retriever_agent.py`、`scripts/run_ragas_eval.py`、`scripts/measure_latency.py`
  三处不再各自实现检索链路，统一调 `core.pipeline.retrieve()`。
- **Bug #2 已修**：run_ragas_eval / measure_latency 之前抽取了 filters 却没应用
  sender/date/label 后过滤，评测链路与产品链路不一致。现在三处走同一个 `retrieve()`。

### 决策：graph_workflow 不并入
`agents/graph_workflow.py` 的检索是**故意更精简的**——只有 rewrite + hybrid +
rerank，没有 filter 抽取和后过滤，因为 Self-RAG 用 `grade` 节点替代过滤。强行并入
`retrieve()` 会改变 `/chat/graph` 行为，超出"纯重构"范围。graph_workflow 留到
agent loop 步骤（Step 3-4 重做 Self-RAG 路径时）一并处理。

### 验证
- 52 个单测全绿（45 回归 + 7 个新增 `test_pipeline.py`）。
- 全部模块导入冒烟通过（含 `api.main` 和两个脚本）。
- `core/retriever.py` `core/reranker.py` `core/generator.py` 未改动。
- 未重跑完整 RAGAS 评测：`run_ragas_eval` 会覆写已提交的 `data/eval_results/*.json`
  （README 公开数字的来源）。若要实测后过滤对指标的影响，需单独跑并用临时输出目录。

## Step 2 — 工具层 agents/tools.py（已完成）

5 个工具，每个 = 函数 + OpenAI function schema + 分发表项：
- `search_emails(query, sender?, date_hint?, labels?, limit?)` — 混合检索 + 后过滤 + rerank
- `get_email(email_id)` — 取整封邮件（按 chunk_index 拼接）
- `summarize_emails(query)` — 复用 `SummarizerAgent`
- `draft_reply(query)` — 复用 `WriterAgent`
- `email_stats()` — 复用 `compute_email_stats()`（已从 AnalyzerAgent 提为模块级函数）

`TOOL_SCHEMAS` / `TOOL_DISPATCH` / `call_tool()` 供 Step 3 的 agent loop 使用。
`call_tool` 暂为最小实现，参数校验 / 错误回灌留到 Step 5。
验证：63 单测全绿（52 + 11 新增 `test_tools.py`）。
