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

## Step 3 — agent loop + /chat/agent（已完成）

`agents/agent_loop.py`：function-calling 循环。规划模型 `AGENT_PLANNER_MODEL`
（默认 `deepseek-chat`）。while 返回 `tool_calls` 就执行并把结果作为 tool 消息回灌，
返回纯文本即最终答案。唯一护栏 = `AGENT_MAX_STEPS`（默认 6）硬上限，超限后强制
无工具收尾。新增 `/chat/agent` 端点，旧端点不动。

### 验证
- 68 单测全绿（63 + 5 新增 `test_agent_loop.py`）。
- 真实集成冒烟（throwaway 脚本，已删）：query「Q3 预算评审会议是谁发的？」
  → agent 自主走了 2 步工具链 `search_emails → get_email → 作答`，3 次 LLM 调用
  自然收尾，答案正确（识别出发件人）。验证了 DeepSeek 工具调用回灌协议端到端跑通。

## Step 4 — 多步任务 + 工具补齐（已完成）

- `draft_reply` 工具支持 `email_id` 参数：可对一封确定的邮件精确起草回复（多步
  任务把 `search_emails` 结果里的 `email_id` 直接传进来，工具间传结构化数据）。
  从 WriterAgent 提取 `draft_reply_for_email(email, instruction)` 模块级函数，
  新旧路径共用。
- agent system prompt 加多步指引：逐封任务先 search、再用 email_id 逐个处理。
- 决策：不改 `coordinator.py`——agent loop 本身就是升级版 planner，旧的意图路由
  作为 `/chat` 的降级路径保留（符合项目"旧实现留作降级"的约定）。

### 验证
- 71 单测全绿（68 + 3 新增）。
- 真实多步冒烟（throwaway，已删）：query「找一封关于预算评审的邮件，帮我起草一封
  回复，确认我会参加会议」→ agent 自主走 3 步链
  `search_emails → get_email → draft_reply(email_id=...)`，email_id 在工具间传递，
  6 次 LLM 调用自然收尾，产出完整、有依据的回信草稿。

## Step 5 — agent 护栏加固（已完成）

agent loop 的失败模式护栏：
- **死循环检测**：同一 (工具, 参数) 调用超过 `AGENT_MAX_REPEAT`（默认 2）次即拦截，
  回灌提示让模型基于已有结果作答，不再执行该工具。
- **参数校验**：`call_tool` 用 `inspect.signature` 丢弃模型幻觉出来的多余 kwarg、
  对缺失必填参数返回错误（而非 TypeError）。
- **工具报错回灌**：工具内部抛异常被 `call_tool` 捕获，转成 `{"error": ...}` 回灌，
  不再让整个 loop 崩。
- **输出截断**：单次工具结果超过 `AGENT_TOOL_OUTPUT_LIMIT`（默认 4000 字符）截断，
  防多步任务上下文膨胀。

### 验证
- 76 单测全绿（71 + 5 新增失败模式测试）。
- 护栏靠确定性单测验证（mock 一个永远循环的 LLM、坏参数、抛异常的工具）——这类
  失败模式无法靠真实 LLM 稳定复现，单测才是正确的验证手段。
