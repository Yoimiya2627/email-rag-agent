# Code Walkthrough — 私人面试速查

> 这份文档不入库（gitignore），面试前 30 分钟翻一遍。
>
> 当前分支重点看 17 个关键文件，其中最新 Agent 升级最容易被问的是 `agents/agent_loop.py`、`agents/tool_registry.py`、`agents/mcp_adapter.py`、`mcp_server.py`、`agents/approvals.py`、`agents/tracing.py`、`agents/tools.py`、`api/main.py`、`core/pipeline.py`，旧 RAG 链路仍要能讲清 `coordinator / retriever / reranker / memory / api`。
>
> 三份文档分工：`resume_interview_question_bank.html` 负责广度题库，`interview_qa_walkthrough_v2_full.html` 负责深度讲稿，本文件负责“看到代码文件能马上讲”。

---

## 0.0 先记住真实代码链路

| 入口/脚本 | 真实代码路径 | 面试口径 |
|---|---|---|
| `/chat` | `api/main.py` → `Coordinator` → `RetrieverAgent` → `core.pipeline.retrieve()` | 标准 RAG 路径，完整复用统一 pipeline |
| `/chat/stream` | `api/main.py` 的 SSE event_generator → worker 线程 → `asyncio.Queue` | 重点不是新检索算法，而是同步 SDK 和 async FastAPI 的流式桥接 |
| `/chat/graph` | `agents/graph_workflow.py` → `hybrid_search` → `rerank` → grade / rewrite / generate | Self-RAG 独立状态机，复用底层组件，但不是完整调用 `retrieve()` |
| `/chat/agent`（默认） | `agents/agent_loop.py` → local backend → `agents/tools.py` → search/get/summarize/draft/send/stats | function-calling 多步工具编排；`send_email` 是高风险工具，只创建待审批单 |
| `/chat/agent`（MCP backend） | `agents/agent_loop.py` → `agents/mcp_adapter.py` → MCP `tools/list` / `tools/call` → `mcp_server.py` | MCP-ready 工具后端；默认不启用，设置 `AGENT_TOOL_BACKEND=mcp` 后动态发现并调用 MCP 工具 |
| MCP server | `mcp_server.py` → `agents/tool_registry.py` → `agents/tools.py` | 对外暴露 6 个 tools、2 个 resources、2 个 prompts；默认 `127.0.0.1:8001/mcp`；支持 bearer token |
| 审批 API | `api/main.py` → `agents/approvals.py` | `/agent/approvals` 查看、approve、reject 高风险动作 |
| Trace 汇总 | `agents/tracing.py` → `scripts/summarize_agent_traces.py` | 记录 agent_start/tool_call/agent_end，汇总 tool error、approval_required 和 latency |
| RAGAS/延迟评测 | `scripts/run_ragas_eval.py` / `scripts/measure_latency.py` → `core.pipeline.retrieve()` | 评测和标准产品 RAG 同口径，避免 pipeline drift |

---

## 0. 最新分支必须会：Agent + MCP-ready 四件套

### `agents/agent_loop.py`

**职责一句话**：基于 DeepSeek 原生 function calling 实现 ReAct-style agent loop：planner 看到工具 schema，返回 tool_calls；后端执行工具（默认 local，也可切 MCP backend），把结果作为 tool message 回灌；直到模型不再调用工具，输出最终答案。

**核心函数 / 状态**：
- `run_agent_loop(request, memory)`：主循环，记录 `metadata["steps"]`，用于调试和 agent eval。
- `_get_tool_backend()`：根据 `AGENT_TOOL_BACKEND` 选择 local backend 或 MCP backend。
- `tool_backend.tool_schemas()`：随每轮 planner 调用传给模型；local 下来自 `TOOL_SCHEMAS`，MCP 下来自 `tools/list` 转换。
- `call_counts: Counter`：统计 `(tool, args)`，超过 `AGENT_MAX_REPEAT` 就拦截重复调用。
- `AGENT_MAX_STEPS`：硬上限；触顶后禁用工具，强制模型基于已有信息回答。
- `AGENT_TOOL_OUTPUT_LIMIT`：截断过长工具输出，避免上下文膨胀。

**60 秒回答**：
> `/chat/agent` 不是一次意图分类，而是 plan → act → observe → re-plan。模型拿到 6 个工具 schema 后，如果返回 tool_calls，我解析函数名和 JSON 参数，通过工具 backend 调真实工具，再把结果以 `role=tool` 回灌给模型。默认 backend 是本地 function calling 工具层，保持原路径稳定；也可以切到 MCP backend，用 `tools/list` 动态发现工具、`tools/call` 执行工具。护栏包括 max_steps、同工具同参数重复检测、坏 JSON 降级、工具错误回灌、长输出截断和高风险发信人审。

**对应测试**：`tests/test_agent_loop.py`：无工具直接回答、传入 schema、单工具、多工具、坏 JSON、重复调用拦截、max_steps、输出截断、MCP backend 切换。

### `agents/tool_registry.py`

**职责一句话**：工具元数据单一事实源，把 6 个邮件工具的 name / description / parameters / function_name / risk_level / requires_approval 收敛到一个 registry，避免 function-calling schema 和 MCP server 注册各写一份。

**核心函数 / 状态**：
- `ToolSpec`：工具定义的数据结构，包含 `as_openai_tool()`。
- `TOOL_REGISTRY`：6 个工具的统一元数据，其中 `send_email` 是 `risk_level=high`、`requires_approval=True`。
- `openai_tool_schemas()`：从 registry 派生 DeepSeek/OpenAI-compatible `tools` schema。
- `tool_dispatch()`：从 registry 解析到 `agents.tools` 里的真实 Python 函数。

**60 秒回答**：
> 我没有把 MCP 当成另起炉灶，而是先抽了 `tool_registry.py`。同一份工具定义可以派生两套入口：一套给 DeepSeek function calling 的 `tools` 参数，一套给 FastMCP 注册 MCP tools。现在 registry 还承载风险等级，`send_email` 这类 high-risk tool 从 schema 层就标记需要人审。

**对应测试**：`tests/test_tool_registry.py`：registry 工具名、OpenAI schema shape、`agents.tools` 是否复用 registry 输出。

### `agents/mcp_adapter.py`

**职责一句话**：把 MCP 世界转换成现有 agent loop 能理解的工具后端：`tools/list` → OpenAI-compatible schema，`tools/call` → model-facing tool result。

**核心函数 / 状态**：
- `mcp_tool_to_openai_schema(tool)`：把 MCP tool 的 `name/description/inputSchema` 转成 function calling schema。
- `MCPToolBackend`：同步 backend facade，给 `agent_loop` 提供 `tool_schemas()` 和 `call_tool()`；内置 schema cache 和 audit log。
- `LocalToolBackend`：本地工具 backend，保持原 function-calling 路径。
- `StreamableHttpMCPClient`：用 MCP Python SDK 的 streamable HTTP client 连接 standalone MCP server；`MCP_AUTH_TOKEN` 非空时带 bearer token。
- `create_mcp_backend_from_settings()`：从 `MCP_SERVER_URL` 构建 MCP backend。

**60 秒回答**：
> MCP 不是替代模型的 function calling，而是替代后端工具接入协议。我这里让 planner 仍然用 DeepSeek tool_calls 做决策，但工具 schema 可以来自 MCP `tools/list`，工具执行可以走 MCP `tools/call`。这样 `/chat/agent` 的思考循环不用推倒，只把“工具从哪里来、怎么执行”变成可插拔 backend。

**对应测试**：`tests/test_mcp_adapter.py`：MCP tool schema 转换、structuredContent 解析、工具异常转 `{"error": ...}`。`tests/test_mcp_production.py`：auth header、schema cache、audit JSONL、server token verifier。

### `mcp_server.py`

**职责一句话**：用 MCP Python SDK / FastMCP 把邮件系统能力标准化暴露出去，供外部 MCP Host 或内部 MCP backend 调用。

**MCP 能力**：
- Tools：`search_emails` / `get_email` / `summarize_emails` / `draft_reply` / `send_email` / `email_stats`
- Resources：`email://{email_id}` / `email-corpus://stats`
- Prompts：`draft_reply_prompt` / `summarize_emails_prompt`

**60 秒回答**：
> `mcp_server.py` 是这次从“项目内部工具层”升级到“标准工具服务”的关键。它不重新实现工具，而是从 `tool_registry.py` 注册工具、调用 `agents.tools` 里的真实函数。默认用 Streamable HTTP 跑在 `127.0.0.1:8001/mcp`，避开 FastAPI 的 8000 端口。现在 `MCP_AUTH_TOKEN` 非空时会注入 bearer token verifier；面试时要讲清边界：这是作品集级生产化，线上还要补 OAuth、租户隔离、TLS 和密钥轮换。

**对应测试**：`tests/test_mcp_server.py`：fake FastMCP 注册 6 tools、2 resources、2 prompts；真实 SDK 导入和 `build_server()` 已做过冒烟。

### `agents/approvals.py`

**职责一句话**：Human-in-the-loop 的本地审批存储。高风险工具不会直接执行，而是创建 `pending` 审批单，再由 API approve/reject。

**核心函数 / 状态**：
- `ApprovalStore.create(action_type, payload, requested_by, risk_level)`：创建审批单，生成 `approval_id`。
- `ApprovalStore.list(status)` / `get(approval_id)`：给 API 或排查查看审批状态。
- `approve()`：把 pending 改成 approved，当前返回 `simulated_send`，表示安全边界已打通但未接真实 SMTP/企业邮箱。
- `reject()`：把 pending 改成 rejected，返回 `blocked_by_human`。

**60 秒回答**：
> 我没有让 agent 直接发邮件。`send_email` 是 high-risk tool，只会把 to/subject/body/rationale 写成 pending approval。人类通过 `/agent/approvals/{id}/approve` 或 reject 决定是否执行。当前是 JSON 文件存储和 simulated send，生产环境可以把 `ApprovalStore` 换成 Redis/DB，再接企业邮箱 API。

**对应测试**：`tests/test_approvals.py`：创建 pending action、approve/reject 状态流转；`tests/test_tools.py` 验证 `send_email` 不绕过审批。

### `agents/tracing.py`

**职责一句话**：把 agent run 的关键事件写入 JSONL，提供最小可观测能力。

**核心函数 / 状态**：
- `AgentTraceRecorder.record(event, **payload)`：记录 `agent_start`、`tool_call`、`agent_end`。
- `summarize_events(events)`：汇总 runs、tool_calls、tool_errors、approval_required、avg_tool_latency_ms。
- `scripts/summarize_agent_traces.py`：命令行汇总 `AGENT_TRACE_LOG_PATH`。

**60 秒回答**：
> Agent 出错不能只看最终回答。现在每次 `/chat/agent` 可以带 trace_id，trace 里有每一步 tool call、状态、耗时，以及是否触发人审。`run_agent_eval.py` 也把 trace_id 写进评测记录，这样面试官问“线上调错工具怎么排查”，我可以从 eval case 反查完整工具轨迹。

**对应测试**：`tests/test_agent_tracing.py`：JSONL 写入与汇总指标；`tests/test_agent_loop.py` 验证 loop 产出 trace metadata。

### `agents/tools.py`

**职责一句话**：保留 6 个真实工具函数和 `call_tool()` 参数校验；schema / dispatch 元数据已迁移到 `agents/tool_registry.py` 派生。

**6 个工具**：
- `search_emails(query, sender, date_hint, labels, limit)`：混合检索 + post-filter + rerank，返回 compact hit，含 `email_id`。
- `get_email(email_id)`：按 email_id 拼回完整邮件。
- `summarize_emails(query)`：复用 `SummarizerAgent`。
- `draft_reply(instruction, email_id, query)`：有 email_id 就精确回信，否则走 WriterAgent 检索起草。
- `send_email(to, subject, body, rationale)`：高风险动作，只创建 pending approval，不直接发送。
- `email_stats()`：复用 Analyzer 的统计函数。

**60 秒回答**：
> 工具不是凭空造的新能力，而是把原来的 Retriever/Summarizer/Writer/Analyzer 能力工具化。现在 `TOOL_SCHEMAS` 和 `TOOL_DISPATCH` 都从 registry 派生，避免 MCP server 和 function calling schema 各维护一份。`call_tool()` 仍负责本地执行路径的护栏：用 `inspect.signature` 丢弃幻觉 kwarg、检查必填参数、捕获异常并返回 `{"error": ...}`，这样工具失败不会把整个接口打成 500，而是作为观察结果回灌给模型。

**对应测试**：`tests/test_tools.py`：工具实现、schema 和 dispatch 一致、缺参、未知参数、工具异常；`tests/test_tool_registry.py` 补 registry 一致性。

### `core/pipeline.py`

**职责一句话**：统一检索链路的单一事实源：rewrite → extract filters → hybrid_search → post-filter → rerank。

**为什么重要**：
> 之前产品链路、RAGAS 评测、延迟测试各自复制检索逻辑，导致评测脚本抽了 filters 却没应用 post-filter，测的不是用户真实走的 pipeline。现在收敛到 `retrieve()`，产品和评测共用同一条链路，减少 pipeline drift。

**边界别说错**：这里说的是标准 RAG 路径和离线评测路径。`/chat/graph` 和 `/chat/agent` 会复用 `hybrid_search`、`apply_post_filters`、`rerank` 等底层组件，但不是完整调用 `retrieve()`。

**对应测试**：`tests/test_pipeline.py`：验证 `retrieve()` 的执行顺序、filter fallback、后过滤行为和 rerank 入口。

---

## 1. `agents/coordinator.py`

**职责一句话**：把用户 query 用 LLM 分类成 5 种意图，路由到对应 Agent；任何异常都回退到 GENERAL，绝不让用户看到 500。

**核心函数 / 类**：
- `classify_intent(query) -> IntentType`：调 DeepSeek 用 `_INTENT_SYSTEM` prompt，解析返回的 JSON `{"intent": "...", "reason": "..."}`。3 段防御：① ```json ... ``` 围栏剥离 ② 找最后一对 `{...}` ③ 失败 / 异常 / 未知 intent 全部回退 GENERAL。
- `route(request, memory) -> AgentResponse`：分类后通过内部 `agent_map` 选 Agent 实例化并 run；GENERAL 也走 RetrieverAgent（不报错）。

**关键状态变量**：
- `_client`（module-level，懒加载 OpenAI client）
- `agent_map`：是 **`route()` 内部局部 dict**，不是模块全局——这点很重要，后面"易混"会再提。

**最容易被问的面试问题**：你为什么不用关键词匹配做意图分类？

**60 秒回答**：
> 关键词列表写死会漏同义词——"找一下" / "搜搜" / "帮我看看" / "给我找" 全是 retrieve 意图，列举不完。LLM 分类有泛化能力，不用维护词典。代价是一次 LLM 调用 ~1s 延迟和 token 成本，能接受。三段降级护底：JSON 解析失败 / LLM 抛异常 / 未知 intent 字符串都回退 GENERAL。Coordinator 是软入口不是硬卡点——分错了大不了走默认管线，永远不让用户看到 500。

**对应测试文件**：`tests/test_coordinator.py`（11 个用例：5 种 intent 参数化 + reasoning_content 兜底 + ```json 围栏剥离 + 3 类异常 fallback + `route()` GENERAL smoke test）

**容易混淆 / 注意点**：
- IntentType 枚举有 5 个，agent 类只有 4 个（GENERAL 复用 RetrieverAgent）。
- `agent_map` 是局部变量，**不能直接 assert `set(agent_map.keys()) == set(IntentType)`**——要测就测 GENERAL fallback 的实际行为（`route()` 是否真的把请求送到 RetrieverAgent.run）。
- `max_tokens=1500` 给推理模型留余量；intent 分类推理量小，1500 够用，不需要 3000。

---

## 2. `agents/retriever_agent.py`

**职责一句话**：普通 RAG 检索 Agent。当前分支里检索细节已经下沉到 `core.pipeline.retrieve()`，它主要负责拿 contexts、调用 generator 生成答案，并给 `/chat/stream` 暴露 `prepare_contexts()`。

**核心函数 / 类**：
- `prepare_contexts(query)`：公开接口，内部直接调用 `retrieve(query)`，返回 reranked SearchResult list；`/chat/stream` 依赖它做 token 级流式生成。
- `run(request, memory)`：调用 `prepare_contexts()` 拿上下文，再用 `generate_answer()` 生成最终答案。
- 具体 rewrite/filter/post-filter/rerank 已移动到 `core.pipeline.py`，这里不要再说 RetrieverAgent 自己实现整条链。

**关键状态变量**：RetrieverAgent 本身基本无状态；检索开关和 top_k/rerank 参数在 `config.settings`，检索执行细节在 `core.pipeline`。

**最容易被问的面试问题**：filter 没匹配到为什么不报错？

**60 秒回答**：
> 当前代码里 filter 逻辑在 `core.pipeline.apply_post_filters()`，不是 RetrieverAgent 私有方法。三个 filter 都是软过滤：filter 后空集就 fall back 原 list。理由是用户的 sender / date hint 多数是模糊提示，LLM 抽出来的字段未必和 metadata 严格匹配。硬过滤一旦没命中就直接没结果，用户体验崩。软过滤的代价是召回稍多噪声，由 reranker 兜底排序。

**对应测试文件**：`tests/test_pipeline.py` 覆盖检索链路；`tests/test_retriever.py` 覆盖底层 hybrid_search / BM25 / RRF。

**容易混淆 / 注意点**：
- `prepare_contexts` 是给 `/chat/stream` 用的公开接口，**不是私有的 _prepare**，名字别误改。
- 面试时不要再说 RetrieverAgent 内部有 `_rewrite_query/_extract_filters`，这些已经抽到 `core.pipeline.py`。

---

## 3. `core/retriever.py`

**职责一句话**：向量 + BM25 双路检索，根据 `ENABLE_BM25 / ENABLE_RRF` flag 切三个分支：纯向量 / 简单合并 / RRF 融合。

**核心函数 / 类**：
- `_tokenize(text)`：正则 `[一-鿿]|[a-zA-Z0-9]+`，中文按字、英文 / 数字按整体、小写化。
- `vector_search / bm25_search / hybrid_search`：三个公开接口。
- `_get_bm25_index()`：返回 cached `(BM25Okapi, all_chunks, corpus)`，用 `collection.count()` 当 cache key。
- `invalidate_bm25_cache()`：被 `/index` 和 `/index/clear` 显式调用。

**关键状态变量**：
- `_bm25_lock = threading.Lock()`：保护"check + rebuild"整段，防 cache stampede。
- `_bm25_cache: Optional[Tuple[int, BM25Okapi, List[dict], List[str]]]`：tuple = `(chunk_count, bm25, all_chunks, corpus)`，模块级单例。
- `RRF_K = 60`：在 `hybrid_search` 内部 hardcoded（不是 cfg）。
- `cfg.VECTOR_WEIGHT = 0.7` / `cfg.BM25_WEIGHT = 0.3`：在 cfg。

**最容易被问的面试问题**：BM25 缓存怎么保证并发安全？

**60 秒回答**：
> threading.Lock 包**整个 check-then-rebuild 块**——这一步关键。如果只锁 rebuild，并发首次冷启动时多个线程都会 miss → 都去重建一次（cache stampede），白白做几次 800ms 的工作。Cache key 选择也讲究——第一版想用语料 hash，但要算 hash 就得先读全量数据，缓存白做。改用 ChromaDB 的 `collection.count()`：O(1)、不读数据，覆盖"邮件数没变就索引可复用"这个最常见的不变式。冷启动 846ms → 命中 22ms，40 倍提速。

**对应测试文件**：`tests/test_retriever.py`（9 个用例：tokenizer 中英 / `ENABLE_BM25=False` 跳过 / RRF 融合公式逐分计算 / 简单合并去重 / cache 命中 / 漂移失效 / 主动 invalidate / 空集合 / 零分过滤）

**容易混淆 / 注意点**：
- 简单合并分支（`ENABLE_RRF=False`）：先 `result_map = {bm25}` 再 `result_map.update({vec})`，**vec 覆盖同 key bm25 的 score**——结果里同一 chunk 取的是 vec 分。
- BM25 score `<= 0` 的会被过滤，不进 results。
- RRF 公式：`fused[k] = VECTOR_WEIGHT/(rank_v + 60) + BM25_WEIGHT/(rank_b + 60)`。

---

## 4. `core/reranker.py`

**职责一句话**：LLM-based reranker，给候选 chunk 打 0~10 分；带熔断器，连续 3 次失败就降级到原（RRF）排序。

**核心函数 / 类**：
- `rerank(query, results, top_n)`：主入口，受 `ENABLE_RERANKER` flag 控制；`len(results) <= 1` 直接返回。
- `reset_circuit_breaker()`：评测脚本每版开始时调一次（关键！）。
- `_get_client()`：懒加载 OpenAI client。
- `max_tokens=3000`（推理量大时 1500 不够，实测 reasoning_len 可达 2700）。

**关键状态变量**：
- `_consecutive_failures = 0`（**module-level 全局**！这是熔断器跨版本污染的根源）
- `_FAILURE_THRESHOLD = 3`
- `_client`（懒加载单例）

**最容易被问的面试问题**：你的熔断器和教科书的有什么差别？

**60 秒回答**：
> 教科书三状态机：closed → open → half-open + timer。我这是简化版：只有 closed / open，靠下一个请求成功才能恢复——但下一个请求来了就会跳过 LLM，所以"实际是一直降级直到 RAGAS 跑完时被 reset_circuit_breaker() 显式重置"。这是已知简化，对 demo 项目够用，上线必须补 timer。最大的坑是 `_consecutive_failures` 是 **module-level 全局** + 长跑进程 —— V3 失败的计数会泄漏到 V4/V5/V6，污染消融对比。修复加 `reset_circuit_breaker()` 评测每版调一次。**同样的问题在生产长跑进程上是同一类**——熔断器从早高峰带到晚高峰，从 staging 带到 prod。

**对应测试文件**：`tests/test_eval.py::test_evaluate_version_resets_circuit_breaker_and_aggregates`（spy 断言每版调用一次 reset）。**reranker 本体没单独测试文件** —— `max_tokens=3000` / 失败计数累加 / 重试逻辑是已知测试空白。

**容易混淆 / 注意点**：
- 失败计数 reset 时机：**仅在 LLM 成功打分后**；熔断打开后**不会**自动 reset，必须靠 `reset_circuit_breaker()` 或下一次成功调用。
- `len(results) <= 1` 直接 `return results[:top_n]`——一个候选不需要重排。
- score count mismatch（LLM 返回的 scores 长度和候选数不等）也会触发熔断器计数 +1。

---

## 5. `core/memory.py`

**职责一句话**：线程安全的滑动窗口对话记忆，保留最近 `max_turns` 轮（user + assistant 各算一条消息）。

**核心函数 / 类**：
- `class ConversationMemory(max_turns=5)`：默认 5 轮 = `max_messages = 10` 条消息。
- `add(role, content)` / `to_messages()` / `clear()` / `__len__()`：每个方法都用 `self._lock` 保护。

**关键状态变量**：
- `self._turns: List[Turn]`（`Turn` 是 `@dataclass` 含 `role` 和 `content`）
- `self._lock = threading.Lock()`（**第二把锁**——session 内部的）

**最容易被问的面试问题**：为什么要加锁？单进程不就只有一个用户吗？

**60 秒回答**：
> FastAPI 用 thread pool 处理请求，**同一 session_id 可能被多个线程并发访问**——比如一个流式请求还在后台 stream，新请求又来读 history。无锁的话 list 在 slice-rebind（`_turns[-max_messages:]`）时可能被撕裂——读到一半的状态。**双锁分工**：第一把在 `api/main.py::_get_session`，保护 session 字典本身的 get-or-create 原子性；第二把在 `ConversationMemory` 内部，保护 session 的对话列表。一把大锁也能跑，但所有用户串行化——双锁让不同 session 之间真正并行，同 session 内部串行。

**对应测试文件**：`tests/test_memory.py`（7 个用例：默认 max_turns=5 / 滑窗裁剪 / `to_messages` 顺序 / `clear` / `_get_session` 隔离 / **8 线程 × 50 条并发 add 不丢消息** / `max_turns=1` 边界）

**容易混淆 / 注意点**：
- `max_turns=5` 实际是 `max_messages = max_turns * 2 = 10` 条消息（user + assistant 配对）。
- session store `_sessions: dict[str, ConversationMemory]` 在 `api/main.py` 里，是**进程内 dict**——多 worker 不共享，要换 Redis。
- 滑窗是 `_turns[-max_messages:]` 切片重绑——所以并发场景必须锁。

---

## 6. `core/chunker.py`

**职责一句话**：邮件按段落切 chunk，超长段落强切并保留 overlap，过短尾巴并入前一个 chunk。

**核心函数 / 类**：
- `chunk_text(text, chunk_size, chunk_overlap, min_chunk_size)`：主入口，段落优先。
- `_split_paragraphs(text)`：按 `\n\n+` split + strip 过滤空行。
- `_force_split(text, size, overlap)`：滑动窗口强切，`step = max(1, size - overlap)` 防 `overlap >= size` 死循环。
- `chunk_email(email)`：包装成 `Subject: {subject}\n\n{body}`，labels / recipients 序列化为 JSON 字符串（ChromaDB metadata 只接受 primitives）。

**关键状态变量**：无（纯函数）

**最容易被问的面试问题**：你最近修过什么 bug？（讲 `chunk_overlap=0`）

**60 秒回答**：
> 写 Day 10 测试时构造 `chunk_overlap=0` 的边界用例，输出和预期对不上。追下去发现代码写的是 `overlap = chunk_overlap or cfg.CHUNK_OVERLAP` —— `0 or 50 == 50`，0 当 falsy 被吞。顺手发现 `_force_split` 在 `overlap >= size` 时 step 是 0 会无限循环。修成 `is None` 判定 + `step = max(1, size - overlap)`，加回归测试 `test_zero_overlap_is_respected`。**这个 bug 体现的是测试当 design tool 用——测试不只是验证已知行为，还能反向暴露规约 bug**。这是工程素养的指标，不是 boilerplate。

**对应测试文件**：`tests/test_chunker.py`（8 个用例：短文本 / 段落合并 / 强切 overlap / **零 overlap 回归** / `_force_split` 直测 / 短尾合并 / 空文本 / `chunk_email` 元数据序列化）

**容易混淆 / 注意点**：
- ChromaDB metadata 只接受 primitives（str / int / float / bool）—— labels / recipients 必须 `json.dumps`。
- `thread_id or ""`：None 转空字符串，因为 metadata 不接受 None。
- 切分管线是四阶段：段落切 → 缓冲合并 → 超长强切 → 短尾合并。每一阶段都有边界情况测试。

---

## 7. `api/main.py`

**职责一句话**：FastAPI 入口，提供 `/chat`（Coordinator 稳定链路）/ `/chat/stream`（SSE 真流式）/ `/chat/graph`（Self-RAG）/ `/chat/agent`（function-calling Agent）四种问答端点 + `/index` 系列；session 存进程内 dict。

**核心函数 / 类**：
- `_get_session(session_id)`：双锁中的"外锁"——保护 session 字典 get-or-create 的原子性。
- `/chat/stream` 的 `event_generator`：**worker 线程 + `asyncio.Queue` + `loop.call_soon_threadsafe`** 桥接同步 SDK 和 asyncio 事件循环。
- `/chat/agent`：调用 `run_agent_loop(request, memory)`，让 planner 自主选择 search/get/summarize/draft/stats 工具。
- 三种 SSE 事件：`__intent__`（先发意图标识让 UI 渲染对应 spinner）/ token / `__error__`。
- 流结束后才把 `user query + 最终 answer` 写进 memory（不像同步 `/chat` 立即写）。

**关键状态变量**：
- `_sessions: dict[str, ConversationMemory]`（**进程内**——多 worker 不共享）
- `_sessions_lock = threading.Lock()`（**第一把锁** / 外锁）

**最容易被问的面试问题**：你 SSE 怎么实现的？

**60 秒回答**：
> OpenAI SDK 的 stream 是**同步阻塞迭代器**（`__iter__`），不是 async iterator——FastAPI 是 asyncio 框架，直接 `for token in stream` 会阻塞事件循环。原代码用 `tokens = list(stream_generate(...))` eager 消费，等所有 token 收齐才 yield，等于伪流式。改成 worker 线程跑阻塞 `for token in stream_generate(...)` 循环，每个 token 通过 `loop.call_soon_threadsafe(queue.put_nowait, token)` **跨线程投递**；消费者协程 `await queue.get()` 逐个 yield SSE 事件。`call_soon_threadsafe` 是跨线程往事件循环投递任务的**唯一安全方式**——asyncio.Queue 本身不是线程安全的。三个候选方案（换 AsyncSDK / 全同步 / Queue 桥接）选 Queue 桥接因为侵入最小。

**对应测试文件**：`tests/test_memory.py::test_get_session_isolates_session_ids_and_returns_same_instance`（验证 `_get_session` 隔离）；**SSE 流式逻辑没单测**（依赖 asyncio 测试比较麻烦，是已知空白）。

**容易混淆 / 注意点**：
- `/chat`、`/chat/graph`、`/chat/agent` 端点都会调 `memory.add` 两次（user + assistant）；`/chat/stream` 在流结束后**一次性**写。
- `__intent__` / `__error__` / token 三种事件用 tuple tag 区分（tuple 表示控制事件，str 表示 token）。
- `loop.run_in_executor(None, producer)` 不 await——producer 在后台跑，`event_generator` 通过 `SENTINEL` 知道何时退出循环。
- `/chat/stream` 用 `prepare_contexts` 而非 `run`——因为 run 会一次性 generate，stream 需要 token 级流。

---

## 7.5 三个补充文件：面试常被顺手追

### `agents/graph_workflow.py`

**职责一句话**：Self-RAG 的 LangGraph 状态机。它先检索，再用 grade 判断上下文够不够；不够就 rewrite 重试，够了才 generate。

**要讲清的边界**：这条链路复用 `hybrid_search` 和 `rerank`，但没有完整调用 `core.pipeline.retrieve()`。所以它是“Self-RAG 独立实验路径”，不是标准 `/chat` 的简单包装。

> 60 秒回答：`/chat/graph` 不是为了替代默认链路，而是用 LangGraph 验证“检索后自检”。如果 grade 觉得上下文不相关，就 rewrite 再查；如果相关就直接生成。它的价值是提高可解释性和纠错能力，代价是多一次或多次 LLM 调用，延迟更高。

### `scripts/run_agent_eval.py`

**职责一句话**：评估 Agent 是否会正确选择工具，而不是只看最后答案漂不漂亮。

- `task_success_rate`：最终回答是否完成任务。
- `tool_accuracy`：实际工具调用是否覆盖预期工具；允许合理中间步骤，比如多一次 `get_email`。
- `avg_steps`：平均工具轮数，观察是否绕路。
- `max_steps_reached_rate`：是否经常撞到最大步数，撞多了说明 planner 或工具设计有问题。

> 60 秒回答：RAGAS 看检索和生成质量，agent eval 看“模型会不会用工具”。我现在首批 8 条任务是小样本健康检查，不能吹成生产级 benchmark，但它能防止 search、get、draft 这种多步链路退化。

### MCP 相关文件

**`agents/tool_registry.py`**：工具元数据单一来源，防止 function calling schema 和 MCP tool schema 漂移。

**`agents/mcp_adapter.py`**：把 MCP `tools/list` 的 tool 转成 OpenAI-compatible schema，把 MCP `tools/call` 的结果转回 agent loop 的 tool result。

**`mcp_server.py`**：standalone MCP server，暴露 6 tools + 2 resources + 2 prompts。启动命令：

```powershell
.\.venv\Scripts\python.exe mcp_server.py --transport streamable-http
```

> 60 秒回答：MCP 这次不是“追热点”，而是为了把本地工具层标准化。默认 `AGENT_TOOL_BACKEND=local` 保持原 agent 路径稳定；当设置为 `mcp` 时，agent loop 通过 MCP 动态发现和调用工具。现在还补了 bearer token、schema cache、audit JSONL、人审和 trace。这样既有兼容性，也为外部 MCP Host 或企业系统接入留了协议边界。

### `frontend/app.py`

**职责一句话**：Streamlit 前端，负责模式选择、session_id、历史展示、普通请求和 SSE 流式渲染。

- `st.session_state["session_id"]`：每个浏览器会话一个 sid，发给后端隔离 memory。
- 普通请求：POST `/chat`、`/chat/graph` 或 `/chat/agent`，一次性拿答案。
- 流式请求：POST `/chat/stream`，用 `requests(stream=True)` + `sseclient` 消费 SSE。

> 60 秒回答：前端不是浏览器 JS，而是 Streamlit Python。SSE 也是 Python 端用 `sseclient` 解析，所以面试时要区分“项目实际实现”和“如果换 Web 前端会用 fetch streaming”。

---

## 8. `scripts/run_ragas_eval.py`

**职责一句话**：6 版消融评测脚本——按 `VERSION_FLAGS` 切换 `ENABLE_*` flag，跑 testset，三维度打分（LLM 主路径 / 向量降级 / 全 0 兜底），输出 `V{1..6}.json` + `comparison.json`。

**核心函数 / 类**：
- `VERSION_FLAGS`：V1-V6 配置 dict，是消融实验的 source of truth。
- `apply_flags(flags)`：直接 `setattr(cfg, ...)` 改全局 flag。
- `run_single(client, q, gt)`：跑一题完整 RAG 管线（rewrite → extract filters → hybrid_search → post-filter → rerank → generate）。
- `score_response(client, q, a, contexts)`：**三段降级**——LLM 3 次重试 → embedding 余弦 → 全 0。
- `_score_by_embedding`：用 bge-m3 算 query / answer / contexts 的余弦相似度近似三维度。
- `_extract_json_obj(text)`：抽出 ```json ... ``` 围栏内或最后一对 `{...}`。
- `evaluate_version(version, testset, limit, client)`：**每版开始调 `reset_circuit_breaker()`**，单题失败不污染整版（error 写进 record 但其他题继续）。

**关键状态变量**：无 module-level 状态（直接改 `cfg` 属性，每版 `apply_flags` 覆盖）。

**最容易被问的面试问题**：你为什么不用官方 RAGAS 包？

**60 秒回答**：
> 两个原因。一，**推理模型不兼容**：DeepSeek `deepseek-v4-flash` 输出 `content` + `reasoning_content` 双字段，max_tokens 不够时 content 空但 reasoning 里有 JSON。官方 ragas 按 OpenAI 标准 schema 解析，识别不出，会判 LLM 失败。二，**降级路径需要显式控制**：希望 LLM 打分失败时确定性降级到向量相似度（而不是返回 NaN，让结果不可复现）。所以做了 RAGAS-style 自实现：LLM 主路径 3 次重试 → embedding 余弦兜底 → 全 0 终极兜底。每版开始还会 `reset_circuit_breaker()`，避免上一版的 reranker 失败计数泄漏污染消融对比。

**对应测试文件**：`tests/test_eval.py`（10 个用例：apply_flags 写回 cfg / `_extract_json_obj` 噪声 + 围栏 / score_response LLM 成功 / 3 次失败 → embedding 降级 / 都失败 → 全 0 / reasoning_content 兜底 / evaluate_version 输出结构 / **`reset_circuit_breaker` spy 断言每版一次** / 单题失败隔离）

**容易混淆 / 注意点**：
- `--limit 30` 默认每版只跑 30 题，全 100 题要显式传 `--limit 100`。
- LLM 打分时 `max_tokens=3000`（reasoning 占大头）。
- `apply_flags` 直接改 cfg 属性—— **测试一定要 monkeypatch 撤销**，否则污染后续测试。
- `data/eval_results/V{1..6}.json` 含逐题记录（answer / contexts）；`comparison.json` 只含三维度均值汇总。
- **不要混淆 RAGAS（评测）和 reranker（管线组件）**——前者是离线评估工具，后者是在线检索环节。

---

## 一句话装进脑子

> **当前关键文件分三层**：
>
> - **API 层**（`api/main.py`）+ **Agent 层**（`coordinator` / `agent_loop` / `tools` / `tool_registry` / `mcp_adapter` / specialist agents）= 用户路径、工具编排和 MCP-ready backend
> - **Core 层**（`pipeline` / `chunker` / `retriever` / `reranker` / `memory`）= 检索、数据、状态原语
> - **MCP 层**（`mcp_server.py`）= 标准化暴露 tools/resources/prompts
> - **Scripts 层**（`run_ragas_eval` / `run_agent_eval`）= 检索评测 + Agent 行为评测
>
> **module-level 全局可变状态**出现在三处——reranker 熔断器、memory 的 session 字典、retriever 的 BM25 cache——**每处都搭配显式锁或 reset 钩子**。
>
> **测试边界**清楚：unit test 测 deterministic 代码（107 用例 mock 全过），RAGAS-style evaluation 测检索/生成质量，agent eval 和 trace 测真实多步工具行为。
>
> **降级层叠**也清楚：LLM 调用 → reasoning_content 兜底 → 异常回退默认值；reranker → 熔断器 → 原排序；score → embedding → 全 0；intent → GENERAL。每一层都有兜底，永远不让用户看到 500。

---

**END**
