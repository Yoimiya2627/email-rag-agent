# Email RAG Agent

> **Author**: 赵伟鑫 (Yoimiya2627) — Agent 开发工程师 / 大模型应用开发工程师
> **Contact**: a1486807398@163.com | [GitHub](https://github.com/Yoimiya2627)

一个面向邮件场景的 Agentic RAG 系统。底层是向量检索 + BM25 + RRF 的混合检索 RAG；上层是 DeepSeek 原生 function calling 的 ReAct-style agent loop；工具层已经升级为 MCP-ready backend，支持 FastMCP tools/resources/prompts、可选 MCP 鉴权、工具级权限策略、审计查询、人审审批和 EvalOps trace/report。

这个项目的重点不是“调一个 LLM API”，而是把邮件 RAG 能力做成可编排、可评测、可回归、可审计的 Agent 工程系统。

## 目录

- [核心亮点](#核心亮点)
- [Demo](#demo)
- [功能能力](#功能能力)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [系统架构](#系统架构)
- [Agent 工具层](#agent-工具层)
- [MCP Server](#mcp-server)
- [Human-in-the-loop](#human-in-the-loop)
- [Trace 与 Eval](#trace-与-eval)
- [评测结果](#评测结果)
- [测试](#测试)
- [API 端点](#api-端点)
- [目录结构](#目录结构)
- [已知限制和下一步](#已知限制和下一步)

## 核心亮点

- **Function-calling Agent Loop**：`/chat/agent` 使用 DeepSeek 原生 tool calls，多轮执行 `plan -> tool_call -> observe -> re-plan`。
- **MCP-ready Tool Backend**：工具定义集中在 `agents/tool_registry.py`，同源派生本地 function schema 和 FastMCP 注册。
- **MCP 生产化基础**：MCP client 支持 bearer token header；server 可启用 token verifier；工具调用写 JSONL 审计；MCP tools/list 有 schema cache；MCP server 支持 allowed-tools 和 read-only 工具可见性策略。
- **Human-in-the-loop 安全链路**：新增高风险 `send_email` 工具，但它只创建 pending approval，不会直接发信；人类通过 API approve/reject。
- **Agent EvalOps**：54 条 agent 任务集覆盖多步、异常、歧义、权限和高风险发信；eval record 关联 `trace_id`，可输出失败归因和 Markdown 报告。
- **RAG 消融评测**：6 版 RAGAS-style 对比，量化 BM25、RRF、reranker、query rewrite 的 ROI。
- **工程护栏**：max steps、重复工具调用检测、坏 JSON 降级、参数校验、工具异常回灌、工具输出截断。
- **118 个 pytest**：覆盖 RAG、pipeline、tools、tool registry、MCP adapter/server/production/policy/audit API、approval、trace、EvalOps、agent loop、agent eval。

## Demo

[![Demo preview](docs/demo.png)](docs/demo.mp4)

约 1 分 40 秒：邮件检索、预算查询、统计分析、evaluation 表格。点击预览图打开 MP4。

## 功能能力

- 邮件问答：从 5000 封邮件中检索并回答事实问题。
- 多步 Agent：例如“找出报销邮件并帮我起草回复”，agent 会自主 `search_emails -> get_email -> draft_reply`。
- 人审发信：`send_email` 创建审批单，必须人类确认后才进入 simulated send。
- 批量摘要：按主题检索多封邮件并生成结构化摘要。
- 回信草稿：针对检索到的邮件或指定 `email_id` 起草回复。
- 统计分析：发件人 Top、标签分布、每日邮件量。
- 多轮记忆：按 session 隔离，默认保留 5 轮滑窗。
- SSE 真流式：worker 线程 + `asyncio.Queue` 桥接同步 LLM SDK 与 FastAPI SSE。
- Self-RAG：LangGraph 状态机，检索结果不相关时 rewrite query，最多重试 2 次。
- Agent EvalOps：`data/agent_testset.json` 维护 54 条带类型/风险/期望工具/禁用工具/成功标准的任务，eval 可生成 Markdown 报告。

## 快速开始

### Windows PowerShell

```powershell
copy .env.example .env
# 编辑 .env，填 DEEPSEEK_API_KEY
.\tasks.ps1 install
.\tasks.ps1 index
.\tasks.ps1 run
```

如果 PowerShell 拦截脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS / Linux

```bash
cp .env.example .env
# 编辑 .env，填 DEEPSEEK_API_KEY
make install
make index
make run
```

打开：

- Frontend: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- MCP server: http://127.0.0.1:8001/mcp

## 配置说明

唯一必填环境变量是 `DEEPSEEK_API_KEY`。完整配置见 [`.env.example`](.env.example)。

| 配置 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | 普通 RAG 生成、重排和打分模型 |
| `AGENT_PLANNER_MODEL` | `deepseek-chat` | Agent planner，function calling 更轻更快 |
| `AGENT_MAX_STEPS` | `6` | 单次 agent 任务最多工具轮数 |
| `AGENT_MAX_REPEAT` | `2` | 同工具同参数重复超过后拦截 |
| `AGENT_TOOL_OUTPUT_LIMIT` | `4000` | 单次工具结果最大字符数 |
| `AGENT_TOOL_BACKEND` | `local` | `local` 进程内工具；`mcp` 走 MCP server |
| `MCP_SERVER_URL` | `http://127.0.0.1:8001/mcp` | Streamable HTTP MCP 地址 |
| `MCP_AUTH_TOKEN` | 空 | 填写后 MCP client/server 启用 bearer token |
| `ENABLE_MCP_AUDIT` | `true` | MCP 工具调用写审计 JSONL |
| `MCP_AUDIT_LOG_PATH` | `./data/audit/mcp_audit.jsonl` | MCP 审计日志 |
| `MCP_ALLOWED_TOOLS` | 空 | 逗号分隔的 MCP 可见工具白名单；空表示全部可见 |
| `MCP_READ_ONLY_MODE` | `false` | `true` 时 MCP server 只暴露 low-risk 且不需要人审的只读工具 |
| `APPROVAL_STORE_PATH` | `./data/approvals/pending_actions.json` | 人审审批存储 |
| `ENABLE_AGENT_TRACE` | `false` | 是否记录 agent trace JSONL |
| `AGENT_TRACE_LOG_PATH` | `./data/traces/agent_traces.jsonl` | trace 输出路径 |

Feature flags 默认是 V2 推荐配置：`BM25=true, RRF=true, RERANKER=false, REWRITE=false`。

| Flag | 默认 | 说明 |
|---|---:|---|
| `ENABLE_BM25` | true | 向量 + BM25 混检 |
| `ENABLE_RRF` | true | Reciprocal Rank Fusion |
| `ENABLE_RERANKER` | false | LLM reranker，质量可能提升但延迟高 |
| `ENABLE_QUERY_REWRITE` | false | LLM query rewrite |

## 系统架构

```text
Streamlit UI
   |
FastAPI
   |-- /chat         -> Coordinator -> Specialist Agents -> core.pipeline.retrieve()
   |-- /chat/stream  -> Coordinator -> worker thread -> asyncio.Queue -> SSE
   |-- /chat/graph   -> LangGraph Self-RAG -> retrieve/grade/rewrite/generate
   |-- /chat/agent   -> function-calling planner -> local/MCP tool backend
   |-- /agent/approvals -> human approve/reject high-risk actions
   |-- /agent/mcp-audit -> query MCP tool-call audit logs
   |
Core RAG
   |-- query rewrite
   |-- hybrid search: bge-m3 vector + BM25
   |-- RRF fusion
   |-- metadata post-filter
   |-- optional reranker
   |-- DeepSeek generation
```

Agent 工具 backend：

```text
agents/tool_registry.py
   |-- openai_tool_schemas() -> agents/tools.py -> LocalToolBackend
   |-- ToolSpec metadata     -> mcp_server.py -> FastMCP tools/resources/prompts

agents/agent_loop.py
   |-- AGENT_TOOL_BACKEND=local -> in-process call_tool()
   |-- AGENT_TOOL_BACKEND=mcp   -> tools/list + tools/call via Streamable HTTP
```

完整架构见 [docs/architecture.md](docs/architecture.md)。

## Agent 工具层

| Tool | 风险级别 | 是否人审 | 作用 |
|---|---|---:|---|
| `search_emails` | low | 否 | 混合检索邮件，支持 sender/date/labels 过滤 |
| `get_email` | low | 否 | 按 `email_id` 获取完整邮件 |
| `summarize_emails` | low | 否 | 检索并总结相关邮件 |
| `draft_reply` | medium | 否 | 起草回信，不发送 |
| `send_email` | high | 是 | 只创建 pending approval，不直接发信 |
| `email_stats` | low | 否 | 邮件统计聚合 |

`call_tool()` 会丢弃模型幻觉参数、检查必填参数、捕获工具异常并返回 `{"error": ...}`，让错误以 tool result 形式回灌给模型，而不是把 HTTP 请求打成 500。

## MCP Server

启动 MCP server：

```powershell
.\.venv\Scripts\python.exe mcp_server.py --transport streamable-http
```

或：

```bash
python mcp_server.py --transport streamable-http
```

切换 `/chat/agent` 到 MCP backend：

```env
AGENT_TOOL_BACKEND=mcp
MCP_SERVER_URL=http://127.0.0.1:8001/mcp
MCP_AUTH_TOKEN=optional-shared-secret
```

MCP 暴露能力：

| 类型 | 内容 |
|---|---|
| Tools | 6 个 agent tools |
| Resources | `email://{email_id}`、`email-corpus://stats` |
| Prompts | `draft_reply_prompt`、`summarize_emails_prompt` |

生产化基础：

- Bearer token：`MCP_AUTH_TOKEN` 非空时 client 自动带 `Authorization: Bearer ...`，server 注入 `StaticBearerTokenVerifier`。
- Schema cache：`MCPToolBackend` 缓存 `tools/list` 结果，避免每轮 planner 都重新发现工具。
- Audit JSONL：MCP tool call 写入 `MCP_AUDIT_LOG_PATH`，记录 tool、status、latency、request_id。
- Tool policy：`MCP_ALLOWED_TOOLS` 可按工具白名单收敛暴露面；`MCP_READ_ONLY_MODE=true` 时只暴露 low-risk 且不需要人审的工具。
- Audit query：`GET /agent/mcp-audit?tool=email_stats&status=success&limit=20` 可按工具和状态查看审计事件。

## Human-in-the-loop

`send_email` 是高风险工具。agent 调用时只会生成审批单：

```json
{
  "status": "pending_approval",
  "approval_id": "...",
  "message": "发送邮件属于高风险动作，已创建待审批请求，需要人工确认后才会执行。"
}
```

审批 API：

| Method | Path | 说明 |
|---|---|---|
| `GET` | `/agent/approvals?status=pending` | 查看审批单 |
| `POST` | `/agent/approvals/{approval_id}/approve` | 人工批准，当前为 simulated send |
| `POST` | `/agent/approvals/{approval_id}/reject` | 人工拒绝 |

当前实现是本地 JSON 文件存储，面试口径是“接口和边界已经打通，生产环境可把 `ApprovalStore` 换成 Redis/DB，并接企业邮箱 API”。

## Trace 与 Eval

开启 trace：

```env
ENABLE_AGENT_TRACE=true
AGENT_TRACE_LOG_PATH=./data/traces/agent_traces.jsonl
```

每次 agent run 会记录：

- `agent_start`
- `tool_call`：tool、arguments、status、latency_ms
- `agent_end`：status、steps、answer_chars

汇总 trace：

```powershell
.\.venv\Scripts\python.exe scripts\summarize_agent_traces.py --json
```

Agent eval：

```powershell
.\.venv\Scripts\python.exe scripts\run_agent_eval.py --limit 8 --report-output data/eval_results/agent_eval_report.md
```

`scripts/run_agent_eval.py` 会输出 task_success_rate、tool_accuracy、avg_steps、max_steps_reached_rate、forbidden_tool_violation_rate，并把每条记录关联 `trace_id`。任务集现在是 54 条元数据化 case，每条包含 `id`、`task_type`、`risk_level`、`expected_tools`、`forbidden_tools`、`success_criteria`；报告会给出 `failure_category`，用于区分 missing_expected_tool、forbidden_tool、tool_error、approval_required、max_steps 等问题。

## 评测结果

RAG 消融脚本：

```powershell
.\.venv\Scripts\python.exe scripts\run_ragas_eval.py --versions V2
.\.venv\Scripts\python.exe scripts\run_ragas_eval.py
```

6 个版本：

| Version | BM25 | RRF | Reranker | Rewrite |
|---|---:|---:|---:|---:|
| V1 | false | false | false | false |
| V2 | true | true | false | false |
| V3 | true | true | true | false |
| V4 | true | true | true | true |
| V5 | true | false | true | true |
| V6 | true | true | false | true |

当前结论：

- V2 是默认推荐：相关性高、延迟最低，适合对话默认路径。
- Reranker 对 precision 有帮助，但 LLM reranker 延迟和方差较高，下一步更适合换 cross-encoder。
- Query rewrite 在部分场景提升 relevancy，但需要更稳 benchmark 和真实数据集验证。

详细数据见 [docs/evaluation.md](docs/evaluation.md)。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

当前回归结果：`118 passed`。

覆盖重点：

- RAG：chunker、retriever、pipeline、memory、coordinator、eval。
- Agent：tool registry、tools、agent loop、agent eval、EvalOps failure attribution/report。
- MCP：tool schema 转换、MCP backend、server 注册、auth header、token verifier、schema cache、audit JSONL、tool policy、audit query API。
- Safety：approval store、`send_email` pending approval、approve/reject。
- Trace：JSONL recorder、agent trace metadata、trace summary。

## API 端点

| Method | Path | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `POST` | `/index` | 索引邮件 |
| `POST` | `/index/clear` | 清空索引 |
| `GET` | `/index/status` | 索引状态 |
| `POST` | `/chat` | Coordinator 固定意图路由 |
| `POST` | `/chat/stream` | SSE 流式聊天 |
| `POST` | `/chat/graph` | LangGraph Self-RAG |
| `POST` | `/chat/agent` | Function-calling Agent |
| `DELETE` | `/chat/history` | 清空 session memory |
| `GET` | `/agent/approvals` | 查看人审审批单 |
| `POST` | `/agent/approvals/{id}/approve` | 批准高风险动作 |
| `POST` | `/agent/approvals/{id}/reject` | 拒绝高风险动作 |
| `GET` | `/agent/mcp-audit` | 查询 MCP 工具调用审计事件 |
| `POST` | `/query` | 直接 RAG 查询 |

## 目录结构

```text
api/main.py                    FastAPI 入口和审批 API
frontend/app.py                Streamlit UI
mcp_server.py                  FastMCP server
agents/
  agent_loop.py                Function-calling ReAct loop
  tool_registry.py             工具元数据单一事实源
  tool_policy.py               MCP 工具可见性策略
  evalops.py                   Agent eval 失败归因和报告生成
  tools.py                     6 个工具实现和 call_tool 护栏
  mcp_adapter.py               MCP backend/client/audit
  approvals.py                 Human-in-the-loop approval store
  tracing.py                   Agent trace JSONL recorder
  coordinator.py               LLM 意图分类和固定路由
  graph_workflow.py            LangGraph Self-RAG
core/
  pipeline.py                  标准 RAG pipeline
  retriever.py                 向量 + BM25 + RRF
  reranker.py                  LLM reranker + circuit breaker
scripts/
  run_ragas_eval.py            RAGAS-style 消融评测
  run_agent_eval.py            Agent 任务评测
  summarize_agent_traces.py    Trace 汇总
tests/                         118 个单测
docs/                          架构、评测、复盘和面经
```

## 已知限制和下一步

- 合成邮件数据不能代表真实企业邮箱分布；下一步接脱敏真实数据并标注 supporting chunks。
- `send_email` 当前是 simulated send；生产要接企业邮箱 API、权限系统、审计和撤销策略。
- `ApprovalStore` 当前是本地 JSON；生产应换 Redis/DB。
- MCP 已有 token verifier、工具级可见性策略和审计查询；生产还需要更完整的 OAuth、租户隔离、密钥轮换、部署层 TLS 和限流。
- Agent eval 已扩到 54 条元数据化任务；下一步扩到 100+ 条并接入人工复核样本。
- 检索侧下一步优先换 cross-encoder reranker，减少 LLM reranker 延迟和飘分。
