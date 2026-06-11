# Email RAG Agent

> 作者：赵伟鑫 (Yoimiya2627)<br>
> 方向：后端工程师转型 Agent / 大模型应用工程<br>
> 联系方式：a1486807398@163.com | [GitHub](https://github.com/Yoimiya2627)

一个面向邮件场景的 Agentic RAG 项目：底层是混合检索 RAG，上层是 function-calling Agent loop，外层补了 MCP-ready 工具层、人审审批、Gmail 只读同步、trace 和 EvalOps gate。

这个项目不是一个简单的聊天壳子。它的核心目标是证明：邮件 Agent 可以被工程化地检索、规划、调用工具、审批高风险动作、接入真实邮箱数据，并用可重复的评测结果约束质量。

当前定位：**production-oriented prototype**。它已经具备接近生产系统的工程骨架和验证闭环，但还不是 enterprise production-ready。

## 当前状态

本机最新验证日期：2026-06-11。

| 维度 | 当前结果 |
|---|---|
| 单元/集成测试 | `195 passed` |
| Synthetic Agent EvalOps | 105 tasks，gate PASS |
| Synthetic Agent task success | `0.8095` |
| Synthetic Agent tool accuracy | `0.8857` |
| Synthetic Agent forbidden tool violation | `0.0000` |
| Real Gmail Agent EvalOps | 30 tasks，gate PASS |
| Real Gmail Agent task success | `0.9667` |
| Real Gmail Agent tool accuracy | `1.0000` |
| Real Gmail Agent max steps reached | `0.0000` |
| Real Gmail retrieval recall | 100 gold cases，V2 recall@10 `0.9700` |
| MCP / approval / Gmail / EvalOps | 已有测试和任务入口 |

核心质量指标快照：

| 指标 | Synthetic full | Real Gmail |
|---|---:|---:|
| Agent task success | `0.8095` `[################----]` | `0.9667` `[###################-]` |
| Agent tool accuracy | `0.8857` `[##################--]` | `1.0000` `[####################]` |
| Forbidden tool violation | `0.0000` `[clear]` | `0.0000` `[clear]` |

真实邮箱正文、OAuth token、真实 eval 报告和 trace 都在 ignored 路径下，不进入 Git。

## 项目亮点

- **Agentic RAG**：向量检索 + BM25 + RRF + 可选 Cross-Encoder reranker，支持 query rewrite 和 Self-RAG。
- **Function-calling Agent Loop**：`/chat/agent` 使用 DeepSeek function calling，多轮执行 `plan -> tool_call -> observe -> re-plan`。
- **任务型 Skill Profiles**：`mail_search`、`reply_drafting`、`mail_digest` 按任务收敛工具集合和 planner 指令。
- **MCP-ready 工具层**：工具定义集中在 `agents/tool_registry.py`，同源派生 OpenAI tool schema 和 FastMCP tools/resources/prompts。
- **Human-in-the-loop 安全链路**：`send_email` 是 high-risk 工具，只创建 pending approval；审批后默认 simulated，Gmail 模式只创建 draft，不直接发送。
- **真实 Gmail 只读数据链路**：独立 `gmail.readonly` scope，把 Gmail 邮件同步到本地 JSON，再复用清洗、切分、索引、评测链路。
- **Agent EvalOps**：支持 trace、Markdown report、failure category、checkpoint/resume、offline gate。
- **生产化基础护栏**：bearer auth、CORS 配置、基础限流、tenant-aware approval/session、MCP audit、tool policy、SQLite-backed state。

## Demo

[![Demo preview](docs/demo.png)](docs/demo.mp4)

约 1 分 40 秒：邮件检索、预算查询、统计分析、evaluation 表格。点击预览图打开 MP4。

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

启动后访问：

- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- MCP server: http://127.0.0.1:8001/mcp

## 常用命令

Windows：

```powershell
.\tasks.ps1 help
.\tasks.ps1 index
.\tasks.ps1 run
.\tasks.ps1 verify
```

macOS / Linux：

```bash
make help
make index
make run
make verify
```

核心任务入口：

| 命令 | 作用 |
|---|---|
| `install` | 安装依赖并预加载 `bge-m3` |
| `index` | 把 `data/emails.json` 写入 ChromaDB |
| `run` | 同时启动 FastAPI `:8000` 和 Streamlit `:8501` |
| `mcp` | 启动 standalone MCP server `:8001` |
| `verify` | compile + tests + Agent EvalOps smoke gate |
| `agent-eval` | 跑 Agent task evaluation |
| `agent-eval-full` | 对已有 100+ synthetic eval 结果跑 strict gate |
| `gmail-preflight` | 检查真实 Gmail 本地 readiness |
| `gmail-sync-index` | Gmail 只读同步并重建真实邮箱索引 |
| `context-recall-real` | 对真实 Gmail gold 跑 retrieval recall gate |
| `gmail-agent-testset` | 由真实 Gmail gold 构建 Agent eval taskset |
| `agent-eval-real` | 跑真实 Gmail Agent EvalOps |
| `agent-eval-real-gate` | 对真实 Gmail Agent EvalOps 结果跑 offline gate |

## 系统架构

```text
Streamlit UI
   |
FastAPI
   |-- /chat         -> Coordinator -> Specialist Agents -> RAG pipeline
   |-- /chat/stream  -> worker thread -> asyncio.Queue -> SSE
   |-- /chat/graph   -> LangGraph Self-RAG
   |-- /chat/agent   -> function-calling planner -> local/MCP tool backend
   |-- /agent/approvals -> human approval for high-risk actions
   |-- /agent/mcp-audit -> MCP audit query
   |
Core RAG
   |-- clean / chunk / embed
   |-- vector search: bge-m3 + ChromaDB
   |-- BM25 lexical search
   |-- RRF fusion
   |-- metadata post-filter
   |-- optional reranker: LLM scorer or Cross-Encoder
   |-- DeepSeek generation
```

Agent 工具层：

```text
agents/tool_registry.py
   |-- openai_tool_schemas() -> agents/tools.py -> LocalToolBackend
   |-- ToolSpec metadata     -> mcp_server.py -> FastMCP

agents/agent_loop.py
   |-- AGENT_TOOL_BACKEND=local -> in-process tools
   |-- AGENT_TOOL_BACKEND=mcp   -> Streamable HTTP MCP
   |-- agents/skills.py         -> task-specific tool filtering
```

更完整的模块说明见 [docs/architecture.md](docs/architecture.md)。

## Agent 能力

| 能力 | 说明 |
|---|---|
| 邮件问答 | 在索引邮件中检索证据并回答事实问题 |
| 多步任务 | 例如 `search_emails -> get_email -> draft_reply` |
| 回信草稿 | 针对指定邮件或检索结果生成 draft |
| 人审发信 | `send_email` 只创建审批单，审批后 simulated 或 Gmail draft-only |
| 批量摘要 | 按主题检索多封邮件并汇总 |
| 统计分析 | sender top、标签分布、邮件量 |
| 多轮记忆 | tenant + session 隔离，支持 memory / SQLite backend |
| SSE 流式 | 同步 LLM SDK 通过 worker thread 桥接 FastAPI SSE |
| Self-RAG | LangGraph 状态机，低相关时 rewrite query，最多重试 2 次 |

Agent tools：

| Tool | 风险 | 人审 | 作用 |
|---|---|---:|---|
| `search_emails` | low | 否 | 混合检索邮件，支持 sender/date/labels 过滤 |
| `get_email` | low | 否 | 按 `email_id` 读取完整邮件 |
| `summarize_emails` | low | 否 | 检索并总结相关邮件 |
| `draft_reply` | medium | 否 | 起草回信，不发送 |
| `send_email` | high | 是 | 创建 pending approval |
| `email_stats` | low | 否 | 邮件统计 |

Skill profiles：

| Skill | 允许工具 | 适用场景 |
|---|---|---|
| `general` | 全部工具 | 默认通用 Agent |
| `mail_search` | `search_emails`, `get_email`, `email_stats` | 只读查询、事实问答、统计 |
| `reply_drafting` | `search_emails`, `get_email`, `draft_reply`, `send_email` | 找邮件、起草回复、提交审批 |
| `mail_digest` | `search_emails`, `summarize_emails`, `email_stats` | 主题摘要、批量汇总 |

请求示例：

```json
{
  "query": "帮我查一下 Q3 预算会议是谁发的",
  "context": {"skill": "mail_search"}
}
```

## MCP Server

启动：

```powershell
.\.venv\Scripts\python.exe mcp_server.py --transport streamable-http
```

切换 Agent 到 MCP backend：

```env
AGENT_TOOL_BACKEND=mcp
MCP_SERVER_URL=http://127.0.0.1:8001/mcp
MCP_AUTH_TOKEN=optional-shared-secret
```

MCP 暴露：

| 类型 | 内容 |
|---|---|
| Tools | 6 个 agent tools |
| Resources | `email://{email_id}`、`email-corpus://stats` |
| Prompts | `draft_reply_prompt`、`summarize_emails_prompt` |

已实现的生产化基础：

- `MCP_AUTH_TOKEN` bearer token verifier
- `MCP_TOOL_SCHEMA_CACHE_SECONDS` schema cache
- `ENABLE_MCP_AUDIT` JSONL 审计
- `MCP_ALLOWED_TOOLS` 工具白名单
- `MCP_READ_ONLY_MODE=true` 只暴露 low-risk read-only tools
- `GET /agent/mcp-audit` 查询审计事件

## Human-in-the-loop

`send_email` 不会直接发送真实邮件。Agent 调用它时只会创建审批单：

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
| `POST` | `/agent/approvals/{approval_id}/approve` | 批准；默认 simulated，Gmail 模式创建 draft |
| `POST` | `/agent/approvals/{approval_id}/reject` | 拒绝 |

审批存储支持 `json` 和 `sqlite` backend，并按 `tenant_id` 隔离。Gmail 模式使用 `gmail.compose` scope，只调用 `users.drafts.create` 创建草稿，保持 `sent=false`。

## Gmail 真实数据链路

真实 Gmail 接入分成两条权限链路：

| 链路 | Scope | 用途 |
|---|---|---|
| Draft-only | `https://www.googleapis.com/auth/gmail.compose` | 审批后创建 Gmail draft |
| Read-only | `https://www.googleapis.com/auth/gmail.readonly` | 同步真实邮件到本地 ignored JSON |

典型流程：

```powershell
.\tasks.ps1 gmail-preflight
.\tasks.ps1 gmail-sync-index
.\tasks.ps1 gmail-gold-template
.\tasks.ps1 gmail-gold-quality
.\tasks.ps1 context-recall-real
.\tasks.ps1 gmail-agent-testset
.\tasks.ps1 agent-eval-real
.\tasks.ps1 agent-eval-real-gate
```

注意：

- `data/real_emails/`、`data/mail_sync/`、真实 eval report 和 trace 都已 gitignore。
- `gmail-preflight` 只做本地 readiness 检查，不触网。
- `gmail-sync-index` 会让 Chroma 索引指向真实 Gmail corpus。
- 如果之后要跑 synthetic full eval，需要先重新索引默认 `data/emails.json`。
- `gmail-agent-testset` 会过滤 fragment、boilerplate、隐藏字符和重复 source email，避免低质量 chunk 污染 Agent EvalOps。

## EvalOps

Agent eval 输出：

- `task_success_rate`
- `tool_accuracy`
- `avg_steps`
- `max_steps_reached_rate`
- `forbidden_tool_violation_rate`
- 每条 case 的 `trace_id`
- failure category，例如 `missing_expected_tool`、`forbidden_tool`、`tool_error`、`approval_required`、`max_steps`

Synthetic full eval：

```powershell
$env:ENABLE_AGENT_TRACE='true'
$env:AGENT_TRACE_LOG_PATH='data/traces/agent_traces_full.jsonl'
.\.venv\Scripts\python.exe scripts\run_agent_eval.py `
  --output data/eval_results/agent_eval.json `
  --report-output data/eval_results/agent_eval_report.md `
  --trace-input data/traces/agent_traces_full.jsonl

.\tasks.ps1 agent-eval-full
```

Real Gmail agent eval：

```powershell
.\tasks.ps1 gmail-agent-testset
.\tasks.ps1 agent-eval-real
.\tasks.ps1 agent-eval-real-gate
```

Real Gmail eval 前必须保证 Chroma 索引和 real taskset 使用同一份真实 Gmail corpus。

## 评测结果

### Agent EvalOps

| Dataset | Tasks | Success | Tool accuracy | Avg steps | Max steps | Forbidden tool | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Synthetic full | 105 | 0.8095 | 0.8857 | 2.68 | 0.0190 | 0.0000 | PASS |
| Real Gmail | 30 | 0.9667 | 1.0000 | 2.47 | 0.0000 | 0.0000 | PASS |

可视化对比：

```text
Task success
Synthetic full  0.8095 | ################----
Real Gmail      0.9667 | ###################-

Tool accuracy
Synthetic full  0.8857 | ##################--
Real Gmail      1.0000 | ####################
```

### Retrieval Recall

| Dataset | Version | Cases | Retrieval window | mean_context_recall | hit_rate | perfect |
|---|---|---:|---|---:|---:|---:|
| Synthetic gold chunks | V2 | 30 | top 5 | 0.6167 | 0.6333 | 0.6000 |
| Synthetic gold chunks | V7 | 30 | top 5 | 0.8000 | 0.8000 | 0.8000 |
| Real Gmail gold | V2 | 100 | top 10 / fetch 80 | 0.9700 | 0.9700 | 0.9700 |

```text
Mean context recall
Synthetic V2     0.6167 | ############--------
Synthetic V7     0.8000 | ################----
Real Gmail V2    0.9700 | ###################-
```

### RAG Ablation

| Version | BM25 | RRF | Reranker | Backend | Rewrite | answer_relevancy | faithfulness | context_precision |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| V1 | false | false | false | llm | false | 0.8667 | 0.9233 | 0.5937 |
| V2 | true | true | false | llm | false | 0.9567 | 0.9000 | 0.5713 |
| V3 | true | true | true | llm | false | 0.9333 | 0.9017 | 0.7147 |
| V4 | true | true | true | llm | true | 0.9533 | 0.8783 | 0.6427 |
| V5 | true | false | true | llm | true | 0.9467 | 0.9083 | 0.6147 |
| V6 | true | true | false | llm | true | 0.9600 | 0.8967 | 0.6050 |
| V7 | true | true | true | cross_encoder | false | 0.9750 | 0.9267 | 0.6103 |

指标领先项：

| 指标 | 当前最佳 | 结果 | 解释 |
|---|---|---:|---|
| answer_relevancy | V7 | 0.9750 | Cross-Encoder reranker 的质量优先路径表现最好 |
| faithfulness | V7 | 0.9267 | 回答和检索上下文一致性最高 |
| context_precision | V3 | 0.7147 | LLM reranker precision 最高，但引入额外 LLM 延迟和方差 |

当前 serving policy：

- 默认对话路径选 V2：组件少、延迟低、稳定。
- 质量优先选 V7：Cross-Encoder 提升 recall，避免额外 LLM scorer 方差。
- V3 作为高 precision 对照：效果好，但引入额外 LLM 延迟和成本。

详细评测见 [docs/evaluation.md](docs/evaluation.md)。

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
| `GET` | `/agent/mcp-audit` | 查询 MCP 工具调用审计 |
| `POST` | `/query` | 直接 RAG 查询 |

## 配置

唯一必填项：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

常用配置：

| 配置 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | 普通 RAG 生成、重排和打分 |
| `AGENT_PLANNER_MODEL` | `deepseek-chat` | Agent planner |
| `EMAIL_DATA_PATH` | `./data/emails.json` | 默认 synthetic 邮件数据 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 本地 embedding |
| `RERANKER_BACKEND` | `cross_encoder` | `cross_encoder` 或 `llm` |
| `ENABLE_BM25` | `true` | 开启 BM25 |
| `ENABLE_RRF` | `true` | 开启 RRF |
| `ENABLE_RERANKER` | `false` | 默认不走 reranker |
| `ENABLE_QUERY_REWRITE` | `false` | 默认不做 query rewrite |
| `AGENT_TOOL_BACKEND` | `local` | `local` 或 `mcp` |
| `AGENT_MAX_STEPS` | `6` | Agent 最大工具轮数 |
| `AGENT_MAX_REPEAT` | `2` | 重复工具调用 guard |
| `API_AUTH_TOKEN` | 空 | 非空时保护 API 端点 |
| `RATE_LIMIT_ENABLED` | `false` | 基础固定窗口限流 |
| `APPROVAL_STORE_BACKEND` | `json` | `json` 或 `sqlite` |
| `SESSION_STORE_BACKEND` | `memory` | `memory` 或 `sqlite` |
| `MAIL_PROVIDER` | `simulated` | `simulated` 或 `gmail` draft-only |
| `MCP_AUTH_TOKEN` | 空 | MCP bearer token |
| `MCP_READ_ONLY_MODE` | `false` | MCP 只读工具模式 |
| `ENABLE_AGENT_TRACE` | `false` | Agent trace JSONL |

完整配置见 [.env.example](.env.example)。

## 测试与验证

```powershell
.\tasks.ps1 test
.\tasks.ps1 verify
.\tasks.ps1 agent-eval-full
.\tasks.ps1 agent-eval-real-gate
```

`verify` 包含：

1. Python compile check
2. 全量 pytest
3. Agent EvalOps smoke gate

当前测试覆盖：

- RAG：chunker、retriever、pipeline、reranker、context recall、generation budget。
- Agent：tool registry、skills、tools、agent loop、EvalOps、failure attribution、gate。
- MCP：schema 转换、server 注册、auth、schema cache、audit、tool policy。
- Safety / Mail：approval store、pending approval、approve/reject、Gmail draft provider、Gmail read-only sync。
- API / State：auth、CORS、rate limit、tenant-aware approval/session、SQLite session store。

## 目录结构

```text
api/main.py                    FastAPI 入口、auth/rate-limit、审批 API
frontend/app.py                Streamlit UI
mcp_server.py                  FastMCP server
agents/
  agent_loop.py                Function-calling ReAct loop
  skills.py                    Agent Skill profiles
  tool_registry.py             工具元数据单一事实源
  tools.py                     6 个工具实现和 call_tool 护栏
  mcp_adapter.py               MCP backend/client/audit
  approvals.py                 Human-in-the-loop approval store
  mail_providers.py            Simulated/Gmail draft-only executor
  gmail_readonly.py            Gmail read-only provider
  evalops.py                   Agent eval 失败归因和报告生成
  tracing.py                   Agent trace JSONL recorder
  graph_workflow.py            LangGraph Self-RAG
core/
  pipeline.py                  RAG pipeline
  retriever.py                 向量 + BM25 + RRF
  reranker.py                  LLM / Cross-Encoder reranker
  reranker_policy.py           V2/V3/V7 serving policy
  session_store.py             memory / SQLite session backend
scripts/
  index_emails.py              索引邮件
  run_agent_eval.py            Agent task eval
  check_agent_eval_gate.py     Offline Agent EvalOps gate
  build_real_agent_testset.py  真实 Gmail gold -> Agent taskset
  sync_gmail_readonly.py       Gmail read-only sync/index
  evaluate_context_recall.py   Gold chunk recall eval
  check_real_gold_quality.py   真实 gold quality gate
tests/                         pytest 回归测试
docs/                          架构、评测、计划和复盘
```

## 生产边界

已经具备：

- Gmail read-only sync、Gmail draft-only provider、真实 Gmail retrieval/Agent eval。
- Human approval、tool risk 分级、forbidden tool gate。
- MCP bearer token、tool policy、audit query。
- API bearer token、CORS 配置、基础 rate limit。
- SQLite-backed approval/session 选项。
- Trace、report、offline gate、checkpoint/resume。

仍需补强：

- 企业级 OAuth/RBAC、密钥轮换、审计留存策略。
- Redis/Postgres-backed approval/session store，迁移、备份、连接池。
- 多 worker / 多实例共享状态和分布式限流。
- 真实发送邮件的撤销、幂等、风控和合规模块。
- 更大规模真实邮箱 gold set、人工复核、趋势看板。
- 部署层 TLS、observability、告警和成本监控。

## 面试可讲的工程价值

这个项目适合作为 Agent 转型项目展示，重点可以讲：

- 不只是 RAG demo，而是把 Agent 工具调用、权限、人审、MCP、真实数据和 EvalOps 串成闭环。
- 有明确的风险控制：高风险动作不会直接执行，必须 approval。
- 有可复现质量证据：synthetic 105 tasks、real Gmail 30 tasks、real retrieval 100 gold cases。
- 有生产化意识：auth、rate limit、tenant isolation、SQLite state、audit、trace、gate 都有工程落点。
- 对边界有清醒判断：当前是 production-oriented prototype，不把它包装成 enterprise production-ready。
