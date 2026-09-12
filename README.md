# Email RAG Agent

2026-09-12 新增 163 邮箱历史分批回补、后台定时增量同步及独立本地全文搜索。真实邮件不自动进入模型问答索引；使用与本轮实测范围见 [163 后台同步与验证说明](docs/163-background-sync-validation-2026-09-12.md)。

> **Author**: 赵伟鑫 (Yoimiya2627) — Agent 开发工程师 / 大模型应用开发工程师
> **Contact**: a1486807398@163.com | [GitHub](https://github.com/Yoimiya2627)

2026-09-11 已修复 Gmail 审批账号/授权绑定、完整原件版本重放、草稿附件覆盖提示，以及未完成模型结果误作检索条件四处问题。兼容变化与回家后的验证步骤见 [修复与验收说明](docs/bugfix-2026-09-11.md)。当前离线验证仍有 Chroma/gRPC 原生 DLL 环境阻碍，尚未完成真实模型和邮箱验收。

2026-09-10 上下文开发已加入中文/编号历史检索、命中窗口、任务与约束版本、可选语义摘要/候选记忆、工具结果分页、闭合交互引用压缩及 checkpoint v2。使用、开关、迁移和验收边界见 [上下文优化说明](docs/context-development-2026-09-10.md)。语义摘要与候选抽取默认关闭，确定性回归不代表真实模型质量成绩。

一个面向邮件场景的 Agentic RAG 系统。底层是向量检索 + BM25 + RRF + 可选 Cross-Encoder reranker 的混合检索 RAG；上层是 DeepSeek 原生 function calling 的 ReAct-style agent loop；工具层已经升级为 MCP-ready backend，支持 FastMCP tools/resources/prompts、可选 MCP 鉴权、工具级权限策略、审计查询、人审审批、Gmail draft-only provider、Gmail read-only 增量同步和 EvalOps trace/report/gate。

2026-09-09 优化版已补齐检索约束、运行预算、事务审批、API 身份和评测一致性检查。迁移、兼容变化和验证范围见 [本次优化说明](docs/optimization-2026-09-09.md)。下方历史模型成绩不代表本次修改后的实测成绩。

后续 Astra 复审发现的 MCP、同步持久化、邮件正文保真、历史预算、索引预检与评分校验问题已修复。新旧数据和评测格式的迁移要求见 [BUG 修复与兼容说明](docs/bugfix-2026-09-09.md)。

当前个人单机部署采用一个 API worker、持久会话/后台任务和审批结果核对。先读 [部署指南](docs/personal-deployment.md)、[状态备份与留存](docs/state-retention-and-recovery.md) 和 [离线验证说明](docs/offline-validation.md)。容器构建与依赖安装默认不下载模型权重；实际 Docker 启动、托管 CI 和真实模型质量需要在目标环境验收。

2026-09-10 的会话恢复、证据回读、后台任务与迁移操作见 [个人使用与接口说明](docs/personal-use-2026-09-10.md)。本轮测试记录与验收边界单独保存，下面的旧演示和模型分数均为历史结果。

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
- [Gmail Read-only Sync](#gmail-read-only-sync)
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
- **Human-in-the-loop 安全链路**：高风险 `send_email` 只创建 pending approval；审批通过后默认 simulated，配置 `MAIL_PROVIDER=gmail` 时只创建 Gmail draft，不直接发送。
- **真实邮箱 read-only 接入**：Gmail read-only provider 使用独立只读 OAuth scope，把真实邮件增量同步到本地忽略 JSON，再复用清洗、切分、embedding 索引链路。
- **Agent EvalOps**：105 条 agent 任务集覆盖多步、异常、歧义、权限、高风险发信和 Gmail draft；eval record 关联 `trace_id`，可输出失败归因、Markdown 报告和 CI gate。
- **RAG 消融评测**：7 版 RAGAS-style 对比，量化 BM25、RRF、LLM reranker、Cross-Encoder reranker、query rewrite 的 ROI；新增 synthetic gold chunk baseline 和 `context_recall` 确定性评测，历史 V7 在 30 题上达到 `mean_context_recall=0.8000`。
- **工程护栏**：max steps、重复工具调用检测、坏 JSON 降级、参数校验、工具异常回灌、工具输出截断、rerank 输入截断、生成上下文预算。
- **离线回归**：pytest 用例与可独立执行的 unittest 覆盖检索、API、审批事务、MCP、运行预算、证据来源和评测门禁；具体执行范围见本次优化说明。

## Demo

[![Demo preview](docs/demo.png)](docs/demo.mp4)

约 1 分 40 秒：邮件检索、预算查询、统计分析、evaluation 表格。点击预览图打开 MP4。

## 功能能力

- 邮件问答：从当前兼容索引中检索并回答；历史演示使用 5000 封样本，不代表当前机器已完成该容量验收。
- 多步 Agent：例如“找出报销邮件并帮我起草回复”，agent 会自主 `search_emails -> get_email -> draft_reply`。
- 人审发信：`send_email` 创建审批单，必须人类确认后才执行；默认 simulated，Gmail 模式只创建 draft。
- 批量摘要：按主题检索多封邮件并生成结构化摘要。
- 回信草稿：针对检索到的邮件或指定 `email_id` 起草回复。
- 统计分析：发件人 Top、标签分布、每日邮件量。
- 多轮记忆：会话记录持久化，默认给模型提供最近 30 轮成功上下文，另有字符/token与缓存预算；历史回查不代表原邮件证据。
- SSE 流式：worker 线程 + 有界 `queue.Queue` 桥接同步 LLM SDK 与 FastAPI SSE，包含来源与错误事件。
- Self-RAG：LangGraph 状态机，检索结果不相关时 rewrite query，最多重试 2 次。
- Agent EvalOps：`data/agent_testset.json` 维护 105 条带类型/风险/期望工具/禁用工具/成功标准的任务，eval 可生成 Markdown 报告并通过 gate 脚本做阈值检查；离线 gate 的最低任务数门槛仍是 `>=100`。

## 快速开始

### Windows PowerShell

Windows 的 Chroma 索引完整路径必须仅含 ASCII 字符。项目在中文目录时，请在 `.env` 中设置 `CHROMA_PERSIST_DIR=E:/email-agent-runtime/chroma_db`，再运行索引和服务。

```powershell
copy .env.example .env
# 编辑 .env，填 DEEPSEEK_API_KEY
python -m venv .venv
.\tasks.ps1 install
# 可选：显式准备配置的嵌入模型（可能下载权重）
.\tasks.ps1 preload
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
python3 -m venv .venv
make install
# 可选：显式准备配置的嵌入模型
make preload
make index
make run
```

打开：

本机启动脚本将前端固定在 `127.0.0.1:8501`，不会自动打开浏览器；端口被占用时启动失败。前端使用本机 API 身份访问邮件和审批，请保持仅本机可访问。

- Frontend: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- MCP server: http://127.0.0.1:8001/mcp

## 配置说明

本机模型调用需填写 `DEEPSEEK_API_KEY`；Docker、反向代理或远程访问还需设置 `API_AUTH_TOKEN`。完整配置见 [`.env.example`](.env.example)。Self-RAG 模式另需安装 `requirements-graph.txt`。

| 配置 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | 普通 RAG 生成、重排和打分模型 |
| `DEEPSEEK_THINKING_MODE` | `disabled` | `enabled` / `disabled`；仅向官方 DeepSeek 的 V4 Flash/Pro 请求注入，调用方显式 `extra_body.thinking` 优先 |
| `AGENT_PLANNER_MODEL` | `deepseek-chat` | Agent planner，function calling 更轻更快 |
| `RERANKER_BACKEND` | `cross_encoder` | `cross_encoder` 使用本地 Cross-Encoder；`llm` 兼容旧重排 |
| `CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-Encoder reranker 模型 |
| `CROSS_ENCODER_DEVICE` | `cpu` | Cross-Encoder 推理设备；有 CUDA 可改为 `cuda` |
| `CROSS_ENCODER_MAX_LENGTH` | `512` | Cross-Encoder 单对输入最大长度 |
| `RERANK_INPUT_CHAR_LIMIT` | `1200` | 送入 reranker 的单 chunk 字符上限，返回结果仍保留原文 |
| `GENERATION_CONTEXT_CHAR_LIMIT` | `6000` | 拼接到生成 prompt 的上下文总字符预算 |
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
| `APPROVAL_STORE_PATH` | `./data/approvals/pending_actions.json` | 旧 JSON 只读导入旁边的 SQLite；也可直接指定 .sqlite3 |
| `MAIL_PROVIDER` | `simulated` | 审批通过后的邮件执行 provider；`simulated` 或 `gmail` |
| `GMAIL_CREDENTIALS_PATH` | `./credentials/gmail_credentials.json` | Gmail OAuth client secret 文件 |
| `GMAIL_TOKEN_PATH` | `./credentials/gmail_token.json` | Gmail OAuth token 缓存 |
| `GMAIL_SCOPES` | `https://www.googleapis.com/auth/gmail.compose` | Gmail draft-only scope |
| `GMAIL_USER_ID` | `me` | Gmail API user id |
| `ENABLE_REAL_EMAIL_SEND` | `false` | 保留开关；当前实现不默认真实发送 |
| `GMAIL_READONLY_TOKEN_PATH` | `./credentials/gmail_readonly_token.json` | Gmail read-only OAuth token 缓存 |
| `GMAIL_READONLY_SCOPES` | `https://www.googleapis.com/auth/gmail.readonly` | Gmail read-only sync scope，和 draft scope 分离 |
| `GMAIL_SYNC_QUERY` | `newer_than:30d` | Gmail 同步查询条件 |
| `GMAIL_SYNC_MAX_RESULTS` | `100` | 单次 read-only 同步最多拉取消息数 |
| `GMAIL_SYNC_OUTPUT_PATH` | `./data/real_emails/gmail_emails.json` | 同步后的本地 JSON；已 gitignore |
| `GMAIL_SYNC_STATE_PATH` | `./data/mail_sync/gmail_sync_state.json` | 增量同步状态；已 gitignore |
| `ENABLE_AGENT_TRACE` | `false` | 是否记录 agent trace JSONL |
| `AGENT_TRACE_LOG_PATH` | `./data/traces/agent_traces.jsonl` | trace 输出路径 |

Feature flags 默认是 V2 推荐配置：`BM25=true, RRF=true, RERANKER=false, REWRITE=false`。

| Flag | 默认 | 说明 |
|---|---:|---|
| `ENABLE_BM25` | true | 向量 + BM25 混检 |
| `ENABLE_RRF` | true | Reciprocal Rank Fusion |
| `ENABLE_RERANKER` | false | 可选 reranker；由 `RERANKER_BACKEND` 决定 LLM 或 Cross-Encoder |
| `ENABLE_QUERY_REWRITE` | false | LLM query rewrite |

## 系统架构

```text
Streamlit UI
   |
FastAPI
   |-- /chat         -> Coordinator -> Specialist Agents -> core.pipeline.retrieve()
   |-- /chat/stream  -> Coordinator -> worker thread -> bounded queue.Queue -> SSE
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
   |-- optional reranker: LLM scorer or Cross-Encoder
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
| `get_email` | low | 否 | 按 `email_id` 分页读取邮件，保留来源版本和可见范围 |
| `get_thread` | low | 否 | 分页查看线程证据与成员邮件 |
| `summarize_emails` | low | 否 | 检索并总结相关邮件 |
| `draft_reply` | medium | 否 | 起草回信，不发送 |
| `send_email` | high | 是 | 只创建 pending approval；审批通过后 simulated 或 Gmail draft-only |
| `email_stats` | low | 否 | 邮件统计聚合 |

`call_tool()` 会拒绝不符合 schema 的参数、检查必填字段并返回结构化工具结果；失败、预算不足和不确定执行状态分别记录，不能被当成已完成的成功结果。

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
| Tools | 与共享 registry 同源的 agent tools（含线程分页读取） |
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
| `POST` | `/agent/approvals/{approval_id}/approve` | 人工批准；默认 simulated，`MAIL_PROVIDER=gmail` 时创建 Gmail draft |
| `POST` | `/agent/approvals/{approval_id}/reject` | 人工拒绝 |

当前实现使用 SQLite 事务；旧 JSON 只读迁移到旁边的 SQLite。`ApprovalStore.approve()` 默认返回 simulated result；`MAIL_PROVIDER=gmail` 时通过 Gmail API `users.drafts.create` 创建草稿，返回 `draft_id/message_id`，并保持 `sent=false`。先运行 `python -m agents.mail_providers authorize` 完成本机授权；审批请求不会打开浏览器或自动刷新 token。unknown/executing 通过人工 reconcile 结案，不自动重试。真实发送未实现。

Gmail draft-only 配置示例：

```env
MAIL_PROVIDER=gmail
GMAIL_CREDENTIALS_PATH=./credentials/gmail_credentials.json
GMAIL_TOKEN_PATH=./credentials/gmail_token.json
GMAIL_SCOPES=https://www.googleapis.com/auth/gmail.compose
GMAIL_USER_ID=me
ENABLE_REAL_EMAIL_SEND=false
```

## Gmail Read-only Sync

Phase 9 新增 Gmail 只读同步，不复用 draft-only scope，也不请求发送权限。同步结果写到 `data/real_emails/`，增量状态写到 `data/mail_sync/`，两个目录都已加入 `.gitignore`，避免真实邮箱内容入库。

```env
GMAIL_CREDENTIALS_PATH=./credentials/gmail_credentials.json
GMAIL_READONLY_TOKEN_PATH=./credentials/gmail_readonly_token.json
GMAIL_READONLY_SCOPES=https://www.googleapis.com/auth/gmail.readonly
GMAIL_SYNC_QUERY=newer_than:30d
GMAIL_SYNC_MAX_RESULTS=100
```

```powershell
.\.venv\Scripts\python.exe scripts\sync_gmail_readonly.py
.\.venv\Scripts\python.exe scripts\sync_gmail_readonly.py --index
```

同步 CLI 默认核对账号与授权范围，分批回补当前 query 的邮件，复核已知邮件的标签/范围变化。只有当前范围完整列举且无未决失败后才移除范围外邮件；404 单条删除不会卡住队列。`--max-results` 是单次新增处理预算，可重复运行续接；`--query` 仍限定邮箱范围。`--index` 按完整规范语料快照发布新索引代，使删除传播到当前查询。旧未绑定账号的输出需显式迁移到新输出，或明确选择兼容 `--append-only`；后者不提供完整核对语义。原件按版本保留，可用 `scripts/replay_gmail_captures.py` 离线重解析。

## Trace 与 Eval

开启 trace：

```env
ENABLE_AGENT_TRACE=true
AGENT_TRACE_LOG_PATH=./data/traces/agent_traces.jsonl
```

每次 agent run 会记录：

- `agent_start`
- `tool_call`：tool、status、latency_ms 等运行字段，不记录原始参数、正文或自由错误文本
- `agent_end`：status、steps、answer_chars

汇总 trace：

```powershell
.\.venv\Scripts\python.exe scripts\summarize_agent_traces.py --json
```

Agent eval：

```powershell
.\.venv\Scripts\python.exe scripts\run_agent_eval.py --limit 8 --report-output data/eval_results/agent_eval_report.md
```

`scripts/run_agent_eval.py` 会输出 task_success_rate、tool_accuracy、avg_steps、max_steps_reached_rate、forbidden_tool_violation_rate，并把每条记录关联 `trace_id`。任务集现在是 105 条元数据化 case（gate 门槛 `>=100`），每条包含 `id`、`task_type`、`risk_level`、`expected_tools`、`forbidden_tools`、`success_criteria`；报告会给出 `failure_category`，用于区分 missing_expected_tool、forbidden_tool、tool_error、approval_required、max_steps 等问题。

离线 gate 不调用 LLM；从非空、唯一 ID 的 records 重算指标，校验执行状态、工具列表与源码/任务集指纹。历史报告需在当前版本重新生成；仅手改 summary 不能通过。默认成功率/期望工具覆盖率至少 80%，禁用工具违规必须为 0，预算耗尽不超过 5%，任务数至少 100。

```powershell
.\.venv\Scripts\python.exe scripts\check_agent_eval_gate.py `
  --input data/eval_results/agent_eval.json `
  --min-tasks 100 `
  --min-task-success-rate 0.80 `
  --min-tool-accuracy 0.80 `
  --max-forbidden-tool-violation-rate 0.00 `
  --max-max-steps-reached-rate 0.05
```

## 评测结果

RAG 消融脚本：

```powershell
.\.venv\Scripts\python.exe scripts\run_ragas_eval.py --versions V2
.\.venv\Scripts\python.exe scripts\run_ragas_eval.py
.\.venv\Scripts\python.exe scripts\measure_latency.py --limit 10
.\.venv\Scripts\python.exe scripts\measure_reranker_latency.py --versions V2,V7
.\.venv\Scripts\python.exe scripts\evaluate_context_recall.py --init-template --gold data\gold_chunks.json --limit 30
.\.venv\Scripts\python.exe scripts\evaluate_context_recall.py --gold data\gold_chunks.json --versions V2,V7
```

7 个版本：

| Version | BM25 | RRF | Reranker | Backend | Rewrite | answer_relevancy | faithfulness | context_precision |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| V1 | false | false | false | llm | false | 0.8667 | 0.9233 | 0.5937 |
| V2 | true | true | false | llm | false | 0.9567 | 0.9000 | 0.5713 |
| V3 | true | true | true | llm | false | 0.9333 | 0.9017 | **0.7147** |
| V4 | true | true | true | llm | true | 0.9533 | 0.8783 | 0.6427 |
| V5 | true | false | true | llm | true | 0.9467 | 0.9083 | 0.6147 |
| V6 | true | true | false | llm | true | 0.9600 | 0.8967 | 0.6050 |
| V7 | true | true | true | cross_encoder | false | **0.9750** | **0.9267** | 0.6103 |

确定性 context_recall（`data/gold_chunks.json`，前 30 题 synthetic gold chunks）：

| Version | n | mean_context_recall | chunk_hit_rate | perfect_recall_rate |
|---|---:|---:|---:|---:|
| V2 | 30 | 0.6167 | 0.6333 | 0.6000 |
| V7 | 30 | **0.8000** | **0.8000** | **0.8000** |

当前结论：

- V2 仍适合作为默认对话路径：组件少、延迟低，relevancy 处于第一梯队。
- V7 验证了 Cross-Encoder 的工程价值：不再把 rerank 变成一次额外 LLM 评分调用，本次 30 题实测 relevancy / faithfulness 最高，precision 比 V2 有提升但低于旧 LLM V3；在同一批 gold chunks 上，`mean_context_recall` 从 V2 的 0.6167 提升到 0.8000。
- V3 的旧 LLM reranker 仍拿到最高 context_precision，但代价是额外 LLM 延迟和评分方差；生产路径更适合作为可选高精度模式，而不是默认模式。
- Phase 8 已补 `core.reranker_policy` 和 `scripts/measure_reranker_latency.py`：默认对话仍选 V2；质量优先选 V7；需要高 precision 且允许额外 LLM scorer 时才把 V3 当对照。
- Phase 10 已补 `scripts/evaluate_context_recall.py`、`data/gold_chunks.json` 和 `data/eval_results/context_recall.json`：可从 RAGAS testset 生成人工 gold chunk 模板，并用 `gold_chunk_ids` 对 V2/V7 等版本计算确定性的 `context_recall`。
- Query rewrite 在部分场景提升 relevancy，但需要更稳 benchmark 和真实数据集验证。

详细数据见 [docs/evaluation.md](docs/evaluation.md)。

## 测试

```powershell
.\.venv\Scripts\python.exe scripts/offline_tests.py tests/ -q
```

旧版记录的 `146 passed` 属于历史结果。当前离线 runner 禁用网络与 dotenv、隔离临时状态并阻止源码目录写入；依赖快照和 CI 验证范围见 [离线验证说明](docs/offline-validation.md)。本地通过的测试不代表 Docker 或托管 CI 已成功运行。HTTP 延迟/TTFT、并发与 24 小时验收步骤见 [服务测量指南](docs/service-performance-experiments.md)。

覆盖重点：

- RAG：chunker、retriever、pipeline、Cross-Encoder reranker、reranker serving policy、gold chunk context_recall、generation context budget、memory、coordinator、eval。
- Agent：tool registry、tools、agent loop、agent eval、EvalOps failure attribution/report/gate。
- MCP：tool schema 转换、MCP backend、server 注册、auth header、token verifier、schema cache、audit JSONL、tool policy、audit query API。
- Safety / Mail：approval store、`send_email` pending approval、approve/reject、Gmail draft provider、Gmail read-only sync。
- Trace：JSONL recorder、agent trace metadata、trace summary。

## API 端点

| Method | Path | 说明 |
|---|---|---|
| `GET` | `/health` | 进程存活 |
| `GET` / `POST` | `/ready` / `/warmup` | 就绪状态 / 显式服务内预热 |
| `POST` | `/index` | 索引邮件 |
| `POST` | `/index/clear` | 清空索引 |
| `GET` | `/index/status` | 索引状态 |
| `POST` | `/chat` | Coordinator 固定意图路由 |
| `POST` | `/chat/stream` | SSE 流式聊天 |
| `POST` | `/chat/graph` | LangGraph Self-RAG |
| `POST` | `/chat/agent` | Function-calling Agent |
| `GET` / `DELETE` | `/chat/history` | 分页读取 / 删除完整会话记录 |
| `GET` | `/chat/sessions` | 会话列表分页 |
| `GET` | `/chat/history/search`、`/chat/history/summary` | 历史搜索与带轮次摘录 |
| `GET` / `POST` | `/chat/facts` | 读取 / 记录有来源与版本的显式任务约束 |
| `GET` | `/chat/evidence` | 持久引用范围 |
| `GET` | `/evidence/email`、`/evidence/thread` | 分页回读索引正文与线程材料 |
| `POST` | `/evidence/reread` | 校验指定版本、片段与可见范围 |
| `POST` | `/jobs/agent`、`/jobs/index` | 建立有界后台任务 |
| `GET` | `/jobs`、`/jobs/{id}`、`/jobs/operations/{key}` | 分页、状态与提交结果核对 |
| `POST` | `/jobs/{id}/cancel`、`/jobs/{id}/resume` | 取消后续调用 / 显式安全恢复 |
| `GET` | `/agent/approvals` | 查看人审审批单 |
| `POST` | `/agent/approvals/{id}/approve` | 批准高风险动作 |
| `POST` | `/agent/approvals/{id}/reject` | 拒绝高风险动作 |
| `POST` | `/agent/approvals/{id}/reconcile` | 根据人工核对结果结案未知动作 |
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
  tools.py                     共享工具实现和 call_tool 护栏
  mcp_adapter.py               MCP backend/client/audit
  approvals.py                 Human-in-the-loop approval store
  mail_providers.py            Simulated/Gmail draft-only approval executor
  gmail_readonly.py            Gmail read-only provider + MIME -> Email schema
  tracing.py                   Agent trace JSONL recorder
  coordinator.py               LLM 意图分类和固定路由
  graph_workflow.py            LangGraph Self-RAG
core/
  pipeline.py                  标准 RAG pipeline
  reranker_policy.py           Reranker serving policy：V2/V3/V7 业务选型
  retriever.py                 向量 + BM25 + RRF
  reranker.py                  LLM / Cross-Encoder reranker + circuit breaker
scripts/
  run_ragas_eval.py            RAGAS-style 消融评测
  measure_latency.py           端到端 RAG latency benchmark
  measure_reranker_latency.py  Reranker-only latency benchmark
  evaluate_context_recall.py   Gold chunk 模板和 context_recall 评测
  sync_gmail_readonly.py       Gmail read-only 增量同步到本地 JSON/索引
  run_agent_eval.py            Agent 任务评测
  check_agent_eval_gate.py     Agent EvalOps 离线阈值 gate
  summarize_agent_traces.py    Trace 汇总
tests/                         pytest 与可独立运行的 unittest 回归
docs/                          架构、评测、复盘；docs/面经 为本地忽略目录
```

## 已知限制和下一步

- Gmail read-only provider 已能把真实邮件增量同步到本地忽略 JSON；下一步是在个人/脱敏邮箱上实际跑同步、清洗质量抽检，并标注 supporting chunks。
- `send_email` 已支持审批后 simulated 或 Gmail draft-only；真实发送、撤销策略和企业邮箱多用户授权仍是后续工作。
- `ApprovalStore` 已使用 SQLite 事务；执行结果为 unknown/executing 时必须人工核对，不自动重放远端动作。会话与任务检查点持久化，内存缓存仍受容量与 TTL 约束；仅支持单 owner、单实例、单 worker。
- MCP 已有 token verifier、工具级可见性策略和审计查询；生产还需要更完整的 OAuth、租户隔离、密钥轮换、部署层 TLS 和限流。
- Agent eval 已扩到 105 条元数据化任务并支持离线 gate；下一步接入真实失败样本、人工复核和历史趋势对比。
- 检索侧已接入 Cross-Encoder reranker，并补了 rerank-only benchmark harness、serving policy 和 synthetic gold context_recall baseline；下一步是在真实模型环境跑 30+ 题 × 多轮 latency、加 batch 推理，并基于真实邮箱 gold chunk 产出正式 recall 数据。
