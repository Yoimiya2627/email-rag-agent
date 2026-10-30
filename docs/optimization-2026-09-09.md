# 2026-09-09 优化与迁移说明

本轮按 code-review skill 实施，由三个 Astra 子 agent 分工修改并交叉审查，主 agent 负责 API、评测、部署和最终整合。只修改个人项目；公司项目作为设计参考。没有安装缺失依赖、下载模型、连接真实邮箱或调用收费模型。

## 1. 修改目标与结果

优先修复会影响答案正确性、执行可靠性和数据边界的具体问题。保留个人项目的 RAG、固定路由、Self-RAG、function calling 和 MCP 结构。

| 领域 | 原问题 | 当前行为 | 主要代码 |
|---|---|---|---|
| 检索约束 | 筛选为空可能回退到未筛选结果 | 发件人、日期、标签是硬条件；无匹配返回空；条件解析失败明确报错 | `core/pipeline.py` |
| 日期 | 字符串/模糊时间判断不完整 | 按固定时区日历区间计算今天、昨天、本周、上周、本月、上月、今年和明确日期范围 | `core/pipeline.py` |
| 调用一致性 | 摘要、草稿、API 的检索参数语义不同 | 共用 `retrieve(query, *, filters=None, top_n=None, fetch_k=None, history=None)` | `core/pipeline.py`、各专家 agent、`api/main.py` |
| 草稿定位 | 邮件定位问题与写作指令混杂 | 显式区分 query / instruction；已知 email_id 直接读取索引邮件 | `agents/writer_agent.py`、`agents/tools.py` |
| Self-RAG | 评分失败/全否决后仍采用未经认可的候选 | 全否决保持无证据；关闭 rewrite 时不重复同一检索 | `agents/graph_workflow.py` |
| 邮件读取 | 读取整库再找 ID，重叠块重复拼接 | Chroma 按 email_id 查询，去除可识别的分块重叠并保留 chunk ID | `core/embedder.py` |
| 索引更新 | 邮件变短残留旧尾块；等量更新不刷新 BM25 | upsert 后清理旧尾块；SQLite revision 与中断 marker 使缓存失效；embedding 保留跨邮件批处理 | `core/embedder.py`、`core/retriever.py` |
| 执行结果 | 工具异常、MCP isError、待审批混成成功 | local/MCP 共用结构化状态；unknown 不继续自动执行；空模型回答标记错误 | `agents/runtime.py`、`agent_loop.py`、`mcp_adapter.py` |
| 运行预算 | 仅限制循环次数 | 增加实际工具调用总数、共享 deadline、上下文字符预算；嵌套模型不做 SDK 隐式重试 | `agents/runtime.py`、`agent_loop.py` |
| 证据 | 返回来源不代表回答已引用 | sources 标记为检索候选；校验回答 `[email_id#chunk_id]` 是否真实存在，单列有效与无效引用 | `agents/agent_loop.py` |
| 审批 | JSON 读改写并发覆盖；执行失败可重复 | SQLite 事务认领，绑定 owner/session/request/hash/TTL；成功重放返回旧结果，未知状态禁止重放 | `agents/approvals.py` |
| Provider | 模拟执行可能声称 sent | simulated 始终 `sent=False`；Gmail 只创建 draft | `agents/mail_providers.py` |
| API | 敏感操作缺鉴权、默认共享 session | 本机身份边界/可选 Bearer token；服务端生成 session UUID；按 owner/session 隔离、容量限制、TTL、并发冲突 | `api/security.py`、`api/sessions.py` |
| SSE | 队列增长、历史遗漏、前端丢失错误/来源 | 有界队列、历史传递、来源与错误事件；超时/断连后不启动下一阶段 | `api/main.py`、`frontend/app.py` |
| 日志 | 查询/工具实参/供应商异常原文可能入日志 | 运行日志保留类型、状态、哈希、长度；不保存原始 query/arguments/正文 | `agents/tracing.py`、`mcp_adapter.py` 等 |
| 评测 | judge 只看最终话术；gate 信任 summary | judge 收到标准、执行状态和候选证据；确定性失败不能被 judge 覆盖；gate 重算 records 并验证源码/数据指纹 | `scripts/run_agent_eval.py`、`agents/eval_contract.py` |
| 打包部署 | COPY 整库可能带入凭据；前端运行时缺 SSE/配置依赖 | 源码 COPY 白名单、私密目录忽略、前端独立 Dockerfile 与依赖清单、端口仅发布到本机 | `Dockerfile*`、`docker-compose.yml`、`.dockerignore` |

## 2. 配置及兼容变化

### API 与会话

- 默认 `API_HOST=127.0.0.1`。无 token 时必须同时符合本地 TCP peer、本地 Host、允许的浏览器 Origin。
- Docker、反向代理和远程访问设置随机长 `API_AUTH_TOKEN`。请求用 `Authorization: Bearer <token>`；Streamlit 自动转发。
- 这是单 owner 部署，token 代表 `API_OWNER_ID`，不是多用户账号体系。不要把同一实例当成公司多租户后端。
- `MCP_OWNER_ID` 默认继承 `API_OWNER_ID`。MCP 服务端只使用自己的配置身份；客户端身份不匹配时拒绝工具调用。MCP 远程服务和 API 必须使用同一 owner 与审批持久化存储，API 才能查看/批准远端产生的审批。
- 省略 session_id 时返回新的 `metadata.session_id`，客户端应保存后续复用。`DELETE /chat/history` 必须提供 session_id。同一会话正在执行时返回 409，清除也返回 409。
- 会话默认最多 1000 个，空闲 TTL 3600 秒；超容量淘汰空闲会话。内存不会跨进程、重启或多 worker 共享。
- `/health` 仅表示 API 存活，不证明模型、Chroma 或 Gmail 可用。

### 工具与来源

- `call_tool` 的结果变成 `_tool_result=1` envelope，状态为 success/error/approval_required/unknown 等；旧的直接读取裸工具结果的外部调用方需要适配 `data`。
- `metadata.steps` 包含 status/error_code/argument_summary，不再包含原始 arguments。UI 已适配。
- `metadata.sources_kind=retrieved_candidates`，`citation_check=identity_only_not_entailment`。引用 ID 存在只证明来源可定位，**不证明该片段支持整句话**。无引用也不自动等于回答真实。
- 总 deadline 与模型 timeout 控制新操作发起；同步 embedding、HTTP/provider 已经运行时不能强制杀掉。断开 SSE 不代表撤销已提交的外部动作。
- 默认预算：6 个规划轮、12 次实际工具调用、120 秒、60000 个上下文字符、单工具输出 4000 字符。字符预算不是 tokenizer 精确计费。
- 显式 `filters={}` 表示不让模型再提取条件。labels 为 ALL 语义。默认日历时区 UTC+8，可由 `RETRIEVAL_TIMEZONE_OFFSET_HOURS` 修改。

## 3. 审批数据迁移与恢复

先停止旧进程，再备份审批目录。默认仍接受 `pending_actions.json` 路径，但新实现读取旧 JSON 后写入相邻的 `pending_actions.sqlite3`。导入事务只执行一次，不回写 JSON；后续以 SQLite 为准。不要并行运行旧版写 JSON 与新版写 SQLite，也不要通过删除 SQLite 强行重新执行旧审批。

旧数据没有 owner 时归属 `local`。若改用其他 `API_OWNER_ID`，旧单不会自动转交，应在离线核对后由可信维护流程迁移；本轮不提供自动换 owner 操作。

| 状态 | 含义 | 再 approve |
|---|---|---|
| pending | 未执行，等待人工确认 | 未过期且内容哈希一致才可认领 |
| executing | 已持久化认领，执行尚未确认 | 拒绝，避免并发/崩溃后重复 |
| approved + succeeded | 已记录执行结果 | 返回原有结果，不再调用 provider |
| unknown | 远端可能已成功，但不能确认 | 拒绝；核对远端草稿和保存的结果 |
| rejected / expired / failed | 已拒绝、过期或确定未提交的输入错误 | 拒绝；重新审查后创建新请求 |

同 owner 的稳定 request_id 与 payload hash 保证本地相同逻辑请求不重复创建；同一 Agent run 内相同 send 参数复用审批结果。**不同 run、进程重启或客户端重新发起请求没有端到端 exactly-once 保证。** SQLite 与 Gmail 无法组成一个原子事务。

Gmail draft 带稳定 Message-ID 便于人工核对，这不是 Gmail 幂等键。unknown 不提供自动重置、补偿或自动重试接口；不能为了恢复而盲目改回 pending。模拟执行始终没有真实发信。

## 4. 索引更新与恢复

建议在自己的模型环境首次升级后重新索引，并验证固定问题与 ID。更新仅替换传入邮件，输入中省略的邮件不会被删除；Gmail 删除/标签变更的完整 history 游标同步仍未实现。

本应用的写入通过相邻 `corpus_revision.sqlite3` 串行化；写入期间或硬退出残留 `.writing.*` marker 时，不复用旧 BM25。完成后更新 revision。若写入失败，修复原因后重新索引受影响的数据。

这不等于 Chroma 事务快照：失败可能留下部分新数据，查询与写入重叠时也可能看到中间状态。embedding 跨邮件按 batch_size 编码，暂存向量内存随 chunk 数量线性增长。直接绕开应用修改 Chroma 不受 revision 协议保护。

`get_indexed_email` 返回的是清洗/分块后的索引文本，不是原始 MIME 或原始 HTML。需要原件审计时仍应保存独立原始邮件库。

## 5. 评测与验证

新增与更新的离线回归可在已有轻量依赖的环境执行：

```powershell
python -B -m unittest tests.test_api_contracts tests.test_agent_eval_gate tests.test_approval_transactions tests.test_runtime_safety tests.test_retrieval_contracts -v
```

这些测试实际调用当前业务模块、FastAPI ASGI 应用和临时 SQLite；模型、Chroma collection、邮件 provider/远端 MCP 由可控替身代替。它们检验的是明确行为与故障处理，不是实际模型质量、GPU 性能或真实邮件联调。

完整 pytest 在自己的开发环境或 CI 执行：

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q
```

`.github/workflows/tests.yml` 在 push/PR 运行回归；`agent-eval-gate.yml` 在任务集/Agent 报告变更的 PR 或手动触发时检查真实报告，不调用模型。源码单独变更不会自动重跑模型质量 gate，应重新生成报告并提交后检查。仓库现有旧报告预期会被新 gate 拒绝，不能把它当成当前版本的绿色结果。

本次工作没有安装 pytest，也没有运行完整 pytest、Docker build、模型下载、真实 Chroma、真实 Gmail/OAuth 或完整模型评测。仓库已有历史数据与成绩保留，不作为新版本通过证明。

模型环境齐全后重新生成当前版本评测：

```powershell
python scripts/run_agent_eval.py --report-output data/eval_results/agent_eval_report.md
python scripts/check_agent_eval_gate.py --input data/eval_results/agent_eval.json
```

gate 要求 schema_version=2、非空且唯一任务 ID、有 trace 标识、合法状态/工具列表、逐条与汇总一致，以及当前源码和任务集 SHA-256。异常运行也保留失败记录和已执行步骤；未进入 Agent 的异常标注 evaluation 范围 trace。若关闭 trace 持久化，trace_id 仍是关联标识，并不表示磁盘一定存在日志。

`tool_accuracy` 的准确含义是“期望工具覆盖率”，同时提供 `expected_tool_coverage` 同义字段；不等于调用参数正确率。judge 已收到候选来源和成功标准，但仍是模型判断，需人工抽检。指纹和校验防止误用旧报告、指标不一致，不是防恶意伪造的签名证明。

## 6. 三个子 agent 的交叉审查闭环

- 检索实现者复核运行器，发现空模型回答被标记成功、预算终态覆盖待审提示；已修复并补测试。
- 审批实现者复核 API/评测，发现流式超时后新请求、异常记录不符合 gate、本地 Host 缺约束、清空会话忽略失败；已修复并补回归。
- 运行器实现者复核审批的事务认领、owner/hash、迁移、执行成功后落盘失败；并发、故障和重复执行测试通过。
- 主 agent 复核跨模块接口，并发现按邮件 embedding 的批量吞吐退化和同 run MCP 重复审批边界；对应修复和测试纳入最终版本。

## 7. 后续仍需验证或建设

1. 在真实环境执行完整 pytest、Docker build、模型与邮箱联调、全量 Agent eval；没有测量的性能数字不填估计值。
2. 标注真实邮件 supporting chunks，测筛选后的召回率；当前硬过滤作用于有限候选，可能漏召回，但不会放宽约束返回不匹配邮件。
3. 根据个人使用需求再做持久化会话、持久任务队列、任务取消和跨进程恢复；目前不是 durable workflow 系统。
4. Gmail 完整变更同步、原件存储、unknown 人工核对界面、长期审计保留与轮转尚未实现。
5. 多租户 OAuth、细粒度权限、网络部署限流与完整依赖锁定未实现。本次采用轻量单 owner 边界，不直接复制公司系统的复杂基础设施。
