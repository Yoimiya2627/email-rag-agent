# 上下文优化：使用、迁移与验证

本批范围为 CTX-PLAN-1.0 的 CTX-01～20，由三个 Astra 子 agent 分工开发，主 agent 集成和验收。历史向量检索、跨会话记忆、多邮箱连接器、附件二进制解析和真实发送不属于本批。

## 日常使用

1. 正常聊天使用当前原话、当前任务最近成功问答及有界相关历史。中文二元词索引、精确编号校验和命中附近摘录可找回窗口之外的轮次。失败回答保留历史状态，不能作为原邮件事实。
2. 展开“回查历史与记录任务约束”，勾选“查看当前任务与记忆”。可选择任务，或从某次用户要求建立目标和对象。普通话题变化不会自动建立新任务或跨会话搜索。
3. 在“管理显式任务约束”保存、修改或撤销有来源的约束。当前原话更正优先；确定性识别支持预算、日期、语言、发送等键及明确修正措辞。自由文本语义不会直接成为已确认字段，未识别表达保留原话，可通过页面明确固定字段。
4. 历史搜索展示命中窗口、原回答状态和来源轮次；“读取这一轮原文”分页展开。邮件引用仍需核对原文版本与 hash。历史回答、摘要和工具快照不提供执行授权。
5. 回答下可查看遗漏和降级。附件未读、清单未知与空清单保持区别。必要要求超出硬预算时明确停止，不通过扩大预算或裁掉当前请求掩盖问题。

## 默认开关

| 配置 | 默认 | 行为 |
|---|---|---|
| `ENABLE_CONTEXT_OPTIMIZATION` | `true` | 当前任务召回、选材和原话事件；关闭使用兼容召回入口，原始记录保留 |
| `CONTEXT_MATERIAL_CHAR_LIMIT` / `CONTEXT_MATERIAL_TOKEN_LIMIT` | `4000` / `2000` | 补充材料硬上限，保留原基线值 |
| `MODEL_CONTEXT_SAFETY_TOKENS` | `128` | 实际最大输出之外的封装余量 |
| `MODEL_REVISION` | 空 | 未匹配已验证计数器时保守估算 |
| `MODEL_CONTEXT_PROFILES` / `CONTEXT_PURPOSE_WEIGHTS` | `{}` / `{}` | 按模型/阶段收窄容量；材料软配额可借用空余额 |
| `CONTEXT_TOOL_RESULTS_ENABLED` / `CONTEXT_TOOL_COMPACTION_ENABLED` | `true` / `true` | 有界结果存储、闭合只读组确定性引用替换 |
| `ENABLE_CONTEXT_SUMMARY` / `ENABLE_CONTEXT_CANDIDATES` | `false` / `false` | 语义摘要/候选抽取试验，开启会新增模型调用 |

`register_token_counter(model, revision, stage, counter, validation_samples=...)` 接受后端安装并校验的适配器。请求不会下载 tokenizer，也不会将其他模型计数器标为准确。无匹配时使用保守 UTF-8 字节估算，注册适配器运行失败也降级。实际 usage 与估算分开记录，缺 usage/价格保留 unknown。

各调用按实际最大输出、安全余量和工具 schema 计算预算。摘要、候选和 Agent 共用请求费用、token、deadline。恢复先加载原账本，不能先新增调用再覆盖用量。

摘要从有界原始轮次重建，保存逐条引用、覆盖、模型/提示/schema版本及输入指纹。按新增未覆盖量触发，成功输入复用，失败/未知输入受持久尝试上限限制。坏 JSON、来源不符、未完成输出、并发修改、取消或超预算不会发布有效新摘要。结构和来源校验不能证明语义正确，两个语义试验在真实模型验收前保持关闭。

## 接口与范围

| 接口 | 用途 |
|---|---|
| `GET /chat/context` | 任务、约束、候选、修正、摘要和后台 job 进度 |
| `GET/POST /chat/tasks`；`POST /chat/tasks/select` | 明确建立/修改/选择任务，要求 revision |
| `POST /chat/facts/{confirm,revoke,delete}` | 候选确认与约束撤销，要求来源和版本 |
| `GET /chat/history/turn` | 原轮次 query/answer 分页及 hash |
| `GET /chat/summary`；`POST /chat/summary/rebuild` | 有效语义摘要及重建，试验关闭时拒绝重建 |
| `GET /chat/tool-result` | 按 owner/session/run/epoch 校验结果页 |

`/chat/history/summary` 仍为确定性摘录，旧 017 的完成内容不能改称语义摘要。现有身份中间件绑定 owner，模型参数不能选择身份。`/query` 保持无会话语义。

Agent 新增 `search_history`、`get_turn`、`get_tool_result` 只读工具，通过可信 RunContext 注入范围。独立 MCP server 没有可信会话仓库，这些调用返回范围不可用，不因共用 MCP owner 开放全部会话。可选 LangChain 演示自带临时对话窗口，未接本应用的持久会话、可信身份与统一预算，不属于生产上下文入口。

## 工具存储与恢复

`TOOL_RESULT_STORE_PATH` 默认是 `data/sessions/tool_results.sqlite3`。单对象上限 512,000 字节、单 run 4,000,000 字节、全库 64,000,000 字节/4096 对象，默认保留 7 天。超限保留原有有界输出并标记未完整保存，不承诺任意大结果全部保存。受限 JSON 支持分页回读。

引用替换保留调用/结果协议和最后完整组。审批待定、失败、unknown 不跨越压缩边界。日志、操作槽、计数和费用由后端维护；快照不能触发写操作重放。当前采用确定性引用替换，没有新增模型压缩调用。

工具结果默认保留 7 天，后续成功写入会自动回收过期对象以释放容量。不可变 call 的幂等复用限定在留存期内；过期或回收后的原 result_id 不再可读，引用它的 checkpoint 拒绝恢复，不自动重放工具。删除会话也会清理其结果。

checkpoint v2 绑定请求、工具 schema、模型/预算/策略、owner/session/run/删除 epoch、任务约束和实际引用摘要。普通失败转录追加不会重置费用或重放工具。暂停后修改任务/约束、摘要失效、结果缺失/hash不符会拒绝恢复。持久会话 v1 缺少新归属保护，要求重新开始；无会话 v1 保留兼容校验。job 的 2,000,000 字节限额不变。

## 迁移与路径检查

旧会话库首次打开时创建 SQLite 一致性备份 `.pre-context.bak` 及 hash，添加上下文表，不重写原问答。索引失败记录待重建，不因此丢弃已提交转录。先停止 API/UI/MCP/worker，在新路径排练：

```powershell
python scripts/context_maintenance.py migrate-copy --source C:/state/sessions.sqlite3 --destination C:/review/sessions.sqlite3 --service-stopped
python scripts/context_maintenance.py rebuild-index --sessions C:/review/sessions.sqlite3 --service-stopped
python scripts/doctor.py --profile core --probe-filesystem
```

重建分批执行，达到总批次数或有超大记录失败时返回 partial。Windows 检查主文件/侧文件预计长度，可进行临时写入探测；不修改注册表、不假设 long-path 支持。源码与测试临时路径都应简短。

## 备份、删除与回退

```powershell
python scripts/state_maintenance.py backup --sessions C:/state/sessions.sqlite3 --tool_results C:/state/tool_results.sqlite3 --jobs C:/state/jobs.sqlite3 --destination C:/backup/context-1 --service-stopped
python scripts/state_maintenance.py restore --backup-directory C:/backup/context-1 --destination C:/restore/context-1 --service-stopped
```

恢复只写新目录，禁用旧 job checkpoint、隔离未决审批、递增会话 epoch并失效摘要，让恢复工具对象过期。实际审批库另加 `--approvals <path>`。备份保留原内容，当前系统删除不代表历史备份已物理擦除。

删除先写 tombstone 阻断旧运行回写/恢复，再幂等清理结果和 job。不承诺跨库原子事务。API 返回 `history_inaccessible` 和 `cleanup_pending`；重复删除可继续清理。停服务后也可执行：

```powershell
python scripts/context_maintenance.py cleanup --sessions C:/state/sessions.sqlite3 --tool-results C:/state/tool_results.sqlite3 --jobs C:/state/jobs.sqlite3 --service-stopped
```

默认 dry-run，检查目标后加 `--apply`。最小范围/版本墓碑不被普通留存删除。FTS5 不可用时源会话仍可删除，但不宣称不可访问的 FTS 残骸已物理擦除。

回退优先关闭新开关并保留原始记录。恢复旧备份会回到该时点，不保留之后的新问答，不能称为无损回滚。本轮未迁移实际业务库或修改真实邮箱配置。

## 验证与已知事项

回归使用临时 SQLite/Chroma、实际 ASGI/Streamlit 测试框架及替身模型，阻断网络。专项诊断分别记录正确轮次命中、关键句可见和语义质量未验证，保留失败分母。选定模型、真实任务 gold、延迟和费用待单独验收。

K1 附件覆盖漏传已修复。K2 审批邮箱账号绑定、K3 Gmail capture 版本重放仍为独立待修包。现有 Gmail 同步采用分页扫描/检查点与范围核对，未实现 history/startHistoryId 增量机制；reconcile 为人工结案。V1–V7 是历史评测文件，不是本次版本成绩。
