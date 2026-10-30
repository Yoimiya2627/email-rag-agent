# 智能邮件 Agent：生产就绪度与 30K 岗位审查（2026）

审查日期：2026-08-02  
项目定位：production-oriented prototype（面向生产的高级原型），不是 enterprise production-ready 产品。

## 最终判断

**足够。**

就“2026 年深圳、月薪约 30K 的 AI Agent 开发工程师”的简历项目而言，本项目的技术含量已经足够进入面试并支撑一轮有深度的系统设计讨论。这里的“足够”指项目本身的覆盖面和工程深度足够，不代表仅凭仓库即可保证获得该薪资，也不代表系统已经可以无条件承载企业生产流量。

面试能否达到对应水平，取决于候选人能否独立解释、复现和修改这套系统，而不是能否展示页面或背诵 README。

## 当前能力画像

| 维度 | 当前水平 | 证据 |
|---|---|---|
| Agent 设计 | 较强 | Function calling 多步循环、Skill、工具策略、循环保护、上下文预算、MCP 后端 |
| RAG | 较强 | Vector + BM25 + RRF、reranker、query rewrite、Self-RAG、真实/合成评测 |
| 安全 | 较强 | 可信 principal、RBAC、租户隔离、Prompt Injection 边界、敏感日志脱敏、fail-closed 配置 |
| 高风险动作 | 较强 | pending → executing → approved/rejected、幂等、原子 claim、Gmail draft-only、人工对账 |
| EvalOps | 较强 | 合成/真实任务集、失败归因、质量 gate、Dashboard、stale/provenance 提示 |
| 后端工程 | 中上 | FastAPI、SQLite durable state、限流、审计、测试、CI、配置校验 |
| 分布式生产能力 | 尚未完成 | 尚未采用 OIDC、Postgres/Redis、队列、分布式锁/限流和完整 OpenTelemetry |

## 本轮完成的四项生产化改造

### 1. 可信身份、RBAC 与多租户隔离

- Bearer token 映射为服务端可信的 `subject / tenant_id / roles`，生产模式强制 principals 配置并禁用 legacy 共享管理员 token。
- 客户端无法通过 `X-Tenant-ID` 覆盖可信租户。
- 租户上下文贯穿 Chroma、BM25、邮件详情、会话、审批、审计和 MCP 路由。
- reader/operator/approver/admin 权限下沉到 API、模型可见工具 schema 和工具执行时三层；本地工具、Skill backend、远端 MCP backend 都会二次鉴权。
- 每个租户只能索引被精确分配的数据文件；生产环境缺少租户数据源映射时拒绝启动。

### 2. Prompt Injection 防护与安全评测

- 邮件正文和工具输出作为 JSON 结构化“不可信数据”进入模型，不使用可由正文闭合的文本标签。
- generator、reranker、graph、summarizer、writer 和 Agent tool-result 六条模型链路共享安全 policy。
- 检测集覆盖英文、中文、delimiter escape、工具诱导与数据泄露请求。
- 安全 gate 同时运行确定性检测和模拟模型端到端回归：即使模型伪造 `send_email` 调用，reader 也无法触达执行器。

需要诚实说明：正则检测不是 Prompt Injection 的根本解法；真正的安全边界是最小权限、结构化数据、工具执行时鉴权、审批和数据隔离。后续还应增加真实模型、多轮和变体攻击评测。

### 3. 审批状态机、幂等与崩溃恢复

- SQLite 使用唯一幂等键、冲突校验和原子执行 claim，避免并发重复发送。
- provider 调用不占用长数据库事务；失败后可恢复 pending。
- Gmail draft 带稳定幂等标识，便于外部核对。
- 若进程在外部成功后、本地提交前崩溃，管理员必须先核实外部状态：确认成功时只补记结果；确认未执行时才退回 pending，不自动重放。
- JSON backend 只保留 demo 用途，生产模式明确禁止。

### 4. Eval Dashboard 与证据可信度

- Dashboard 展示合成/真实 Agent 成功率、工具准确率、失败分类、版本趋势、安全指标、样本量和数据新鲜度。
- 失败样例默认不展示邮件正文等敏感内容。
- 历史评测如果没有 commit/config/dataset provenance，或代码已变化/工作区 dirty，会明确标记 stale，不冒充当前版本成绩。

## 验证结果

- 独立只读审查线程最终验收：通过，P0 0 项、P1 0 项。
- 全量测试：218 passed。
- Python 编译检查：PASS。
- correctness-critical Ruff：PASS。
- 依赖一致性：PASS。
- Prompt Injection gate：14 个检测样例通过；攻击检出率 100%，误报率 0%，结构边界失败 0；模拟模型端到端安全回归通过。
- 历史 Agent EvalOps 产物：105 tasks，任务成功率 80.95%，工具准确率 88.57%，禁用工具违规率 0%，gate PASS。
- 历史真实 Gmail Agent 产物：30 tasks，任务成功率 93.33%，工具准确率 100%。

最后两组模型指标来自改造前的历史评测产物，不能当作本轮代码的最新成绩。Dashboard 会将其标为 stale。简历正式使用前，应在干净 commit 上重跑，并把 commit SHA、配置 hash、数据集 hash 和时间写入产物。

## 仍未达到企业级生产的部分

这些不否定其作为 30K 简历项目的含金量，但面试时必须主动说明：

1. 身份仍是静态 token → principal 映射，不是企业 OIDC/OAuth2、短期令牌、密钥轮换和组织目录集成。
2. SQLite/本地 Chroma 适合单机或小规模部署；多实例应迁移到 Postgres、Redis、托管向量库和队列。
3. 审批崩溃窗口依赖人工对账；更高等级系统还需 execution owner、lease、fencing token 和 provider webhook/reconciliation worker。
4. 限流是进程内实现，不支持多实例统一配额。
5. 可观测性已有日志、trace 和 audit，但缺完整 OpenTelemetry、SLO、告警、成本与 token 用量监控。
6. Prompt Injection 仍需更大规模真实模型红队集、多轮污染、Unicode/编码变体和数据外泄自动判定。
7. `get_email` 等部分查询在大邮箱下仍有 O(N) 路径，需要按 metadata 精确查询和压测。
8. Docker 运行环境在本次本机审查中不可用，因此未完成镜像构建和 compose 集成验证；代码级部署约束和 Streamlit smoke 已验证。
9. 仓库仍有遗留格式/编码与非关键 Ruff 技术债；CI 当前只阻断 correctness-critical 规则。

## 达到 30K 面试水平的掌握标准

### 必须能独立讲清楚

1. 从 API 请求到检索、rerank、planner、tool call、审批和 Gmail draft 的完整调用链。
2. 为什么 Agent 不应只依靠 system prompt 保证安全，以及 RBAC 为什么必须同时作用于 schema 和执行器。
3. 混合检索、RRF、cross-encoder reranker 的收益、延迟和适用边界。
4. tenant ContextVar 如何传播，异步任务/线程为什么可能丢上下文，如何测试越权。
5. SQLite `BEGIN IMMEDIATE`、唯一幂等键、provider 调用崩溃窗口和人工 reconcile 的取舍。
6. 为什么“历史评测 PASS”不能证明当前 commit 质量，以及 provenance 应包含什么。
7. workflow、Agent 和 MCP 各自适合解决什么问题，什么时候不该使用自主 Agent。

### 必须能现场完成

- 不看答案新增一个只读工具，并补 schema、RBAC、审计、测试和 eval case。
- 构造一个恶意邮件，证明 reader 无法调用写工具或读取其他租户数据。
- 人为制造 provider 超时、重复审批和执行后崩溃，解释状态如何恢复。
- 解释一次失败评测记录，定位是检索、规划、参数、策略还是生成问题。
- 从空白写出简化版 agent loop、tool dispatcher 和 approval claim 核心逻辑。

### 建议的掌握路线

第一阶段（2 天）：画出系统架构、请求时序和审批状态机，不看代码讲 10 分钟。  
第二阶段（3 天）：逐模块删掉关键实现后重新写一遍，重点练 agent loop、RBAC、幂等和 tenant scope。  
第三阶段（2 天）：准备 10 个故障实验，包括注入、越权、重复发送、模型超时、MCP 断连和索引污染。  
第四阶段（1 天）：在干净 commit 上重跑完整合成/真实 eval，生成带 provenance 的结果和截图。  
第五阶段（持续）：用 3 分钟、10 分钟、30 分钟三个版本讲项目，并准备架构取舍追问。

## 简历表达原则

- 写“构建面向生产的多租户邮件 Agent 原型”，不要写“已上线企业生产系统”。
- 指标必须标注样本量、数据来源和 commit；在最新评测完成前，不把历史指标描述成当前版本结果。
- 把重点放在可验证的难点：工具级 RBAC、租户隔离、审批幂等、Prompt Injection 防御纵深、真实 Gmail eval 和失败归因。
- 面试时主动说出尚未实现的 OIDC、分布式状态和完整可观测性，并给出迁移方案。这通常比宣称“全部生产级”更能体现判断力。
