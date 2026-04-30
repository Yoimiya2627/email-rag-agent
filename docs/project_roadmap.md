# 项目一回家任务清单

> 来源：微信截图（共37张），由 Claude Code 提取整理

---

## 任务列表（共9个）

### 任务1：扩充邮件数据到 5000 封 ✅

- 运行 `scripts/generate_emails.py` 批量生成
- 验证 `data/emails.json` 加载正常，索引正常运行
- **状态：已完成**

---

### 任务2：Query Rewriting（查询改写）✅

- 在 `agents/retriever_agent.py` 添加 `_rewrite_query()` 方法
- LLM 将口语化问题改写为更适合语义检索的查询语句
- 开关：`ENABLE_QUERY_REWRITE=true/false`
- **状态：已完成**

---

### 任务3：RAGAS 评测框架 ✅

**3.1 生成标注数据**
- 脚本：`scripts/generate_ragas_data.py`
- 从 `emails.json` 中随机抽样，LLM 生成 100 条 Q&A 对
- 输出：`data/ragas_testset.json`

**3.2 评测脚本（6版本消融实验）**
- 脚本：`scripts/run_ragas_eval.py`
- 输出：`data/eval_results/comparison.json` + 各版本详情

**6个版本对比：**

| 版本 | BM25 | RRF | Reranker | Query Rewrite | 说明 |
|------|------|-----|----------|---------------|------|
| V1   | ❌   | ❌  | ❌       | ❌            | 纯向量检索（基线） |
| V2   | ✅   | ✅  | ❌       | ❌            | 混合检索 |
| V3   | ✅   | ✅  | ✅       | ❌            | 混合检索 + Reranker |
| V4   | ✅   | ✅  | ✅       | ✅            | 完整流水线 |
| V5   | ✅   | ❌  | ✅       | ✅            | 去掉 RRF（验证 RRF 贡献） |
| V6   | ✅   | ✅  | ❌       | ✅            | 去掉 Reranker（验证 Reranker 贡献） |

**评测指标：**
- `answer_relevancy`：答案与问题的相关度（0-1）
- `faithfulness`：答案是否有据可查（0-1）
- `context_precision`：检索片段的有效比例（0-1）

**运行命令：**
```powershell
# 先生成测试数据（只需跑一次）
python scripts/generate_ragas_data.py

# 跑全部6个版本（每版本默认30条问题）
python scripts/run_ragas_eval.py --limit 30

# 只跑指定版本
python scripts/run_ragas_eval.py --versions V1,V3,V4 --limit 30
```

- **状态：已完成**

---

### 任务4：多轮对话记忆 ✅

- `core/memory.py`：`ConversationMemory` 滑动窗口（`max_turns=5`）
- 每个会话独立 `session_id`，服务端用 dict 存储
- API 新增 `DELETE /chat/history` 端点清除记忆
- `models/schemas.py`：`AgentRequest` 新增 `session_id` 字段
- **状态：已完成**

---

### 任务5：LangGraph Self-RAG 工作流 ✅

- `agents/graph_workflow.py`：手动状态机（不依赖 langgraph 包）
- 节点流程：`rewrite_query → retrieve → grade_contexts → generate`
- Self-RAG 逻辑：若 `grade_contexts` 判断无相关文档，回退重写并重试（最多 2 次）
- API 新增 `POST /chat/graph` 端点
- 前端可通过"Self-RAG 工作流"开关启用
- **状态：已完成**

---

### 任务6：SSE 流式输出 ✅

- `core/generator.py`：`stream_generate()` 使用 OpenAI `stream=True`
- `api/main.py`：`POST /chat/stream` 端点，返回 `text/event-stream`
- 前端：侧边栏"流式输出"开关，token 逐字显示（带光标动画）
- **状态：已完成**

---

### 任务7：LangChain 并行版本 ✅

- `langchain_version/rag_chain.py`
- 使用 `ChatOpenAI`（指向 DeepSeek）+ `Chroma`（复用同一 ChromaDB）
- `ConversationalRetrievalChain` + `ConversationBufferWindowMemory`
- 两版并存，可独立对比效果
- 安装依赖：`pip install langchain langchain-openai langchain-community`
- **状态：已完成**

---

### 任务8：Docker 容器化 ✅

- `Dockerfile`：`python:3.11-slim`，构建时预下载 `bge-m3` 模型进镜像
- `docker-compose.yml`：api（8000）+ frontend（8501）双服务，数据卷挂载
- `.dockerignore`：排除 `.env`、`.venv`、`chroma_db`、`.hf_cache`
- **启动命令：**
  ```bash
  docker-compose up --build
  ```
- **状态：已完成**

---

### 任务9：降级 + 超时策略 ✅

- `LLM_TIMEOUT=15`（秒），在 `.env` 中配置
- **Reranker 熔断器**：连续 3 次失败后自动跳过 LLM Reranking，降级为原始排序
- **Generator 降级**：LLM 超时时返回原始检索片段（不报错，给用户看到内容）
- **状态：已完成**

---

## 执行日程

| 天   | 任务                                                                 |
|------|----------------------------------------------------------------------|
| Day 1 | 任务1验证 + 任务3.1生成100条标注数据 + 任务3.2评测脚本 → 跑 V1/V2/V3 |
| Day 2 | 任务2 Query Rewriting → 跑 V4/V5/V6 → 输出6版本对比表格             |
| Day 3 | 任务4 多轮对话记忆                                                   |
| Day 4 | 任务5 LangGraph Self-RAG                                             |
| Day 5 | 任务6 SSE 流式输出                                                   |
| Day 6 | 任务7 LangChain 版 + 任务8 Docker                                    |
| Day 7 | 任务9 降级策略 + 整体联调 + README 更新                              |

---

## 给 Claude Code 的行为注意事项

- 先读代码再修改，不要凭空创建新文件
- 用 C++ 类比解释 Python 概念（用户有 C++ 背景）
- 每完成一个任务就 git commit 一次
- 不要删除现有代码，用 feature flag 控制新功能开关
- 不确定的地方先问用户
