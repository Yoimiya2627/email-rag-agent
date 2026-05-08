# Email RAG Agent

一个基于检索增强生成（RAG）和多 Agent 架构的邮件智能问答系统。

做这个项目的初衷是想把"RAG 链路里每一层（向量检索 / BM25 / RRF 融合 / LLM 重排 / Query Rewrite）到底各自贡献多少"这件事真正搞清楚——所以从一开始就把所有组件做成可开关的特性，配套写了消融评测脚本去量化它们。

## 它能做什么

- **检索式问答**："Q3 预算评审会议是谁发的？" → 从 5000 封邮件里找出相关邮件回答
- **批量摘要**："这周项目进展整理一下" → 多封相关邮件综合摘要
- **回信草稿**："帮我回 Bob 那封询价邮件" → 起草回复
- **统计分析**："本月每个发件人发了多少封？" → 聚合元数据回答
- **多轮对话**：每个 session 独立记忆，5 轮滑窗（10 条消息）
- **流式输出**：SSE 真流式（不是攒齐再放）

## 快速开始

### Docker（推荐）

```bash
cp .env.example .env
# 编辑 .env，填 DEEPSEEK_API_KEY
docker-compose up --build
```

启动完成后：
- API：http://localhost:8000（Swagger UI: `/docs`）
- 前端：http://localhost:8501

首次启动需要下载 bge-m3 嵌入模型（~570MB），耗时取决于网络环境；模型缓存后后续启动会明显加快。

### 本地开发

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate        # Linux/macOS
pip install -r requirements.txt

cp .env.example .env               # 填 DEEPSEEK_API_KEY
uvicorn api.main:app --reload      # 后端 :8000
streamlit run frontend/app.py      # 前端 :8501
```

第一次访问前端时点侧边栏的"索引邮件"按钮，会把 `data/emails.json` 里的 5000 封邮件向量化并写入 ChromaDB。

## 系统架构

完整的架构图、时序图、状态机详见 [`docs/architecture.md`](docs/architecture.md)，覆盖离线索引、在线问答、混合检索 + RRF、Self-RAG 状态机、SSE 线程桥接、降级策略等。

简化版本：

```
                  ┌──────────────────────────────────────────────┐
   用户 query  →  │  Coordinator（LLM 意图分类）                 │
                  └─────┬─────────┬──────────┬─────────┬─────────┘
                        │         │          │         │
                  RetrieverAgent  Summarizer  Writer   Analyzer
                        │         │          │         │
                        └────┬────┴──────────┘         │
                             ↓                          │
              ┌──────────────────────────┐              │
              │ Query Rewrite (LLM)      │              │
              │ Hybrid Search:           │              │
              │   Vector (bge-m3)        │              │
              │ ⊕ BM25 (rank_bm25)       │              │
              │   → RRF 融合             │              │
              │ → 后过滤 (sender/date)   │              │
              │ → LLM Rerank (熔断器)    │              │
              │ → DeepSeek Generate      │              │
              └──────────────────────────┘              │
                             ↓                          ↓
                          Answer ←── 元数据聚合（发件人 Top / 日期分布）
```

另有一条 Self-RAG 链路（`POST /chat/graph`），用 LangGraph 实现状态机，在生成前加一步"LLM 判断检索结果是否真的相关"，不相关就改写 query 重试（最多 2 次）。

## 技术栈

| 层 | 选择 | 理由 |
|---|---|---|
| 嵌入 | `BAAI/bge-m3`（本地） | 中文好、8192 token 长上下文、零 API 成本 |
| 向量库 | ChromaDB（本地持久化） | 开发期零运维、迁移到 Milvus/Qdrant 改一个文件即可 |
| 关键词检索 | `rank_bm25` + RRF 融合 | RRF 不依赖分数量纲（BM25 无界 vs 余弦 0~1） |
| LLM | DeepSeek `deepseek-v4-flash` | 推理模型，重排和打分质量更稳 |
| 工作流 | LangGraph 1.x | Self-RAG 状态机，支持条件边和循环重试 |
| API | FastAPI + SSE | 流式 token 实时推送 |
| 前端 | Streamlit | 快速搭 chat UI，能切换流式/Self-RAG/普通三种模式 |
| 备选实现 | LangChain | 平行写了一版 `langchain_version/`，验证手写链路和框架版的一致性 |

## 评测

`scripts/run_ragas_eval.py` 实现了 6 个版本的消融对比：

| 版本 | BM25 | RRF | Reranker | Query Rewrite |
|------|------|-----|----------|---------------|
| V1   | ❌   | ❌  | ❌       | ❌            |
| V2   | ✅   | ✅  | ❌       | ❌            |
| V3   | ✅   | ✅  | ✅       | ❌            |
| V4   | ✅   | ✅  | ✅       | ✅            |
| V5   | ✅   | ❌  | ✅       | ✅            |
| V6   | ✅   | ✅  | ❌       | ✅            |

三维度指标（每条都用 LLM 打分，0~1）：
- `answer_relevancy`：答案与问题切题度
- `faithfulness`：答案是否有上下文依据（不编）
- `context_precision`：检索片段中真正有用的比例

跑法：

```bash
python scripts/run_ragas_eval.py --versions V1,V2,V3,V4,V5,V6 --limit 100
```

输出 `data/eval_results/comparison.json` + 终端对比表。

### 实测结果（5000 封语料 / 100 题测试集）

| Version | BM25 | RRF | Rerank | Rewrite | Relevancy | Faithful | Precision |
|---------|:----:|:---:|:------:|:-------:|----------:|---------:|----------:|
| V1 纯向量 | ❌ | ❌ | ❌ | ❌ | 0.8983 | 0.8683 | 0.5077 |
| **V2** +BM25+RRF | ✅ | ✅ | ❌ | ❌ | **0.9517** | 0.9117 | 0.5543 |
| V3 +Reranker | ✅ | ✅ | ✅ | ❌ | 0.9227 | 0.9050 | 0.5787 |
| V4 全开 | ✅ | ✅ | ✅ | ✅ | 0.9117 | **0.9233** | 0.5500 |
| V5 V4-RRF | ✅ | ❌ | ✅ | ✅ | 0.8700 | 0.8517 | **0.6383** |
| V6 V4-Reranker | ✅ | ✅ | ❌ | ✅ | 0.9400 | 0.9117 | 0.5533 |

三个反直觉发现（详见 [`docs/engineering_pitfalls.md`](docs/engineering_pitfalls.md) 第十一节）：

1. **三维赢家分散在三个版本**：answer_relevancy 最优是 V2（向量+BM25+RRF），faithfulness 最优是 V4（全开），context_precision 最优是 V5（V4-RRF）——单维归因都对，叠加在一起就互相抵消，没有"全场最优"的配置
2. **LLM Reranker 是 trade-off 不是单边负贡献**：V2→V3 让 context_precision +0.025（在做它该做的事——剔噪声），代价是 relevancy -0.029；要进一步优化建议替换为 cross-encoder（如 bge-reranker-v2-m3），LLM 当 reranker 在该数据集下收益不稳定且引入 30~50s 延迟
3. **Query Rewrite 是双刃剑**：V3→V4 加 rewrite 让 faithfulness +0.018（升到全场最高），但 relevancy 和 precision 都微降——合成数据本身规整，rewrite 改宽泛反而召回更多边缘 chunk；真实数据上口语化更严重，ROI 应反过来

**业务选型**：在邮件查询场景（关心 relevancy + 低延迟）最终选 V2 方案——在该数据集上 answer_relevancy 优于全开 V4 配置，端到端延迟约低 10×（~3s vs ~38s）。组件不是越多越好，按业务目标和延迟要求选方案。

## 工程问题复盘

完整的问题定位、修复过程和工程反思见 [`docs/engineering_pitfalls.md`](docs/engineering_pitfalls.md)。这里列出最有代表性的几个：

1. **推理模型 `max_tokens` 陷阱**：`deepseek-v4-flash` 的 `max_tokens` 同时覆盖 `reasoning_content` 和 `content`。原代码各处写的是 128~256，结果推理过程吃光预算后 `content` 永远是空字符串——5 处 LLM 调用（RAGAS 打分 / query rewrite / reranker / 意图分类 / Self-RAG grade）全部受影响。修复后普通调用 bump 到 1500、高推理量调用 3000，并加 `reasoning_content` 兜底解析。

2. **SSE 假流式**：`/chat/stream` 端点用 `list(stream_generate(...))` 把所有 token 收完才 yield，等于伪装的非流式。改成 `asyncio.Queue` + 工作线程 `call_soon_threadsafe` 桥接才是真流式。

3. **BM25 每次查询都重建索引**：5000 文档单次 800ms+。加了 `(chunk_count, BM25Okapi, ...)` 三元组缓存 + 用 `collection.count()` 做 cheap probe 检测漂移，命中后 22ms（40× 提升）。

4. **消融实验里 reranker 熔断器状态泄漏**：模块全局变量 `_consecutive_failures` 被 V1~V6 共用，V3 偶发失败会永久关闭后续版本的 reranker。加 `reset_circuit_breaker()` 在每个版本开始前清零。

5. **多线程 session 状态丢失**：`defaultdict(ConversationMemory)` 的 get-or-create 不是原子操作，并发首次访问同一 session 会互相覆盖。换成显式 lock + helper。

## 已知限制

- 单进程方案：`_sessions` 是进程内 dict，多 worker 部署需要换 Redis
- 检索 pipeline 在三处重复实现（RetrieverAgent / graph_workflow / run_ragas_eval），值得抽到 `core/pipeline.py`
- 5000 邮件下 in-memory BM25 仍可接受，到 100 万级别需要换 Elasticsearch

## 后续扩展方向

在邮件场景基础上，逐步抽象可复用的 RAG / Agent / SSE / Memory / Evaluation 能力，计划扩展到客服 FAQ 检索、工单摘要、意图分流等场景。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 健康检查 |
| POST | `/index` | 索引邮件 |
| POST | `/index/clear` | 清空索引 |
| GET  | `/index/status` | 已索引数量 |
| POST | `/chat` | 多 Agent 问答（含意图分类） |
| POST | `/chat/stream` | SSE 流式问答 |
| POST | `/chat/graph` | Self-RAG 工作流（LangGraph） |
| DELETE | `/chat/history` | 清除指定 session 记忆 |
| POST | `/query` | 直连 RAG（不走意图路由） |

请求体（除 `/index` 外）通用：

```json
{
  "query": "最近有哪些重要邮件？",
  "session_id": "user-001"
}
```

## 目录结构

```
.
├── api/main.py                # FastAPI 入口
├── frontend/app.py            # Streamlit
├── agents/                    # Coordinator + 4 个专家 agent + LangGraph 工作流
├── core/                      # loader/cleaner/chunker/embedder/retriever/reranker/generator/memory
├── config/settings.py         # 配置 + .env
├── models/schemas.py          # Pydantic schemas
├── scripts/                   # 数据生成 + RAGAS 评测 + 调试 probe
├── langchain_version/         # LangChain 平行实现
├── data/                      # emails.json + ragas_testset.json + eval_results/
├── chroma_db/                 # runtime 生成，已 gitignore
├── docs/
│   ├── architecture.md        # 架构图与流程详解
│   ├── engineering_pitfalls.md  # 工程问题与修复记录
│   └── project_roadmap.md     # 任务路线图
├── Dockerfile + docker-compose.yml
└── requirements.txt
```

## 开发笔记

- 每个新功能都加了 `ENABLE_XXX` 开关，方便消融实验切换
- 所有 LLM 调用都带超时（默认 60s）+ 三段降级（重试 → 回退到无 LLM 的合理默认 → 友好提示）
- 旧实现保留为降级路径，从不直接删除——这次几次重大重构（SSE / LangGraph / BM25 缓存）都遵守这个规则
- `.env` 在 `.gitignore` 里；密钥不入库
