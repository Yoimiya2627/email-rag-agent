# 项目架构文档

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户 (浏览器)                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ http://localhost:8501
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   frontend/app.py                               │
│                   Streamlit UI 层                                │
│  · 聊天对话界面        · 快捷问题按钮                              │
│  · 索引管理（建立/清除）· 索引状态展示                             │
│  · 来源引用展示        · 统计图表（发件人/标签/每日量）             │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP POST /chat  /index  /index/status
                          │ http://localhost:8000
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     api/main.py                                 │
│                    FastAPI REST 层                               │
│                                                                 │
│   GET  /health          POST /index          GET  /index/status │
│   POST /chat            POST /index/clear    POST /query        │
└──────┬────────────────────────┬──────────────────────────────────┘
       │ /chat                  │ /index
       ▼                        ▼
┌─────────────────┐    ┌─────────────────────────────────────────┐
│  Agent 层        │    │              数据索引流水线               │
│                 │    │                                         │
│ coordinator.py  │    │  loader.py → cleaner.py → chunker.py   │
│  LLM 意图识别    │    │       ↓                                 │
│  路由分发        │    │  embedder.py (index_chunks)             │
│                 │    │  sentence-transformers 编码              │
│  ┌──────────┐   │    │  → ChromaDB upsert                      │
│  │retrieve  │   │    └─────────────────────────────────────────┘
│  │summarize │   │
│  │write     │   │
│  │analyze   │   │
│  └──────────┘   │
└──────┬──────────┘
       │ 前三个 Agent 共享检索链路
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RAG 检索链路                               │
│                                                                 │
│  retriever.py                                                   │
│  ┌────────────────┐   ┌─────────────┐                          │
│  │ vector_search  │   │ bm25_search │                          │
│  │ ChromaDB 余弦   │   │ rank-bm25   │                          │
│  │ 相似度检索      │   │ 关键词检索   │                          │
│  └───────┬────────┘   └──────┬──────┘                          │
│          └─────────┬─────────┘                                  │
│                    ▼ hybrid_search (RRF 融合)                    │
│                    │                                            │
│             reranker.py                                         │
│             LLM 对每个候选打分 0-10，重新排序                     │
│                    │                                            │
│             generator.py                                        │
│             拼装上下文 → LLM 生成最终答案                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 各文件职责

### config/
| 文件 | 职责 |
|------|------|
| `settings.py` | 读取 `.env`，暴露全局配置常量（API Key、模型名、路径、检索参数等） |

### models/
| 文件 | 职责 |
|------|------|
| `schemas.py` | 所有 Pydantic 数据模型：`Email`、`EmailChunk`、`SearchResult`、`AgentRequest/Response`、`IntentType` 枚举等 |

### core/ — RAG 流水线（无状态纯函数）
| 文件 | 职责 |
|------|------|
| `loader.py` | 从 JSON 文件加载邮件，校验必填字段，返回 `List[Email]` |
| `cleaner.py` | 去除 HTML 标签、邮件签名、规范化空白，返回清洗后的 `Email` |
| `chunker.py` | 段落感知切块：长段落强制切分，短段落合并，生成 `List[EmailChunk]` |
| `embedder.py` | 管理 sentence-transformers 模型和 ChromaDB 连接；提供 `index_chunks`、`search_similar`、`get_all_chunks`、`get_collection_stats`、`clear_collection` |
| `retriever.py` | `vector_search`（ChromaDB）+ `bm25_search`（rank-bm25）→ `hybrid_search`（RRF 融合） |
| `reranker.py` | 调用 DeepSeek LLM，对候选片段相关性打分（0-10），重新排序后截取 top-n |
| `generator.py` | 将重排结果拼成上下文，调用 DeepSeek LLM 生成最终自然语言答案 |

### agents/ — Multi-Agent 层
| 文件 | 职责 |
|------|------|
| `coordinator.py` | `classify_intent()`：调用 LLM 判断用户意图（retrieve/summarize/write_reply/analyze/general）；`route()`：实例化对应 Agent 并执行 |
| `retriever_agent.py` | 用 LLM 从问题中提取过滤条件（发件人/日期/标签），执行 hybrid_search → 后过滤 → rerank → generate |
| `summarizer_agent.py` | 检索相关邮件，生成结构化摘要（核心议题/关键信息/待办/结论） |
| `writer_agent.py` | 找到最相关的原始邮件，按用户要求生成专业回复草稿 |
| `analyzer_agent.py` | 对 ChromaDB 中所有邮件元数据做统计（发件人 Top5、标签分布、每日邮件量），LLM 解读结果 |

### api/
| 文件 | 职责 |
|------|------|
| `main.py` | FastAPI 应用；定义 6 个端点；`/chat` 调用 `coordinator.route()`，`/index` 执行索引流水线，`/query` 直连 RAG 跳过 Agent 路由 |

### frontend/
| 文件 | 职责 |
|------|------|
| `app.py` | Streamlit UI；侧边栏管理索引（建立/清除/状态）；主区域聊天对话；展示来源引用和统计图表 |

### 根目录
| 文件 | 职责 |
|------|------|
| `.env` / `.env.example` | 环境变量配置（API Key、模型、路径、检索参数） |
| `requirements.txt` | Python 依赖列表 |
| `data/emails.json` | 示例邮件数据（15 封中文商务邮件） |

---

## 关键数据流

### 索引流程
```
emails.json
  → loader.load_emails()          # List[Email]
  → cleaner.clean_email()         # 去 HTML / 签名
  → chunker.chunk_email()         # List[EmailChunk]
  → embedder.index_chunks()       # 编码 → ChromaDB upsert
```

### 查询流程（以 RetrieverAgent 为例）
```
用户问题
  → coordinator.classify_intent()           # LLM → IntentType
  → RetrieverAgent._extract_filters()       # LLM → sender/date/labels
  → retriever.hybrid_search()               # 向量 + BM25 + RRF
  → _apply_sender/label/date_filter()       # 元数据后过滤
  → reranker.rerank()                       # LLM 打分重排
  → generator.generate_answer()             # LLM 生成答案
  → AgentResponse(answer, sources)
```

---

## 技术选型说明

| 技术 | 选型 | 原因 |
|------|------|------|
| 嵌入模型 | BAAI/bge-m3（本地） | 中英文双语效果好，本地运行零 API 成本 |
| 向量库 | ChromaDB | 轻量、本地持久化，无需部署额外服务 |
| 关键词检索 | BM25（rank-bm25） | 弥补语义检索对精确词汇匹配的不足 |
| 融合策略 | RRF（倒数排名融合） | 无需调参即可稳定融合多路结果 |
| 重排 | LLM 打分 | 理解语义相关性，效果优于 embedding 余弦相似度 |
| LLM | DeepSeek v4-flash | 成本低、中文能力强、兼容 OpenAI SDK |
| 意图识别 | LLM 分类 | 理解口语化表达，优于关键词规则匹配 |
