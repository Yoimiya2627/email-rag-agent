# 邮件智能助手 (Email RAG Agent)

基于 RAG + Multi-Agent 架构的邮件智能问答系统。支持邮件检索、摘要、回复草稿生成与统计分析。

## 技术栈

- **后端**：FastAPI
- **前端**：Streamlit
- **向量库**：ChromaDB（本地持久化）
- **嵌入模型**：`BAAI/bge-m3`（本地 sentence-transformers，零 API 成本）
- **LLM**：DeepSeek（通过 OpenAI SDK 调用）
- **检索**：向量检索 + BM25，使用 RRF 融合
- **重排**：基于 LLM 的相关性打分（0–10）
- **意图识别**：基于 LLM 的分类（非关键词匹配）

## 项目结构

```
.
├── config/settings.py          # 全局配置（读取 .env）
├── models/schemas.py           # Pydantic 数据模型
├── core/                       # RAG 流水线
│   ├── loader.py               # 加载 JSON 邮件
│   ├── cleaner.py              # HTML / 签名清洗
│   ├── chunker.py              # 段落感知切块
│   ├── embedder.py             # 嵌入 + ChromaDB 索引
│   ├── retriever.py            # 向量 + BM25 + RRF
│   ├── reranker.py             # LLM 重排
│   └── generator.py            # 答案生成
├── agents/                     # 多 Agent
│   ├── coordinator.py          # 意图识别 + 路由
│   ├── retriever_agent.py      # 检索问答
│   ├── summarizer_agent.py     # 摘要
│   ├── writer_agent.py         # 回复草稿
│   └── analyzer_agent.py       # 统计分析
├── api/main.py                 # FastAPI 服务（端口 8000）
├── frontend/app.py             # Streamlit UI（端口 8501）
├── data/emails.json            # 示例邮件（15 封中文商务邮件）
├── requirements.txt
└── .env.example
```

## 多 Agent 流程

```
用户问题
   ↓
Coordinator（LLM 意图识别）
   ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ retrieve     │ summarize    │ write_reply  │ analyze      │
│ Retriever    │ Summarizer   │ Writer       │ Analyzer     │
│ Agent        │ Agent        │ Agent        │ Agent        │
└──────────────┴──────────────┴──────────────┴──────────────┘
       ↓ (前三个共享)
   Hybrid Search → LLM Rerank → Generate Answer
```

`AnalyzerAgent` 直接对索引中的元数据做聚合统计（发件人 Top5、标签分布、每日邮件量），再交由 LLM 做自然语言解读。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 3. 启动后端

```bash
uvicorn api.main:app --reload
# 监听 http://localhost:8000
```

### 4. 启动前端

```bash
streamlit run frontend/app.py
# 浏览器打开 http://localhost:8501
```

### 5. 索引数据

在 Streamlit 侧边栏点击 **"🗂️ 索引邮件"**（首次会下载 ~500MB 嵌入模型）。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health`        | 健康检查 |
| POST | `/index`         | 索引邮件（body: `{"data_path": "..."}`，可省） |
| POST | `/index/clear`   | 清空索引 |
| POST | `/chat`          | 多 Agent 问答（含意图识别） |
| POST | `/query`         | 直连 RAG 查询（不走意图路由） |

示例：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "最近有哪些重要邮件？"}'
```

## 邮件 JSON 格式

`data/emails.json` 是一个数组，每条记录字段：

```json
{
  "id": "email_001",
  "subject": "项目进度同步",
  "sender": "alice@example.com",
  "recipients": ["bob@example.com"],
  "date": "2026-04-15 10:30:00",
  "body": "邮件正文...",
  "labels": ["工作", "项目"],
  "thread_id": "thread_001"
}
```

`labels`、`thread_id` 可选；其余字段为必填。

## 关键配置项（`.env`）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | — | 必填 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 本地嵌入模型 |
| `EMBEDDING_DEVICE` | `cpu` | 改为 `cuda` 启用 GPU |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 | 切块参数 |
| `TOP_K` | 5 | 初检数量 |
| `VECTOR_WEIGHT` / `BM25_WEIGHT` | 0.7 / 0.3 | 融合权重 |
| `RERANK_TOP_N` | 3 | 重排后保留数 |

## 设计要点

- **每模块单文件，分层清晰**：core / agents / api / frontend 各自独立。
- **首次运行会下载嵌入模型**（约 500MB），之后离线可用，无嵌入 API 调用成本。
- **重排失败有兜底**：LLM 评分异常时回退到原始顺序。
- **意图识别失败有兜底**：异常时回退到 `general`，由 RetrieverAgent 处理。
- **检索后过滤**：RetrieverAgent 提取的发件人过滤若清空结果会自动回退。
