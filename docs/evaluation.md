# 评测方法与结论

> 这份文档记录 Email RAG Agent 的离线评测设计、V1-V7 消融数据，以及把"哪一版最好"翻译成"按业务目标如何取舍"的过程。
>
> 简短版结论：在本数据集下，三个维度仍然体现出取舍关系。V7（BM25 + RRF + Cross-Encoder reranker）拿到最高 `answer_relevancy=0.9750` 和 `faithfulness=0.9267`，旧 LLM reranker V3 仍拿到最高 `context_precision=0.7147`。把指标、延迟和稳定性一起看，**默认推荐 V2（向量 + BM25 + RRF，`ENABLE_RERANKER=false`）**；当业务更重视高质量回答且能接受本地 rerank 成本时，再显式开启 V7。

---

## 1. 评测方法

### 三个维度

| 维度 | 含义 | 衡量谁 |
|---|---|---|
| `answer_relevancy` | 答案与问题的相关程度（1.0 = 完全切题） | **生成器** 是否答对题 |
| `faithfulness` | 答案是否完全有上下文依据（1.0 = 没幻觉） | **生成器** 是否凭空发挥 |
| `context_precision` | 检索片段中真正有用的比例（1.0 = 全部命中） | **检索器** 噪声多不多 |

三个维度衡量的是 RAG 流水线的不同部位，所以指标互相之间不会有"一个赢则其他都赢"的关系——这是后面"赢家分散"现象的根本原因。

### 自实现 vs 官方 ragas 包

没有用 `ragas` 官方包，原因有两个：

1. **推理模型不兼容**：项目用 DeepSeek `deepseek-v4-flash`（推理模型），输出 `reasoning_content` + `content` 双字段，max_tokens 不够时 `content` 为空但 `reasoning_content` 里实际包含了 JSON 评分。官方 ragas 包按 OpenAI 标准 schema 解析，识别不出这种情况，会判 LLM 调用失败。
2. **降级路径需要显式控制**：希望 LLM 打分失败时能确定性地降级到向量相似度打分（而不是直接返回 NaN），保证评测结果可复现。

实现见 `scripts/run_ragas_eval.py`：
- LLM 打分（主路径）：用 DeepSeek 给出三维度 0-1 分数，3 次重试，max_tokens=3000 给推理过程留足预算，content 为空时从 `reasoning_content` 兜底解析 JSON。
- 向量打分（降级路径）：用 bge-m3 嵌入算 query/answer/contexts 的余弦相似度，对应到三个维度。**降级是兜底，不是默认**——只有 LLM 三次都失败才走这条路。

### 缺什么

没有实现 `context_recall`（检索是否覆盖了所有应该召回的内容），因为合成 testset 没有人工标注的 ground-truth chunk id。这是已知的局限；如果接入真实邮箱 + 人工标注的金标，应该补上。

---

## 2. 测试集

- **来源**：`data/ragas_testset.json`，100 题
- **生成方式**：LLM 基于 5000 封合成邮件生成"问题 + 标准答案 + 引用邮件 ID"三元组（脚本 `scripts/generate_ragas_data.py`）
- **每版评测取前 30 题**（`--limit 30`）
- **方差较大是已知简化**：30 题样本下单题分数波动会被放大；正式上线前应跑全量 100 题 × 多次取均值

### 题目类型示例

```
Q1 OKR 回顾会议定于什么时间、在哪个地点举行？     → 单邮件查事实
根据 2025 年 12 月资金使用情况报告，市场部为什么能节省 20% 的预算？  → 单邮件查原因
周杰在代码审查中指出了 order-service 模块的哪三个主要问题？   → 单邮件查列表
```

100 题里以"单邮件可答"题型为主——这一点会反过来影响 reranker 的 ROI 判断（见第 4 节）。

---

## 3. 7 版配置 × 三指标 × 延迟

每版评测的具体源数据：`data/eval_results/V{1..7}.json`（含 30 条逐题记录）；汇总：`data/eval_results/comparison.json`。延迟表目前来自 `data/eval_results/latency.json` 的 V1-V6 历史重跑；V7 已完成质量评测，后续需单独补 Cross-Encoder latency benchmark。

| 版本 | BM25 | RRF | Reranker | Backend | Rewrite | answer_relevancy | faithfulness | context_precision | 端到端延迟 mean / p95（s） |
|---|---|---|---|---|---|---|---|---|---|
| V1 | ❌ | ❌ | ❌ | llm | ❌ | 0.8667 | 0.9233 | 0.5937 | 7.6 / 33.3 |
| **V2** | ✅ | ✅ | ❌ | llm | ❌ | 0.9567 | 0.9000 | 0.5713 | **8.7 / 14.4** |
| **V3** | ✅ | ✅ | ✅ | llm | ❌ | 0.9333 | 0.9017 | **0.7147** | 20.3 / 30.5 |
| V4 | ✅ | ✅ | ✅ | llm | ✅ | 0.9533 | 0.8783 | 0.6427 | 24.2 / 38.0 |
| V5 | ✅ | ❌ | ✅ | llm | ✅ | 0.9467 | 0.9083 | 0.6147 | 24.4 / 34.8 |
| V6 | ✅ | ✅ | ❌ | llm | ✅ | 0.9600 | 0.8967 | 0.6050 | 11.0 / 14.8 |
| **V7** | ✅ | ✅ | ✅ | cross_encoder | ❌ | **0.9750** | **0.9267** | 0.6103 | 待补 |

> **延迟口径说明**：度量的是 **RAG 主链路** —— `query rewrite → filter 抽取 → hybrid_search → rerank → generate`，不包含 Coordinator 意图分类、Self-RAG 反思循环、SSE 前端渲染、HTTP 请求链路。数字是 **10 题/版** 跑出的 **trimmed mean**（去掉最高最低后均值，抗 API 抖动）和 **p95**，源数据 `data/eval_results/latency.json`，复现脚本 `scripts/measure_latency.py --limit 10`。**由于每版仅 10 题，p95 在该样本量下接近最大值**，主要用于观察尾部延迟风险，正式 benchmark 应跑 30+ 题让 p95 真正反映 95 分位。V1 p95 33s 是冷启动（首题模型加载）造成的离群点，trimmed mean 7.6s 是稳态。绝对值会随 DeepSeek API 抖动浮动 ±20%，但版本间相对差距稳定。

### 逐组件看：单个 delta 的方向并不稳定

很容易给每个组件配一个"单维度故事"——加 BM25+RRF 让某指标涨、加 reranker 让 precision 涨之类。但把这套消融**跑两次对比**就会发现：**单个组件对单个指标的影响方向，并不都稳定**。比如"加 query rewrite 对 faithfulness 是正还是负"，两次跑给出的方向就不一致。

能稳的只有两类信息：① **延迟差异**（旧 LLM reranker / rewrite 各引入一次 LLM 调用，这是确定的——见上表延迟列，LLM reranker 单组件就吃掉 ~12s；Cross-Encoder 已把 V7 的重排改成本地模型调用，但仍需补正式 latency benchmark）；② **跨多次都成立的模式**（见 §4）。单个 RAGAS delta 的正负不要过度解读——n=30 + LLM 打分的方差足以翻转它。

---

## 4. 核心发现：没有"全场最优"，但要小心噪声

### 稳健的发现：没有单一全场最优

把 4 个组件全打开（V4）本以为能"三维全赢"，实际不是；接入 Cross-Encoder 后，V7 拿到 `answer_relevancy` 和 `faithfulness` 双高，但 `context_precision` 仍低于旧 LLM reranker V3。更稳的结论是：**没有一个版本同时统治回答质量、上下文干净度、延迟和稳定性**。

| | relevancy 最优 | faithfulness 最优 | context_precision 最优 |
|---|:---:|:---:|:---:|
| 旧 V1-V6 重跑 | V6 | V1 | V3 |
| 新增 V7 后 | V7 | V7 | V3 |

所以这里能下的稳健结论不是"哪个版本永远第一"，而是**组件之间存在 trade-off**：Cross-Encoder 更利于回答相关性和有据性；旧 LLM reranker 在这组样本上更利于 top-K 干净度；默认链路还要把延迟和失败面算进去。

### 为什么具体排名会变：评测本身有噪声

数字是单次、每版 30 题、LLM 当裁判跑出来的。n=30 是小样本，LLM 打分本身也有波动——两者叠加，足以让版本间 ±0.05~0.1 的差异翻转。**所以本文所有 RAGAS 数字都应当作"有噪声的估计"，4 位小数不代表 4 位精度。** 要更稳的结论需要加大题量（→100）并多次取均值。

### 方向性观察（趋势可参考，精确数值不必较真）

- **reranker：拿精度、但有不同 backend 取舍**。V2→V3 加旧 LLM reranker，`context_precision` 上升到 0.7147，但代价是额外 LLM 延迟和评分方差；V7 改为 Cross-Encoder 后，`answer_relevancy` 和 `faithfulness` 最高，`context_precision` 比 V2 提升但低于 V3。正确结论不是"某个 reranker 全赢"，而是按业务目标在 top-K 干净度、回答质量、延迟和稳定性之间取舍。
- **query rewrite：在规整的合成数据上 ROI 不明显**。加 rewrite 对三个指标互有增减、不构成稳定净增益。合成 testset 口语化程度低，改写能补的信息有限；真实邮箱（口语化、跨邮件）上 rewrite 价值可能不同。

---

## 5. 业务选型

把"赢家分散"翻译成业务取舍——下面的推荐结合**指标梯队 + 延迟**给出，不依赖某一版精确排第几（精确排名有噪声）：

| 业务关心什么 | 推荐版本 | 理由 |
|---|---|---|
| 答案切题（客服 FAQ、邮件查询、对话式查询） | **V2 默认 / V7 高质量模式** | V7 relevancy 最高；但默认链路仍优先 V2，因为 `ENABLE_RERANKER=false` 延迟最低、稳定性最好。质量优先且能接受本地 rerank 成本时再开 V7 |
| top-K 精确（人工复核场景，每条都要看） | V3 作为高 precision 对照 / V7 作为工程化重排 | V3 的 `context_precision=0.7147` 最高，但依赖额外 LLM scorer；V7 更确定、更适合产品化重排，但本次 precision 未超过 V3 |
| 答案有据（合规、审计） | V7 或 V2，看延迟预算 | V7 faithfulness 最高；如果业务不接受 rerank 成本，V2 仍是低延迟默认路径 |

### 默认上线选 V2

V2 的理由不靠"它精确排第一"，而靠两个**够稳的**事实：① 不带 reranker/rewrite，链路最短、失败面最小；② 历史 latency benchmark 中 mean 8.7s，是当前有正式延迟数据的最低延迟路径之一。V7 已证明 Cross-Encoder 在质量上值得接入，但默认是否打开还需要补 latency benchmark 和业务 SLA 取舍。组件不是越多越好。

V6（V4 去掉 reranker）也值得记一下：延迟 mean 11s，跟 V2 几乎平齐，三维指标也都接近——说明 reranker 那 ~12s 是 V3/V4 延迟的主要来源。

---

## 6. 已知局限

1. **30 题样本，方差较大**：正式 benchmark 应跑 100 题 × 3 次取均值。当前数据足以揭示"赢家分散"这个**模式**，但版本间精确排名会随重跑变动（§4 给了两次跑的对比），不要把单次排名当定论。
2. **合成数据偏向"单邮件可答"**：testset 由 LLM 基于单封邮件生成 QA 对，跨邮件多跳推理题占比低。这是 rewrite 在合成数据上 ROI 偏低的可能原因之一；真实邮箱（多线程对话、跨邮件主题串联）上 rewrite 价值可能反转。
3. **缺 context_recall**：合成 testset 没有人工标注的"应召回 chunk id"，无法测召回率。
4. **Cross-Encoder 已接入但延迟 benchmark 未补齐**：旧 LLM reranker 的延迟和方差问题已经通过 V7 的 `RERANKER_BACKEND=cross_encoder` 路径缓解；下一步要补同口径 latency benchmark、真实邮箱 gold chunk 标注和 `context_recall`，再决定是否默认开启 reranker。
5. **延迟样本仍偏小**：当前数字是 10 题/版的 trimmed mean + p95，足以做相对排序；正式上线前应跑 30+ 题 × 多次取均值，并在不同时段重复以观测 API 抖动方差。源数据 `data/eval_results/latency.json`，复现 `python scripts/measure_latency.py --limit 10`。

---

## 7. 复现实验

```bash
# 跑全部 7 版（默认 30 题 × 7 版，含 LLM reranker 和 Cross-Encoder 版本）
python scripts/run_ragas_eval.py

# 只跑指定版本
python scripts/run_ragas_eval.py --versions V1,V2,V7 --limit 30

# 输出
data/eval_results/V{1..7}.json     # 每版逐题记录（含 answer / contexts）
data/eval_results/comparison.json  # 三维度均值汇总
```

评测脚本会自动应用每版的 `ENABLE_*` flag、重置 reranker 熔断器（避免上一版的失败计数泄漏到下一版，详见 [`docs/technical_retrospective.md`](technical_retrospective.md) §4）、把 LLM 打分失败的样本降级到向量相似度。

---

## 8. V7：Cross-Encoder Reranker 更新

### 为什么新增 V7

V1-V6 里的 `ENABLE_RERANKER=true` 使用的是 LLM scorer：把多个候选 chunk 拼成 prompt，让 DeepSeek 输出 `{"scores": [...]}`。这个方案能做语义重排，但工程上有三个问题：

1. **延迟高**：每次 rerank 都多一次 LLM 调用，V3/V4/V5 的端到端耗时明显高于 V2/V6。
2. **方差高**：LLM scorer 受输出格式、reasoning token、API 抖动影响，虽然已有 JSON 兜底和熔断，但仍不是检索链路里最稳定的部件。
3. **成本不适合默认路径**：默认对话路径更需要稳定低延迟；重排如果要常开，更适合本地 Cross-Encoder。

V7 保留 V2 的 `BM25 + RRF` 基线，只把 reranker backend 切到 `cross_encoder`，用于隔离观察 Cross-Encoder 的贡献。

### V1-V7 最新 30 题结果

| Version | BM25 | RRF | Reranker | Backend | Rewrite | answer_relevancy | faithfulness | context_precision |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| V1 | false | false | false | llm | false | 0.8667 | 0.9233 | 0.5937 |
| V2 | true | true | false | llm | false | 0.9567 | 0.9000 | 0.5713 |
| V3 | true | true | true | llm | false | 0.9333 | 0.9017 | **0.7147** |
| V4 | true | true | true | llm | true | 0.9533 | 0.8783 | 0.6427 |
| V5 | true | false | true | llm | true | 0.9467 | 0.9083 | 0.6147 |
| V6 | true | true | false | llm | true | 0.9600 | 0.8967 | 0.6050 |
| V7 | true | true | true | cross_encoder | false | **0.9750** | **0.9267** | 0.6103 |

### 解读

- V7 在本次 30 题重跑里拿到最高 `answer_relevancy` 和 `faithfulness`，说明 Cross-Encoder 把更适合生成回答的 chunk 推到了前面。
- `context_precision` 从 V2 的 0.5713 提升到 0.6103，但没有超过旧 LLM reranker V3 的 0.7147。面试里不要说“Cross-Encoder 全面碾压”，更准确的说法是：**Cross-Encoder 让重排从 LLM 调用变成本地确定性模型调用，提升了回答相关性和有据性，同时把成本/稳定性拉回可控范围；如果业务目标是人工复核 top-K 干净度，旧 V3 仍是一个高精度对照组。**
- 本次 V7 运行时发现 `HF_ENDPOINT=https://hf-mirror.com` 对 `BAAI/bge-reranker-v2-m3` 拉取不稳定；正式复现命令使用 `HF_ENDPOINT=https://huggingface.co`，并已把 `.env.example` 更新为官方端点优先。

复现：

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
.\.venv\Scripts\python.exe scripts\run_ragas_eval.py --versions V7 --limit 30 --output data\eval_results\comparison_v7.json
```

正式汇总文件：

- `data/eval_results/V7.json`
- `data/eval_results/comparison.json`
