# 评测方法与结论

> 这份文档记录 Email RAG Agent 的离线评测设计、6 版消融数据，以及把"哪一版最好"翻译成"按业务目标如何取舍"的过程。
>
> 简短版结论：在本数据集下，三个维度的赢家是分散的——不存在"全维度都最优"的版本。**这套消融跑过两次，"赢家分散"这个模式稳定复现，但具体哪一版赢哪个指标在两次之间会变（见 §4），所以下面的数字应当作有噪声的估计。** 把指标和延迟一起看，**默认推荐 V2（向量 + BM25 + RRF）**——relevancy 处于第一梯队，且不带 reranker/rewrite、延迟最低。

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

## 3. 6 版配置 × 三指标 × 延迟

每版评测的具体源数据：`data/eval_results/V{1..6}.json`（含 30 条逐题记录）；汇总：`data/eval_results/comparison.json`。

| 版本 | BM25 | RRF | Reranker | Rewrite | answer_relevancy | faithfulness | context_precision | 端到端延迟 mean / p95（s） |
|---|---|---|---|---|---|---|---|---|
| **V1** | ❌ | ❌ | ❌ | ❌ | 0.8667 | **0.9233** 🏆 | 0.5937 | 7.6 / 33.3 |
| V2 | ✅ | ✅ | ❌ | ❌ | 0.9567 | 0.9000 | 0.5713 | 8.7 / 14.4 |
| **V3** | ✅ | ✅ | ✅ | ❌ | 0.9333 | 0.9017 | **0.7147** 🏆 | 20.3 / 30.5 |
| V4 | ✅ | ✅ | ✅ | ✅ | 0.9533 | 0.8783 | 0.6427 | 24.2 / 38.0 |
| V5 | ✅ | ❌ | ✅ | ✅ | 0.9467 | 0.9083 | 0.6147 | 24.4 / 34.8 |
| **V6** | ✅ | ✅ | ❌ | ✅ | **0.9600** 🏆 | 0.8967 | 0.6050 | 11.0 / 14.8 |

> **延迟口径说明**：度量的是 **RAG 主链路** —— `query rewrite → filter 抽取 → hybrid_search → rerank → generate`，不包含 Coordinator 意图分类、Self-RAG 反思循环、SSE 前端渲染、HTTP 请求链路。数字是 **10 题/版** 跑出的 **trimmed mean**（去掉最高最低后均值，抗 API 抖动）和 **p95**，源数据 `data/eval_results/latency.json`，复现脚本 `scripts/measure_latency.py --limit 10`。**由于每版仅 10 题，p95 在该样本量下接近最大值**，主要用于观察尾部延迟风险，正式 benchmark 应跑 30+ 题让 p95 真正反映 95 分位。V1 p95 33s 是冷启动（首题模型加载）造成的离群点，trimmed mean 7.6s 是稳态。绝对值会随 DeepSeek API 抖动浮动 ±20%，但版本间相对差距稳定。

### 逐组件看：单个 delta 的方向并不稳定

很容易给每个组件配一个"单维度故事"——加 BM25+RRF 让某指标涨、加 reranker 让 precision 涨之类。但把这套消融**跑两次对比**就会发现：**单个组件对单个指标的影响方向，并不都稳定**。比如"加 query rewrite 对 faithfulness 是正还是负"，两次跑给出的方向就不一致。

能稳的只有两类信息：① **延迟差异**（reranker / rewrite 各引入一次 LLM 调用，这是确定的——见上表延迟列，reranker 单组件就吃掉 ~12s）；② **跨两次都成立的模式**（见 §4）。单个 RAGAS delta 的正负不要过度解读——n=30 + LLM 打分的方差足以翻转它。

---

## 4. 核心发现：没有"全场最优"，但要小心噪声

### 稳健的发现：赢家分散

把 4 个组件全打开（V4）本以为能"三维全赢"，实际不是——**三个维度的最优版本互不相同**。而且这套消融**跑过两次**：

| | relevancy 最优 | faithfulness 最优 | context_precision 最优 |
|---|:---:|:---:|:---:|
| 第一次 | V2 | V4 | V5 |
| 第二次（本文表格） | V6 | V1 | V3 |

两次的具体赢家**完全不同**，但**"三个指标被三个不同配置瓜分"这个模式，两次都成立**。所以这里能下的稳健结论是这个**模式**——不存在同时拿下三冠的配置、组件之间是 trade-off——而不是"V6 的 relevancy 最强"这种精确排名。

### 为什么具体排名会变：评测本身有噪声

数字是单次、每版 30 题、LLM 当裁判跑出来的。n=30 是小样本，LLM 打分本身也有波动——两者叠加，足以让版本间 ±0.05~0.1 的差异翻转。**所以本文所有 RAGAS 数字都应当作"有噪声的估计"，4 位小数不代表 4 位精度。** 要更稳的结论需要加大题量（→100）并多次取均值。

### 方向性观察（趋势可参考，精确数值不必较真）

- **reranker：拿精度、折切题度**。V2→V3 加 reranker，context_precision 上升（在剔检索噪声），代价是 answer_relevancy 略降——它把 V2 已排前的"直接命中" chunk 挤出 top_n。但本项目的 reranker 是 LLM 实现，方差和延迟都偏高（单组件吃掉 ~12s）；更合适的是换 cross-encoder（`bge-reranker-v2-m3`），毫秒级且确定性。
- **query rewrite：在规整的合成数据上 ROI 不明显**。加 rewrite 对三个指标互有增减、不构成稳定净增益。合成 testset 口语化程度低，改写能补的信息有限；真实邮箱（口语化、跨邮件）上 rewrite 价值可能不同。

---

## 5. 业务选型

把"赢家分散"翻译成业务取舍——下面的推荐结合**指标梯队 + 延迟**给出，不依赖某一版精确排第几（精确排名有噪声）：

| 业务关心什么 | 推荐版本 | 理由 |
|---|---|---|
| 答案切题（客服 FAQ、邮件查询、对话式查询） | **V2**（BM25 + RRF） | relevancy 处于第一梯队（与最高的 V6 基本持平）；延迟 mean 8.7s，**不带 reranker/rewrite，约为 V4 的 1/3** |
| top-K 精确（人工复核场景，每条都要看） | V3（+Reranker） | context_precision 明显领先——这一维版本间差异够大、不只是噪声；代价是延迟升到 mean 20s |
| 答案有据（合规、审计） | 看情况 | faithfulness 在 6 版间差异很小（都在 ~0.88–0.92），基本被噪声覆盖——为这一维专门选版本意义不大，按延迟和其它维度选即可 |

### 默认上线选 V2

V2 的理由不靠"它精确排第一"，而靠两个**够稳的**事实：① relevancy 在第一梯队（和最高的 V6 差距在噪声范围内）；② **不带 reranker/rewrite，延迟最低**（mean 8.7s，约为全开 V4 的 1/3）。大多数对话式应用扛不住 V4 那 mean 24s 的等待，所以默认上 V2。组件不是越多越好。

V6（V4 去掉 reranker）也值得记一下：延迟 mean 11s，跟 V2 几乎平齐，三维指标也都接近——说明 reranker 那 ~12s 是 V3/V4 延迟的主要来源。

---

## 6. 已知局限

1. **30 题样本，方差较大**：正式 benchmark 应跑 100 题 × 3 次取均值。当前数据足以揭示"赢家分散"这个**模式**，但版本间精确排名会随重跑变动（§4 给了两次跑的对比），不要把单次排名当定论。
2. **合成数据偏向"单邮件可答"**：testset 由 LLM 基于单封邮件生成 QA 对，跨邮件多跳推理题占比低。这是 rewrite 在合成数据上 ROI 偏低的可能原因之一；真实邮箱（多线程对话、跨邮件主题串联）上 rewrite 价值可能反转。
3. **缺 context_recall**：合成 testset 没有人工标注的"应召回 chunk id"，无法测召回率。
4. **LLM-based reranker 的工程取舍**：当前实现用 DeepSeek 给候选 chunk 打分，方差和延迟都偏高。换 cross-encoder（`bge-reranker-v2-m3`）能拿到毫秒级 + 确定性打分，预计 V3/V4 的指标和延迟会同时受益——这是当前评测里 ROI 最明显的下一步优化。
5. **延迟样本仍偏小**：当前数字是 10 题/版的 trimmed mean + p95，足以做相对排序；正式上线前应跑 30+ 题 × 多次取均值，并在不同时段重复以观测 API 抖动方差。源数据 `data/eval_results/latency.json`，复现 `python scripts/measure_latency.py --limit 10`。

---

## 7. 复现实验

```bash
# 跑全部 6 版（默认 30 题 × 6 版 ≈ 30-60 分钟，含 reranker 版本）
python scripts/run_ragas_eval.py

# 只跑指定版本
python scripts/run_ragas_eval.py --versions V1,V2 --limit 30

# 输出
data/eval_results/V{1..6}.json     # 每版逐题记录（含 answer / contexts）
data/eval_results/comparison.json  # 三维度均值汇总
```

评测脚本会自动应用每版的 `ENABLE_*` flag、重置 reranker 熔断器（避免上一版的失败计数泄漏到下一版，详见 [`docs/technical_retrospective.md`](technical_retrospective.md) §4）、把 LLM 打分失败的样本降级到向量相似度。
