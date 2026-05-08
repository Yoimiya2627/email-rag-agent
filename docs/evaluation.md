# 评测方法与结论

> 这份文档记录 Email RAG Agent 的离线评测设计、6 版消融数据，以及把"哪一版最好"翻译成"按业务目标如何取舍"的过程。
>
> 简短版结论：在本数据集下，三个维度的赢家是分散的（V2 / V4 / V5），不存在"全维度都最优"的版本。把指标和延迟一起看，**默认推荐 V2（向量 + BM25 + RRF）**——relevancy 最高、延迟最低，性价比最优；只有强合规场景才有理由付出 10× 延迟换 V4 的 faithfulness 优势。

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

| 版本 | BM25 | RRF | Reranker | Rewrite | answer_relevancy | faithfulness | context_precision | RAG 主链路端到端延迟（采样观察值） |
|---|---|---|---|---|---|---|---|---|
| V1 | ❌ | ❌ | ❌ | ❌ | 0.8983 | 0.8683 | 0.5077 | ~3 s |
| **V2** | ✅ | ✅ | ❌ | ❌ | **0.9517** 🏆 | 0.9117 | 0.5543 | ~3 s |
| V3 | ✅ | ✅ | ✅ | ❌ | 0.9227 | 0.9050 | 0.5787 | ~35 s |
| **V4** | ✅ | ✅ | ✅ | ✅ | 0.9117 | **0.9233** 🏆 | 0.5500 | ~38 s |
| **V5** | ✅ | ❌ | ✅ | ✅ | 0.8700 | 0.8517 | **0.6383** 🏆 | ~36 s |
| V6 | ✅ | ✅ | ❌ | ✅ | 0.9400 | 0.9117 | 0.5533 | ~5 s |

> **延迟口径说明**：度量的是 **RAG 主链路** —— `query rewrite → filter 抽取 → hybrid_search → rerank → generate`，不包含 Coordinator 意图分类、Self-RAG 反思循环、SSE 前端渲染、HTTP 请求链路。单题平均挂钟时间，采样观察值，非正式 benchmark。基准脚本见 `scripts/measure_latency.py`（每版默认 5 题，输出 mean/median/p95）；上面表格里的数字是开发期手测，待用脚本跑完正式数据回填。绝对值会随 DeepSeek API 网络抖动浮动 ±20%，但版本间相对差距是稳定的。

### 单维度归因（看着都对）

把每一项做成单维度故事，每个都自洽：

- **加 BM25 + RRF（V1 → V2）**：relevancy 0.898 → 0.952、faithfulness 0.868 → 0.912、precision 0.508 → 0.554，**三维全涨**。混合检索弥补向量在术语/数字 query 上的短板，延迟几乎不变（BM25 是 in-memory，毫秒级）。
- **加 Reranker（V2 → V3）**：precision 0.554 → 0.579，**精度涨了**。Reranker 把检索池从 top_k×4 收敛到 top_n，过滤无关 chunk。
- **加 Query Rewrite（V3 → V4）**：faithfulness 0.905 → 0.923，**有据性涨了**。改写后的 query 更接近文档表述，retrieval 召回的 chunk 与 answer 更对齐。
- **关掉 RRF 改简单合并（V4 → V5）**：precision 0.550 → 0.638，**precision 反而涨了**。BM25 在 testset 这种"术语精准"的题上比 RRF 平均化后更准（可参考第 4 节）。

每条单看都讲得通。问题在于把它们叠到一起。

---

## 4. 三个反直觉发现

把 4 个组件全打开（V4，"全开版"）的预期是"三维全赢"，实际跑出来 **三个维度的赢家分散在 V2 / V4 / V5**——没有任何一版同时拿下三冠。

### 发现 1：reranker 不一定提 relevancy

V2 → V3 加了 reranker，relevancy 反而 0.952 → 0.923。原因猜测：本项目的 reranker 是 LLM-based 实现（用 DeepSeek 给候选 chunk 打分，方差和延迟都偏高，更适合替换为 cross-encoder 比如 `bge-reranker-v2-m3`），它把 V2 已经排前的"边相关 + 直接命中" chunk 挤出 top_n，反而让生成器丢掉关键信息。

### 发现 2：rewrite 提 faithfulness，但代价是 relevancy

V3 → V4 加了 rewrite，faithfulness 0.905 → 0.923 **是涨了**，但 relevancy 0.923 → 0.912 **是跌的**。改写过的 query 更"文档化"，召回的 chunk 与生成器的引用更贴合（faithfulness 涨）；但改写引入语义漂移，离用户原始意图更远了（relevancy 跌）。

### 发现 3：RRF 不总是赢

V4 → V5 把 RRF 关掉改成简单分数合并，precision 0.550 → 0.638 是 6 版里最高。RRF 用 rank 平均化，对"两路都中等"的友好；简单合并在"BM25 强势"的题上让 BM25 直接赢，反而过滤掉了向量带来的边缘 chunk。本数据集里术语精准题占多数，BM25 单独发挥更准。**换数据集（语义模糊题占多数）这个结论会反转**。

---

## 5. 业务选型

把上面的"赢家分散"翻译成业务取舍：

| 业务关心什么 | 推荐版本 | 理由 |
|---|---|---|
| 答案切题（客服 FAQ、邮件查询、对话式查询） | **V2**（BM25 + RRF） | relevancy 0.952 最高，延迟 ~3 s 最低，**比 V4 快 10 倍且 relevancy 还更好** |
| 答案有据（合规审计、法律咨询、医疗助理） | V4（全开） | faithfulness 0.923 最高；要接受 ~38 s 延迟，**只有"宁慢勿错"的场景才值得付** |
| top-K 精确（人工复核场景，每条都要看） | V5（V4 - RRF） | precision 0.638 最高；适合"先粗筛再人审"的工作流 |

### 默认上线选 V2

业务选型最有说服力的一刻：**V2 比 V4 延迟低一个数量级，relevancy 还更高。**

V4 唯一赢 V2 的地方是 faithfulness（0.923 vs 0.912，差 0.011），代价是 ~35 s 额外延迟（reranker LLM 调用）。这个差是否值得，完全取决于业务能不能接受用户等 38 s——大多数对话式应用不能，所以默认上 V2。

V6（V4 去掉 reranker）也值得记一下：relevancy 0.940 接近 V2，faithfulness 0.912 接近 V2，延迟 ~5 s，是"加了 query rewrite 但不肯付 reranker 延迟"的中间档。

---

## 6. 已知局限

1. **30 题样本，方差较大**：正式 benchmark 应跑 100 题 × 3 次取均值；当前数据足以做相对排序，绝对值不要外推。
2. **合成数据偏向"单邮件可答"**：testset 由 LLM 基于单封邮件生成 QA 对，跨邮件多跳推理题占比低。这是 rewrite 在合成数据上 ROI 偏低的可能原因之一；真实邮箱（多线程对话、跨邮件主题串联）上 rewrite 价值可能反转。
3. **缺 context_recall**：合成 testset 没有人工标注的"应召回 chunk id"，无法测召回率。
4. **LLM-based reranker 的工程取舍**：当前实现用 DeepSeek 给候选 chunk 打分，方差和延迟都偏高。换 cross-encoder（`bge-reranker-v2-m3`）能拿到毫秒级 + 确定性打分，预计 V3/V4 的指标和延迟会同时受益——这是当前评测里 ROI 最明显的下一步优化。
5. **延迟是采样观察值**：见第 3 节注释。基准脚本 `scripts/measure_latency.py` 已就位（每版 5 题，含 mean/median/p95），但当前文档里的延迟列还没用它的输出回填——下次正式 benchmark 时一并刷新。

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
