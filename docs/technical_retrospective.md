# 技术复盘

> 这份文档记录项目从"能跑"到"跑稳并能解释"过程中遇到的 5 个工程问题。每条按 **现象 → 排查 → 根因 → 修复 → 教训** 展开，对应到具体 commit。
>
> 完整版（含其他次要问题、调试方法论沉淀、待办识别）见 [`engineering_pitfalls.md`](engineering_pitfalls.md)。

---

## 1. 推理模型 max_tokens 陷阱

**现象**：跑 RAGAS 评测，三个维度全部返回 `0.0`。日志显示 LLM 三次重试都失败，`choices[0].message.content` 是空字符串。但 HTTP 是 200，没报错。

**排查**：先怀疑 prompt，调 system message / temperature / `max_tokens=256` 都无效。后写独立 probe `scripts/debug_deepseek_score.py`，绕过业务代码、直接打印 raw response：`finish_reason='length'`、`completion_tokens=256` 顶到上限、`reasoning_content` 里却有完整推理过程。

**根因**：DeepSeek `deepseek-v4-flash` 是推理模型，返回 `content` + `reasoning_content` 两个字段，**`max_tokens` 同时覆盖两者**。原代码 5 处调用都按 GPT-3.5 的习惯写了 `128~256`，推理模型的几百 token 推理过程吃光预算，content 永远输出不出来。

**修复**：普通调用 `max_tokens=1500`、高推理量（rerank / 三维度评分）`3000`；content 仍空时从 `reasoning_content` 抽取最后一个 JSON 对象兜底。涉及 `score_response / _rewrite_query / _extract_filters / classify_intent / grade_contexts / rerank` 共 6 个调用点（commit `4d27325` → `b0fc749`）。

**教训**：从 chat 模型切到推理模型时，`max_tokens` 语义变了。LLM 调用失败的第一反应不该是改 prompt——先看 `finish_reason` 和 `usage.completion_tokens`：`length` + 顶到上限 = max_tokens 问题；非空答非所问 = prompt 问题；`refusal` 非空 = 模型拒答。三种处理完全不同。

---

## 2. SSE 假流式

**现象**：`/chat/stream` 端点声称 SSE 流式，前端开"流式输出"开关后用户体感和非流式一样——等十几秒所有 token 一次性出现。

**排查**：抓 SSE 事件时间戳，发现所有事件 `Last-Event-Time` 几乎相同，`first→last spread ≈ 0` —— 服务端攒齐了再一次性 flush。

**根因**：API 层一行 `tokens = list(stream_generate(...))`。`list(generator)` 会 eager 消费整个生成器、等所有 token 完才返回。等于先把回答全部存到内存、再"假装"逐个 yield 给前端。OpenAI Python SDK 的 stream 是**同步阻塞迭代器**（`__iter__`），不是 async iterator，所以不能直接 `async for`。

**修复**（commit `59f9898`）：用 `asyncio.Queue` + 生产者线程桥接同步 SDK 与 asyncio。生产者线程跑阻塞 `for token in stream_generate(...)`，每个 token 通过 `loop.call_soon_threadsafe(queue.put_nowait, token)` 跨线程投递；消费者协程 `await queue.get()` 逐个 yield SSE 事件。`call_soon_threadsafe` 是跨线程往事件循环投递任务的唯一安全方式——asyncio 队列本身不是线程安全的。

**教训**：`list()` / `tuple()` 包生成器是隐式的"流式杀手"，code review 时要单独留意。同步 SDK + 异步框架混用时，三个候选方案（换 AsyncSDK / 全同步 / Queue 桥接）选 Queue 桥接因为侵入最小——但代价是要懂跨线程投递的细节。

---

## 3. BM25 懒加载缓存（兼防 cache stampede）

**现象**：5000 邮件后，BM25 检索单次 800ms+，是检索环节最慢一步。索引建好后大部分时间是重复劳动——每次查询都重新做"全量 ChromaDB 读 → 全量分词 → 建 Okapi 索引"。

**排查**：分段计时定位耗时分布——ChromaDB 全量读 ~400ms，分词 + 建索引 ~440ms。两段都是"语料没变就不该重做"的纯粹缓存目标。

**根因 + 设计取舍**：第一版的 cache key 想用"语料 hash"——但要算 hash 就得先读全量数据，缓存就白做了。改用 ChromaDB 的 `collection.count()` 做 cache key：O(1)、不读数据，能覆盖"邮件数量没变 = 索引可复用"这个最常见的不变式。

**修复**（commit `8cc5b6e`）：`_bm25_cache: (count, BM25Okapi, chunks, corpus)` 模块级单例，外加 `threading.Lock` 保护**整个 check-then-rebuild 块**——这一步关键。如果不加锁，并发首次冷启动时多个线程都会 miss → 都去重建一次（cache stampede），白白做几次 800ms 的工作。`/index` 和 `/index/clear` 端点显式调 `invalidate_bm25_cache()` 主动失效。

**教训**：cache key 选择要避免"为了算 key 先付一遍 cost"。锁要包住"查 + 建"整段而不只是"建"——否则防不住 stampede。规模化（百万级文档）时 in-memory BM25 不现实，需要换 Elasticsearch；当前缓存是 5000~10 万规模下的对症措施。

实测：冷启动 846ms → 缓存命中 22ms（40× 提速）。

---

## 4. 熔断器跨版本污染（被低估的工程教训）

**现象**：跑 V1→V6 ablation 时，V3 的 reranker 偶发失败会**永久关闭** V4/V5/V6 的 reranker——即使 LLM 已经恢复正常。日志里看到的是 `ENABLE_RERANKER=True`，实际跑的是 `circuit breaker open` 分支返回原始排序。结果是 V4-V6 三版本拿到的对比数据完全失真。

**排查**：发现 V4 跑出来的 precision 反常地低、和 V3 几乎一样。打开 reranker 内部状态打印，看到 `_consecutive_failures=4 > _FAILURE_THRESHOLD=3`——熔断器从 V3 跑到 V4 没重置。

**根因**：`core/reranker.py` 的 `_consecutive_failures` 是 **module-level 全局变量**。V1~V6 共享同一个 Python 进程、同一个 module 对象——计数器跨版本累积，一旦在 V3 跳闸就再也不会自动恢复。

**修复**（commit `4d27325`）：加 `reset_circuit_breaker()` 公开函数，在 `evaluate_version` 每个版本开始前调用一次。考虑过用进程级隔离（每版起新进程），但 bge-m3 加载 5~10 秒 × 6 版 = 多 60 秒；最终选了进程内 reset 的轻量方案。

**教训**：**module-level 全局可变状态 + 长生命周期进程**这个组合天然反消融测试。任何带状态的模块（缓存、计数器、连接池、熔断器、重试预算）在跨条件实验里都需要显式 reset 钩子。这次靠运气只有一处，未来加新状态需要纪律——清单维护比"事后发现污染再补"代价低得多。同样的问题在生产里也会咬人：长跑进程的熔断器从早高峰带到晚高峰、从 staging 带到 prod，都是同一类。

---

## 5. RAGAS 三反直觉发现

**现象**：原假设是"组件越全越好"——V4 全开版应该三维度全冠。实际跑 6 版消融后，**三个维度的赢家分散在 V2 / V4 / V5**。没有任何一版同时拿下三冠。

**排查**：把每条单维度归因列出来——加 BM25+RRF（V1→V2）三维全涨；加 reranker（V2→V3）只涨 precision、降 relevancy；加 rewrite（V3→V4）涨 faithfulness、降 relevancy；关 RRF 改简单合并（V4→V5）反而 precision 最高。每条单看都讲得通、组合起来就互相抵消。

**根因**：三个维度衡量 RAG 流水线的不同部位（生成是否切题 / 是否有据 / 检索是否干净），互相不存在"一个赢则其他都赢"的关系。具体到本数据集：（a）LLM-based reranker（用 DeepSeek 当打分器，**反模式**——正解是 cross-encoder `bge-reranker-v2-m3`）方差大，挤掉 V2 已经排前的边相关 chunk，损伤 relevancy；（b）rewrite 把 query 改"文档化"，召回与 answer 更对齐（faithfulness 涨）但离原始意图更远（relevancy 跌）；（c）RRF 用 rank 平均化，对"两路都中等"友好；本数据集"术语精准题"占多数，BM25 单独发挥更准。

**修复 / 业务取舍**：不存在"修"——这是评测告诉我"全开不是最优"。把"赢家分散"翻译成业务取舍写进了 [`evaluation.md`](evaluation.md)：默认上线选 V2（BM25+RRF），延迟 ~3s、relevancy 0.952 最高；V4 比 V2 慢 10×、faithfulness 仅高 0.011，**只有强合规场景**值得付这个代价。换数据集（语义模糊题占多数）这个结论会反转，需要重测。

**教训**：（1）"组件越多越好"是直觉陷阱，每加一层都要单独测它的 ROI；（2）业务选型不是看综合冠军，是按业务关心的维度选——客服 FAQ 选 relevancy 冠军、合规审计选 faithfulness 冠军、人审复核选 precision 冠军；（3）评测是工程师的"业务对话语言"——只有把 trade-off 量化到表格里，才能跟产品经理讨论"是不是值得多付 30 秒延迟"。

---

## 关键 commit 索引

| Commit | 故事 | 主题 |
|--------|---|------|
| `4d27325` | 1, 4 | 5 处 LLM 调用 max_tokens 修复 + 熔断器 reset_circuit_breaker |
| `b0fc749` | 1 | reranker / scoring max_tokens 提到 3000 |
| `59f9898` | 2 | 真 SSE 流式（asyncio.Queue + call_soon_threadsafe） |
| `8cc5b6e` | 3 | BM25 索引缓存 + 锁防 stampede（40× 提速） |
| 多版评测 | 5 | `data/eval_results/comparison.json` —— V1-V6 全量数据 |
