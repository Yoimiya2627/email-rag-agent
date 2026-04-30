# 项目踩坑与挑战记录

> 这份文档记录在搭建/调优本项目过程中**真实**遇到的工程问题、根因分析、修复方案，以及由此沉淀下来的判断准则。每条都对应一个 commit，可以追溯到代码改动。
>
> 文档目的：面试时能讲出"是哪个具体 bug、怎么发现、怎么修、为什么这样修"，而不是套模板。

---

## 一、推理模型 max_tokens 陷阱（最严重，影响整个评测体系）

### 现象

跑 RAGAS 评测时，所有题目的 `answer_relevancy / faithfulness / context_precision` 三项都是 `0.0`。日志里全是：

```
LLM score attempt 1 failed: Empty response from LLM
LLM score attempt 2 failed: Empty response from LLM
LLM score attempt 3 failed: Empty response from LLM
```

DeepSeek 返回 HTTP 200，但 `choices[0].message.content` 是空字符串。

### 根因

DeepSeek 配置的模型是 `deepseek-v4-flash`——一个**推理模型（reasoning model）**。它的 API 返回结构和 GPT-3.5/4 不一样：

```
choice.message.content            ← 最终答案（可能为空）
choice.message.reasoning_content  ← 推理过程（思考链）
```

`max_tokens` 参数**同时覆盖**这两个字段。原代码各处写的是：
- `score_response`: `max_tokens=128`
- `_rewrite_query`: `max_tokens=128`
- `_extract_filters`: `max_tokens=256`
- `classify_intent`: `max_tokens=128`
- `grade_contexts` (Self-RAG): `max_tokens=64`
- `rerank`: `max_tokens=256`

推理模型对一段几百字的输入往往要写 1500~3000 token 的推理过程。结果就是：**推理还没写完就被截断，content 永远是空字符串**，`finish_reason='length'`。

### 影响（这是面试官会追问的部分）

每个失败点的真实业务后果：

| 出问题的调用 | 表面失败 | 实际影响 |
|---|---|---|
| `score_response` | RAGAS 全 0 分 | **整个评测无效**——基线和优化版本看上去一样 |
| `_rewrite_query` | 永远返回原 query | **V4 实际等于 V3**——query rewrite 这个开关名存实亡 |
| `_extract_filters` | 永远返回空 dict | sender / date / labels 过滤永远不触发 |
| `classify_intent` | 永远返回 `GENERAL` | **SummarizerAgent / WriterAgent / AnalyzerAgent 永远不会被路由到**——多 agent 架构形同虚设 |
| `grade_contexts` | except 分支返回所有结果 | **Self-RAG 重试逻辑永远不触发**（永远判定"全相关"） |
| `rerank` | 走 except 分支 | 连续 3 次后熔断器跳闸 → reranker 在该进程生命周期内永久关闭 → **V3/V4/V5 的 reranker 实际全是关的** |

也就是说：**修这一个 bug 之前，做的所有消融对比都不可信**。

### 诊断过程

1. 第一反应：以为是 prompt 问题，改 system message、调 temperature、加 max_tokens 256。无效。
2. 写了一个独立的 debug probe（`scripts/debug_deepseek_score.py`），不走业务代码、直接对同一个 prompt 调用 DeepSeek API，把完整 response 打印出来：

```python
choice = resp.choices[0]
print(f"finish_reason = {choice.finish_reason}")    # 'length' ← 关键线索
print(f"content       = {choice.message.content}")  # ''
print(f"reasoning     = {choice.message.reasoning_content}")  # 长篇推理
print(f"usage = completion={resp.usage.completion_tokens}")    # 256（顶到 max_tokens 上限）
```

`finish_reason='length'` + `completion_tokens=256` 直接坐实是 token 预算问题。

3. 再用 `max_tokens=1500` 重试同一个 prompt：`finish_reason='stop'`，`content` 拿到完整 JSON。问题确认。

### 修复

- 普通 LLM 调用（短输出）：`max_tokens=1500`（commit `4d27325`）
- 高推理量调用（reranker、三维度评分）：`max_tokens=3000`（commit `b0fc749`）
- **兜底**：当 `content` 仍为空时，从 `reasoning_content` 中抽取最后一个 JSON 对象作为答案（推理模型经常"想到一半就把答案写在思考里了"）

### 面试可能的追问及答法

> **Q: 为什么不一开始就把 max_tokens 设大一点？**
> A: 因为同一份代码写在 OpenAI 的 GPT-3.5/4 上是合理的——256 token 足够输出一个 JSON 评分对象。换成推理模型才暴露：`max_tokens` 参数语义变了。这是从 chat 模型切到推理模型时一个普遍的陷阱。

> **Q: 怎么发现这是推理模型而不是 prompt 问题？**
> A: 看 `finish_reason='length'` + `usage.completion_tokens` 顶到上限。如果是 prompt 问题，模型会输出"我不知道"之类的内容，而不是空字符串。

> **Q: 为什么不直接换 deepseek-chat（非推理模型）？**
> A: 项目用 v4-flash 是因为推理模型在打分这种"需要权衡多维度"的任务上质量更稳。换非推理模型省 token 但会牺牲质量。我们选择适配它的 token 预算。

---

## 二、消融实验数据被熔断器污染

### 现象

跑 V1→V6 ablation 时，V3 偶发的 reranker 失败会**永久关闭** V4/V5/V6 的 reranker——即使 V4/V5/V6 跑的时候 LLM 已经恢复正常。

### 根因

`core/reranker.py` 里熔断器计数器是 module-level 全局变量：

```python
_consecutive_failures = 0
_FAILURE_THRESHOLD = 3
```

V1~V6 共用一个 Python 进程、共用这个变量。一旦在 V3 累积了 3 次失败，后续版本一进来就直接走 `circuit breaker open` 分支返回原始排序——**reranker 实际被禁用了，但日志里看到的还是"V4 ENABLE_RERANKER=True"**，得到的对比表完全失真。

### 修复

`core/reranker.reset_circuit_breaker()` 函数 + `evaluate_version` 在每个版本开始前调用一次（commit `4d27325`）：

```python
def evaluate_version(version, ...):
    apply_flags(flags)
    from core.reranker import reset_circuit_breaker
    reset_circuit_breaker()  # 防止上一版本的失败状态泄漏
```

### 经验

**全局可变状态 + 长生命周期进程**这个组合天然反消融测试。任何带"状态"（缓存、计数器、连接池）的模块在跨条件实验里都需要显式 reset 钩子。

### 面试追问

> **Q: 为什么不用进程级隔离，每个版本启一个新进程？**
> A: 启动开销大——bge-m3 模型加载就 5~10 秒，6 个版本就多 60 秒。用进程内 reset 钩子是工程上更轻量的方案。但代价是要主动维护可重置状态的清单（这次只有熔断器，未来加新状态时容易漏）。

---

## 三、SSE 假流式

### 现象

`/chat/stream` 端点声称是 SSE 流式输出，前端开"流式输出"开关后，用户体感和非流式一样——等十几秒，然后所有 token 一次性出现。

### 根因

`api/main.py` 里这一段：

```python
tokens = await loop.run_in_executor(
    None,
    lambda: list(stream_generate(...)),  # 这里 list() 把所有 token 收完才返回
)
for token in tokens:
    yield f"data: {token}\n\n"
```

`list(generator)` 会等生成器全部 yield 完才返回。等于先把整个回答存到内存，再"假装"一个个流给前端。

### 修复（commit `59f9898`）

用 `asyncio.Queue` + 生产者线程桥接：

```python
async def event_generator():
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def producer():
        try:
            for token in stream_generate(...):  # 阻塞迭代器
                loop.call_soon_threadsafe(queue.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    loop.run_in_executor(None, producer)  # 不 await，让生产线程独立跑

    while True:
        item = await queue.get()
        if item is SENTINEL: break
        yield f"data: {item}\n\n"
```

关键是 `call_soon_threadsafe`——OpenAI SDK 的迭代器是阻塞的、跑在工作线程；asyncio 队列只能在事件循环线程里操作。这个函数是跨线程往事件循环里投递任务的唯一安全方式。

### 验证（写了个独立测试 `scripts/debug_stream_test.py`）

```
received 23 SSE events in 70.55s
first→last spread: 0.34s
```

23 个事件分布在 0.34 秒内（旧代码所有事件时间戳完全相同 = spread ≈ 0）——证明 token 是**逐个**到达的。

### 一个反直觉的发现

修好 SSE 之后，用户体感**仍然**是"等 70 秒然后突然刷出来"。原因：推理模型 70 秒都在产 `reasoning_content`，而 `generator.py` 只 yield `delta.content`，所以前 70 秒 SSE 通道里什么都没有。

**SSE 桥接是对的，模型行为决定了体感。** 真正的 UX 修复是把 reasoning 流也转发出去（独立优化，未做）。

### 面试追问

> **Q: 为什么不直接 `async for token in stream`？**
> A: OpenAI Python SDK 的 stream 对象是同步迭代器（`__iter__`），不是异步迭代器。`async for` 会报错。可以用 `AsyncOpenAI` 客户端拿到异步迭代器，但项目其他地方都用同步客户端，混用维护成本高，所以用线程桥接。

> **Q: 为什么用 `call_soon_threadsafe(put_nowait)` 而不是 `asyncio.run_coroutine_threadsafe(queue.put(...))`？**
> A: 前者更轻——只是往事件循环投递一个回调，不创建新协程。后者会创建一个 Future 还要等返回。生产-消费场景下前者更合适。

---

## 四、多 worker 下 session 状态错乱

### 隐患

```python
_sessions: dict[str, ConversationMemory] = defaultdict(ConversationMemory)
```

两个问题：

1. **defaultdict 的 get-or-create 不是原子操作**。两个线程同时第一次访问 `_sessions["alice"]`：
   - 线程 A：检查不存在 → 调用工厂 `ConversationMemory()`
   - 线程 B：检查不存在 → 调用工厂 `ConversationMemory()`
   - 后者会覆盖前者，A 写入的对话历史丢失。

2. **多 worker FastAPI（`uvicorn --workers 4`）下根本不共享**——每个 worker 进程一个 dict，用户的对话可能这次落到 worker A 看到完整历史，下次落到 worker B 看到空历史。

### 修复（commit `59f9898`）

显式锁 + helper：

```python
_sessions = {}
_sessions_lock = threading.Lock()

def _get_session(session_id) -> ConversationMemory:
    with _sessions_lock:
        memory = _sessions.get(session_id)
        if memory is None:
            memory = ConversationMemory()
            _sessions[session_id] = memory
        return memory
```

`ConversationMemory` 内部 `add/to_messages/clear` 也加了 `threading.Lock`——因为 `add()` 里有 `self._turns = self._turns[-N:]` 这种 read-modify-write，并发下会丢失 turn。

### 多 worker 问题没修，只修了多线程

显式承认**当前实现是单进程方案**。要支持多 worker 需要外置 session store（Redis）。这是个面试加分点：知道边界在哪。

### 面试追问

> **Q: 你这只是修了线程安全，多 worker 怎么办？**
> A: 当前实现是单进程多线程方案，对单机 demo 够用。生产部署需要把 `_sessions` 换成 Redis 之类的外置存储，session_id 是 key、对话历史是 value，多 worker 共享。但项目定位是面试 demo，没引入 Redis 这个额外依赖。

---

## 五、LangGraph 名义实现

### 问题

`agents/graph_workflow.py` 文件头注释写"LangGraph-based Self-RAG workflow"，但代码是手写的 while 循环：

```python
def run(state):
    state = node_rewrite(state)
    for _ in range(MAX_RETRIES + 1):
        state = node_retrieve(state)
        state = node_grade_contexts(state)
        if _should_retry(state) == "generate": break
        state = node_rewrite(state)
    state = node_generate(state)
    return state
```

注释甚至诚实地写了 "Uses a manual loop instead of langgraph to avoid the extra dependency"——但这就让 roadmap "任务 5：LangGraph Self-RAG 工作流"成为虚名。

### 修复（commit `fc7392d`）

装 `langgraph==1.1.10`，重写为真正的 `StateGraph`：

```python
graph = StateGraph(RAGState)
graph.add_node("rewrite", node_rewrite)
graph.add_node("retrieve", node_retrieve)
graph.add_node("grade", node_grade_contexts)
graph.add_node("bump_retry", _bump_retry)
graph.add_node("generate", node_generate)

graph.set_entry_point("rewrite")
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges(
    "grade",
    _should_retry,
    {"retry": "bump_retry", "generate": "generate"},
)
graph.add_edge("bump_retry", "rewrite")
graph.add_edge("generate", END)
compiled = graph.compile()
```

`_should_retry` 改成纯函数（只读 state、返回 edge label）；副作用 `retry_count += 1` 拆到独立的 `bump_retry` 节点——这样条件谓词可测试。

### 面试追问

> **Q: LangGraph 比 if-else 好在哪？**
> A: 三点：(1) 条件分支显式建模为图的边，逻辑可视化、可序列化；(2) 节点解耦，易加 checkpoint / 持久化（langgraph 自带 checkpointer）；(3) 状态变更和路由判断分离，单测容易写。手写 while 循环这三个都做不到。

> **Q: 那为什么一开始没用 LangGraph？**
> A: 一开始想避免新依赖（langgraph 1.x 还在快速演进）。但既然项目里有"LangGraph 任务"这个交付项，名实不符比加一个 25MB 依赖更糟。

---

## 六、BM25 索引每次查询都重建（性能瓶颈）

### 现象

5000 封邮件后，BM25 检索单次耗时 800ms+，是检索环节最慢的部分。

### 根因

```python
def bm25_search(query):
    all_chunks = get_all_chunks()                        # 全量从 ChromaDB 读
    corpus = [c["content"] for c in all_chunks]
    bm25 = BM25Okapi([_tokenize(doc) for doc in corpus]) # 5000 文档全量分词 + 建索引
    scores = bm25.get_scores(_tokenize(query))           # 真正的检索
    ...
```

每次查询都重新做"全量 ChromaDB 读 → 全量分词 → 建 Okapi 索引"——纯重复劳动。

### 修复（commit `8cc5b6e`）

```python
_bm25_cache: Optional[Tuple[int, BM25Okapi, List, List]] = None
_bm25_lock = threading.Lock()

def _get_bm25_index():
    n = get_collection_count()  # ChromaDB cheap count，不读数据
    with _bm25_lock:
        if _bm25_cache is not None and _bm25_cache[0] == n:
            return _bm25_cache[1:]  # cache hit
        # cache miss: 重建
        all_chunks = get_all_chunks()
        bm25 = BM25Okapi(...)
        _bm25_cache = (n, bm25, all_chunks, corpus)
        return ...
```

关键设计：用 `collection.count()`（O(1)）做缓存键，**不**用全量数据当 key——后者要先全量读取一遍，缓存就白做了。

### 实测

```
冷启动:     846 ms  （首次构建）
缓存命中:    22 ms  （40× 提速）
```

### 面试追问

> **Q: 为什么用 count 当 cache key？要是有人删了一封又加了一封，count 不变，缓存就脏了。**
> A: 对，这是已知妥协。生产场景应该用版本号（每次写操作 +1）或者文档 hash。当前项目里 `/index` 端点会调 `invalidate_bm25_cache()` 主动失效——只要走 API 改数据就 OK；要是绕过 API 直接改 ChromaDB 才会出现你说的情况，但那不是支持的用法。

> **Q: BM25 这个量级（5000 文档）有必要缓存吗？要是 100 万文档怎么办？**
> A: 5000 文档冷启动 846ms 是用户感知阈值边缘了；100 万文档纯 in-memory BM25 不现实，应该换 Elasticsearch（roadmap 里也有提到）。这次缓存是当前规模下的对症措施。

---

## 七、Coordinator 路由被绕过

### 问题

`/chat/stream` 端点写死了"调 RetrieverAgent"，没经过 Coordinator 的意图分类：

```python
agent = RetrieverAgent()
rewritten = agent._rewrite_query(...)         # 调私有方法
filters = agent._extract_filters(...)         # 调私有方法
results = hybrid_search(...)
reranked = rerank(...)
for token in stream_generate(...): yield ...
```

后果：用户问"帮我总结最近邮件"也会走检索逻辑，SummarizerAgent 永远不会被调到。同时 API 层调 RetrieverAgent 的私有方法（`_rewrite_query`/`_extract_filters`），破坏了 agent 抽象。

### 修复（commit `a0fc0bd`）

1. RetrieverAgent 加公开方法 `prepare_contexts(query)`，封装 rewrite + filter + retrieve + rerank 整条链：

```python
class RetrieverAgent:
    def prepare_contexts(self, query):
        rewritten = self._rewrite_query(query)
        filters = self._extract_filters(rewritten)
        ...
        return rerank(...)

    def run(self, request, memory=None):
        contexts = self.prepare_contexts(request.query)
        answer = generate_answer(...)
        return AgentResponse(...)
```

2. `/chat/stream` 走 Coordinator 路由：

```python
intent = classify_intent(request.query)
if intent in (IntentType.RETRIEVE, IntentType.GENERAL):
    contexts = RetrieverAgent().prepare_contexts(query)
    for token in stream_generate(query, contexts): yield ...
else:
    response = route(request)  # 走对应专家 agent，不流式
    yield response.answer
```

非检索意图（摘要/写信/分析）当前不支持流式（专家 agent 没 expose 流式接口），整段答案作为一个 SSE 事件返回——比之前完全错路由要好。

### 面试追问

> **Q: 摘要/写信/分析这三个 agent 为什么不支持流式？**
> A: 当前实现是 `agent.run()` 一次性返回 `AgentResponse`，内部用同步的 `chat.completions.create()`。要支持流式需要每个 agent 多 expose 一个流式方法，工作量是 O(N agent)。当前权衡是：检索类是高频用例（80% 流量），优先做；其他类用一次性返回兜底。

---

## 八、误报识别（也很重要）

### Self-RAG 重试 off-by-one 不是 bug

代码审查时另一个 reviewer agent 报"`MAX_RETRIES + 1` 跑 3 轮、spec 是 2 轮"。我没直接接受，重新对了一遍语义：

- spec：**最多 2 次重试**
- 1 次初始 + 2 次重试 = 3 轮 retrieve
- `range(MAX_RETRIES + 1)` = `range(3)` = 3 轮 ✓
- `_should_retry` 条件 `retry_count < MAX_RETRIES`（即 < 2），所以 retry_count=0 和 1 时重试，共 2 次 ✓

**代码本身就是对的，reviewer 误判**。所以没动。

### 面试追问

> **Q: 你怎么判断别人的 review 意见是错的？**
> A: 不是"判断对错"，是把 spec、计数语义、循环边界全部展开列出来一一对照。审查意见是输入，不是结论。这次刚好审查错了；下次也可能我自己写错。重要的是有"展开验证"的习惯而不是盲目接受或拒绝。

---

## 九、调试方法论沉淀

这次几个 bug 反映的方法论：

### 方法 1：永远先调 finish_reason / usage，再改 prompt

LLM 调用失败时第一反应不是改 prompt，是看：

```python
choice.finish_reason   # 'stop' | 'length' | 'content_filter' | ...
resp.usage             # prompt_tokens, completion_tokens
choice.message.refusal # 拒答字段（部分模型）
```

`finish_reason=='length'` → 加 max_tokens；`'content_filter'` → 改 prompt；`refusal` 非空 → 模型拒答。三种处理完全不同，不能瞎改。

### 方法 2：写独立 probe 而不是改业务代码 debug

`scripts/debug_deepseek_score.py` 这种 probe：单独的脚本、纯净的 prompt、打印完整 response 结构。比在业务代码里加 print 高效很多——业务代码有重试、降级、缓存，会掩盖原始信号。

### 方法 3：跨条件实验前清查全局状态

跑 ablation / A-B 之前列清单：哪些模块持有跨调用状态？连接池、计数器、缓存、单例？每条都要一个 reset 入口。这次靠运气只有 reranker 熔断器一处，未来增加新状态需要纪律。

### 方法 4：性能优化的"度量两次"原则

第一版 BM25 缓存只把建索引那一步省了，仍然每次全量 ChromaDB 读——基准测试 743ms→311ms，**省了一半但比预期差远了**。打开 profiler（其实就是分段计时）才发现 ChromaDB 全量 read 是另一个大头。改成 cheap count probe 后到 22ms。

启示：没基准就没优化。先量化、再改、再量化。

---

## 十、待办（坦诚说明）

这次没修但识别出来的：

1. `_get_embedding_fn` 私有函数被 langchain_version 调用——命名不规范
2. `Dockerfile` 没 `EXPOSE 8501`、frontend 容器每次启动跑 `pip install`
3. `lstrip("json")` 写法（应该用 `removeprefix`）出现在 5+ 处
4. retrieval pipeline 在三处重复实现（RetrieverAgent / graph_workflow / run_ragas_eval）
5. ENABLE_BM25=True + ENABLE_RRF=False 的 dict 覆盖式融合不是真正的"score merge"
6. 推理模型的 `delta.reasoning_content` 没转发给前端，导致流式 UX 在长推理任务下体感仍然是"等 + 突然出现"
7. 多 worker session 隔离需要外置 store

面试时如果被追问"还有什么没做"，主动讲这些比假装代码完美更可信。

---

## 关键 commit 索引

| Commit | 主题 |
|--------|------|
| `36cefd4` | RAGAS 打分 max_tokens 修复（首次发现推理模型陷阱） |
| `4d27325` | 5 处 LLM 调用全部 max_tokens 修复 + 熔断器 reset |
| `59f9898` | 真 SSE 流式 + session 锁 + ConversationMemory 锁 |
| `fc7392d` | 手写循环 → 真正的 LangGraph StateGraph |
| `8cc5b6e` | BM25 索引缓存（40× 提速） |
| `aa0f7e1` | LangChain `chain(...)` → `.invoke()` |
| `a0fc0bd` | `/chat/stream` 走 Coordinator + `prepare_contexts` 公开方法 |
| `b0fc749` | reranker / scoring max_tokens 提到 3000（高推理量调用） |
