# SSE 完成消息消费竞态修复

Python 3.10 的 UTC `Z` 解析修复后，提交 `9396f18` 的两组 Ubuntu CI 通过，Windows CI 的日期用例也通过，但 `test_real_graph_and_sse_model_boundaries_retain_constraint_and_commit[stream]` 失败：客户端收到了 intent 和 token，却没有最终 metadata 与 `[DONE]`，只收到 `Stream stopped or timed out`。

该 fixture 没有缩短 `AGENT_RUN_TIMEOUT`，默认截止是 120 秒。独立审查和确定性实验发现 `api/main.py` 消费端存在以下时序：

1. 队列暂时为空，后台 `events.get(..., .2)` 超时。
2. 消费协程尚未处理这个异常时，producer 完成持久化，将 metadata 与 `[DONE]` 入队，然后退出。
3. 消费协程处理 `queue.Empty`，发现 producer 已退出便直接结束，遗弃队列里的完成消息。

这段逻辑的 Git blame 指向既有提交 `ac57ca6a`，不是 UTC 参数兼容补丁引入。CI 日志不足以证明该时序是那次失败的唯一原因，但屏障控制的复现已证明该生产缺陷存在，输出与 CI 失败一致。

## 修复与回归

仅修改消费端 `queue.Empty` 分支：producer 仍活着时继续等待；producer 已退出时先执行不阻塞的 `get_nowait()`，只有队列确实为空才结束。取得事件后仍使用原有 metadata / `[DONE]` 处理。外层截止检查、断连取消和 `finally` 中的停止信号保持不变。

`tests/test_stream_completion_race.py` 使用实际 `/chat/stream` 端点和 SQLite 存储，仅替换 provider 与队列/线程的 API 局部绑定；不修改共享 threading/queue 模块。事件屏障强制制造上述窗口，不靠 sleep 或重复碰运气。测试先确认 producer 已退出、SQLite 已提交两轮、队列仍有两个完成事件，再验证客户端收到完整结果。另一个反向测试确认 producer 已退出且队列真正为空时仍返回错误。

- [修复前确定性失败](sse-before-fix.txt)：1 failed、1 passed；公开摘录仅清理行尾空白。
- [修复后定向回归](sse-targeted-after-fix.txt)：51 passed、8 subtests，含真实 SSE/上下文/日期入口和既有截止保护用例。
- 独立审查确认消费不会重复或遗漏已入队事件，新增读取不阻塞，也不延长截止时间。

该修复提交 `67bc3914a01dff80f17a49ac3324a476e58e0e11` 的全量离线 **1795 passed、3 skipped、190 subtests passed**，以及[三组 CI](https://github.com/Yoimiya2627/email-rag-agent/actions/runs/41571044034)均已通过。随后才启动最后一轮在线配对，得到 **66.67% → 86.67%**；前两轮完整保留。见[最终验收](release-acceptance.md)。

最终证据提交还改进了测试夹具：以成功消费 token 为注入前提，替代读取尝试次数；主动模拟 prefix 到达前的 Empty，确认早期超时不会提前触发竞态屏障。两个测试的数量及业务断言保持不变，相关套件[30 passed、8 subtests passed](sse-harness-final.txt)。这项测试变更不改应用代码；在线记录的 140 个源码/依赖文件哈希保持一致。
