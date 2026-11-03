# Python 3.10 参考时间 CLI 兼容补修

首次推送 `4a651e8` 后，云端离线回归的两组 Ubuntu Python 3.11 任务通过，Windows Python 3.10 任务失败。[失败日志摘录](python310-first-ci-failure.txt)为 **3 failed、1643 passed、31 skipped、190 subtests passed**；三个失败来自同一参数化测试中的同一个成功断言。

`datetime.fromisoformat('2026-11-02T12:00:00Z')` 在 Python 3.10 不接受 UTC 的 `Z` 后缀，本机 Python 3.11 接受。因此本机 1772 项通过不能替代不同 Python 版本的 CI。失败不来自任务时钟冻结、checkpoint 恢复或时钟精度。

## 修改范围

三个评测 CLI 的日期参数解析入口各增加两行：当输入是字符串且以 `Z` 结尾时，仅将终端 `Z` 转为 `+00:00`，随后执行原有解析与 timezone-aware 校验：

- `scripts/evaluate_context_recall.py::_parse_reference_time`
- `scripts/paired_acceptance.py::_parse_reference`
- `scripts/run_agent_eval.py::_reference_time_arg`

不进行全局替换、不去掉空白，不改变显式 `+08:00` 输入。无时区、错误格式、双时区及非字符串输入仍转换为原有 argparse 错误。checkpoint 由程序序列化为带明确 offset 的 ISO 字符串，不是这次失败源。

`tests/test_eval_reference_cli.py` 新增 21 个参数化回归，覆盖三个真实入口的 UTC、微秒精度、偏移保留、无时区、双时区和 `None` 输入。已由独立审查员检查；另外使用真实 Python 3.10.20 从源码提取实际函数 AST，执行每入口 7 个用例，[全部通过](python310-parser-check.json)。该 AST 检查只验证解析函数，完整导入与项目行为仍由全量离线和 CI 覆盖。

补修版本机全量离线已完成：**1793 passed、3 skipped、190 subtests passed，375.08 秒，退出码 0**（此前 1772 + 新增 21）。[完整输出](offline-python310-compat-full.txt)及[被测修改文件哈希](python310-repair-source-snapshot.json)单独留存。云端 CI 与追加在线结果将在补修提交后记录。

## 评测记录的处理

第一轮在线配对使用 `+08:00`，没有执行受影响的 `Z` 分支。其 **63.33% → 80.00%** 仍是代码 `245c4d3` 那一轮的实际结果，原报告及源码哈希保留原样。

后续版本确实增加了三个源码文件的差异，不再声称其全部源码哈希与第一轮相同。为让最终成绩直接绑定补修后的代码，完成离线验证并提交后，将在全新隔离副本上追加一轮完整 30 题在线配对，记录其独立提交与全部结果。追加运行由 CI 兼容缺陷触发，不按成绩选择是否保留；第一轮报告不会被覆盖。该追加复验完成前，不能将第一轮数字表述成补修后新跑出的成绩。

[第一轮完整在线记录](final-paired-revalidation.md)保留为历史。

## 补修提交上的在线结果与后续 CI

补修代码提交为 `9396f1819f9e08140bf7d3511cc289a5d687b4b4`。其干净工作树上的完整 30 题追加配对已完成，退出 0：V2 **76.67%（23/30）**、V7 **90.00%（27/30）**，双方均 29 正常、1 空、0 错误。[独立审计](paired-python310-audit.json)通过 1053 项检查，140 个源码哈希与被测提交一致；[公开逐题报告](paired-python310-public.json)及[三轮比较](paired-three-runs-comparison.json)保留完整证据。

相对第一轮 80.00%，变分仅 `gold_004/014/028/030`：前三题过滤字段和候选发生变化，最后一题由连接失败变为成功。六行 UTC 兼容补丁在两轮使用的 `+08:00` 输入上不会进入新分支；评分、检索与重排实现未改变，不能把跨轮分数变化归因该补丁。最新这一轮同候选 V7 比 V2 提升 13.33 个百分点；此前两轮结果没有被覆盖。

该提交的两组 Ubuntu CI 通过，Windows 日期用例已通过，但出现另一条 SSE 完成消息竞态。后续已完成[该竞态修复](sse-completion-race.md)及本机/三组 CI 验收，并在稳定应用代码 `67bc391` 上完成最后一轮 **66.67% → 86.67%**。因此此处 90.00% 保留为中间版本实验；当前交付以[最终验收](release-acceptance.md)为准。
