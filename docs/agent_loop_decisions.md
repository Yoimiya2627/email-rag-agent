# Agent Loop 升级 — 决策日志

记录 agent 化改造过程中靠实测/验证得出的关键决策，供后续步骤和文档收尾参考。

## Step 0 — 预检（已完成）

探针脚本：`scripts/probe_function_calling.py`（可重复运行复现）。

### 实测结果

| 模型 | function calling | 单次调用延迟 |
|------|:----------------:|------------:|
| `deepseek-v4-flash`（推理模型） | ✅ 返回 `tool_calls` | ~2.3s |
| `deepseek-chat`（非推理） | ✅ 返回 `tool_calls` | ~1.3s |

运行时 `LLM_TIMEOUT` 实测为 **60s**（`.env` 覆盖了 `settings.py` 的默认 15）。

### 决策

1. **agent loop 用原生 function calling**——两个模型都稳定返回 `tool_calls`，不需要手写 JSON 协议兜底。这清除了改造的头号风险。
2. **planning / 选工具调用用 `deepseek-chat`**：更快（~1s vs ~2s）、无推理开销、function calling 已验证。循环中每一步都省时间，累积收益明显。
   → 新增配置项 `AGENT_PLANNER_MODEL`，默认 `deepseek-chat`。
3. **最终答案生成保留 `deepseek-v4-flash`**：需要一点推理质量的地方继续用推理模型。
4. **`LLM_TIMEOUT` 无需改**：实测单次调用 1~2.3s，60s 余量充足。Bug #1（"15s 超时 vs 推理模型"）经实测**不是运行时 bug**——只是 `settings.py` 默认值 15 偏低 + 注释过时。

### 留给 Step 7 的文档/默认值清理

- `settings.py:52` `LLM_TIMEOUT` 默认值 `15` → 改成 `60`，与 `.env`/architecture 一致。
- `core/generator.py:110` 注释称推理模型 `reasoning_content` 要 30~70s——实测仅 ~105 字符 / 2.3s，注释已过时，需更正。
- 三处 `LLM_TIMEOUT` 文档不一致：README 配置表写 15、architecture §9 写 60、settings 默认 15。统一为 60。
- architecture.md §3 "链路总耗时 ~30~50秒" 与 README 评测 "V2 mean 8.7s" 不一致，需对齐。
