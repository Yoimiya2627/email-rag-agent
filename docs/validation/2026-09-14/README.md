# 2026-09-14 项目收尾验收

本目录对应三项工作：更新当前项目说明、整理当前版本演示、补充可复现的小规模效果评测。既有功能保持原有部署边界；验收中发现的界面问题只做局部修复。

本文保留合成演示阶段的验收记录。随后完成的163自动索引、38封真实邮件测试及最新1526项回归结果，见 [真实邮件自动索引验收](163-auto-index.md)；下文“尚未接入”等表述仅对应当时阶段。

## 材料入口

| 材料 | 内容 |
|---|---|
| [项目 README](../../../README.md) | 当前功能、部署边界、演示入口、实测与历史成绩区分 |
| [演示说明](demo.md) | 合成资料导入、索引、问答、来源回读、回复草稿和模拟审批 |
| [当前效果评测](evaluation.md) | 固定题目、真实模型逐题结果、耗时、token 和语义审阅 |
| [环境](environment.json) / [源码指纹](source-manifest.json) | 本轮评测实际依赖、模型配置、工作区文件内容指纹 |
| [Astra 交叉审核](cross-review.md) | 独立审核结论、发现的问题和复验记录 |

## 前一轮工程验证基线

以下是北京时间 2026-09-14 前一轮完整测试的记录，不是本轮文档修改后重新运行所有平台所得的新成绩。相同测试套件跨环境执行，不能把通过数相加作为独立用例数。

| 环境/范围 | 结果 | 证据 |
|---|---:|---|
| Windows 全量回归 | 1479 passed，3 skipped | [完整输出](engineering-windows.txt) |
| Linux 存储依赖环境 | 1476 passed，6 skipped | [完整输出](engineering-linux.txt) |
| 实际 API 镜像加可选测试依赖 | 1476 passed，6 skipped | [完整输出](engineering-api-image.txt) |
| 本机进程验收 | 16 passed，0 failed | [归档摘要](engineering-summary.json) |
| Docker 运行验收 | 9 passed，0 failed | [归档摘要](engineering-summary.json) |

Windows 跳过项为两个无权限创建符号链接的场景及一个 Linux `dir_fd` 场景。Linux 跳过项为两个 PowerShell/Windows 进程清理、一个 DPAPI、三个 Windows Job Object 场景。API 镜像测试中的 LangGraph 为额外安装的可选依赖，不能据此认为默认 API 镜像包含它。

## 本轮发现与修复

**P2：新回答的来源回读按钮首次点击丢失。** 新生成回答使用 `current-response` 作为证据控件标识，下一次 Streamlit 重跑则以 `history-N` 渲染同一回答，首次点击事件无法匹配。用户点击后看不到原文，影响证据核验。

- 真实浏览器录制中复现：问答成功，第一次点击“核验并回读原文”后没有原文。
- 修复前新增的两个回归场景均失败：聊天首页和高级工作台的首次点击均未调用 `/evidence/reread`。
- 修复仅让新回答立即使用它即将占据的历史消息序号作为控件标识，保持重跑前后一致；没有改变证据核验接口或修改索引逻辑。
- 修复后，两个场景均能首次点击调用接口并显示原文；与既有上下文及本轮评分测试合计 14 项通过。

修改位置：[frontend/app.py](../../../frontend/app.py)；回归：[tests/test_ctx_frontend.py](../../../tests/test_ctx_frontend.py)。

## 本轮最终验收结果

| 工作项 | 结果 | 证据 |
|---|---|---|
| README 与历史成绩区分 | 已更新当前能力、部署限制、演示、评测和测试入口；历史实验保留独立标识 | [项目说明](../../../README.md) / [历史实验](../../evaluation.md) |
| 最终代码小规模效果评测 | 12/12 通过；14/14 引用 ID 有效；主 agent 逐题语义核对无事实错误 | [完整响应](evaluation-results.json) / [报告](evaluation.md) |
| 当前版本浏览器演示 | 完整录制 56.52 秒，11 个流程检查点通过，0 个未捕获页面错误 | [视频](demo.mp4) / [任务与审批](demo-workflow.json) |
| 最终 Windows 全量回归 | **1490 passed，3 skipped**，另有 190 个 subtests；187.84 秒 | [完整输出](windows-final-regression.txt) |
| 局部修复及评分回归 | 14 passed；新增 9 项评分/隔离测试和 2 个首击回读场景，其余 3 项是既有上下文回归 | [输出](focused-regression.txt) |
| Astra 独立交叉审核 | 可验收，无阻塞问题；独立 14 项回归通过，12 题事实/引用、指标、源码、视频和 SQLite 归档核对一致 | [审核报告](cross-review.md) |
| 最终证据检查 | 文档本地链接无缺失；源码/评测/demo 指纹一致；未发现实际配置密钥进入公开文本；所有本轮服务已停止 | [检查记录](artifact-verification.json) |

首次全量回归为 1489 passed、1 failed、3 skipped：父进程用 GBK 解码被设置为 UTF-8 的子进程输出，导致 `test_check_does_not_import_heavy_modules_or_settings` 的 JSON 读取失败。这是本轮测试启动配置不一致；统一 `PYTHONUTF8=1` 后该项及整套回归通过，没有为此修改业务代码或放宽断言。首次输出保留为 [编码配置失败记录](windows-initial-encoding-failure.txt)。

本轮全量回归没有重新在 Linux 或重建后的 Docker 镜像执行，所以上方 Linux / Docker 数字明确列为前一轮基线。最终 Windows 通过数较基线增加 11，与本轮新增场景一致。

## 验收边界

- 演示和效果评测使用固定合成邮件，不使用个人邮箱数据。
- 163 同步与本地搜索尚未自动接入 AI 索引；Gmail 接入代码存在，但本轮没有真实 Gmail / 163 在线验收。
- 审批 provider 固定为 simulated；操作不会发送邮件或创建 Gmail 远端草稿。
- 当前效果评测只覆盖 `/query` 的 12 道固定题目。录像中的 Agent 草稿流程另行核对，不计入这个通过率。
- 本轮未进行历史 105 题 Agent gate、V1–V7 重跑、真实大邮箱、长期运行、并发容量、托管 CI 或新的依赖漏洞库扫描。
- 所有新增运行状态和录制中间文件位于 `E:/email-agent-validation/portfolio-20260914/`。对外演示与非敏感证据复制到本目录，密钥不写入报告。
