# 2026-09-12 源码覆盖与本机部署验收

已将《多邮箱Agent.zip》的“项目源码”覆盖到项目，修复部署和运行问题，完成本机真实模型验证。
压缩包 SHA-256：`ef4d91a922d680b6d1bd9ec8e90a5266419383d697e0c7735bd7072e39fdb7d1`。

覆盖前已保存现有工作区文件、Git 差异和本地配置；75 个与新版架构不兼容的旧模块、测试和部署文件移入项目外备份。历史说明和压缩包中的旧 baseline 未作为运行源码。密钥、凭证、模型、私人邮件、会话、审批和数据库均未加入 Git。

## 修复内容

1. 离线测试允许 Python 3.11 在 Windows 的标准库 `_fallback_socketpair` 内部通信，同时继续阻断实际网络访问。
2. 会话数据库迁移与检查连接显式关闭，修复 Windows 文件句柄残留；对应审批测试也关闭自己打开的连接。
3. Gmail 草稿执行的剩余超时不再因浮点舍入超过原始预算。
4. Docker Compose 将工具结果数据库纳入持久化目录。
5. Windows/Makefile 前端启动固定监听 `127.0.0.1:8501`，不自动打开浏览器。
6. Windows 停止任务时清理本次启动的完整进程树，避免虚拟环境 Python 的实际服务子进程残留。
7. 针对实测 Chroma 1.5.8 在 Windows 中文路径无法保存 HNSW 文件的问题，增加 ASCII 完整路径检查、doctor 提示和部署说明。
8. `/warmup` 与 `/ready` 实际读取已发布索引并核对数量，损坏索引返回 503；补充达到 HNSW 落盘阈值后在独立进程恢复读取的回归。
9. 摘要实测出现空模型回答。按 [DeepSeek 官方 thinking 接口说明](https://api-docs.deepseek.com/guides/thinking_mode/)，对官方 V4 请求显式设置 `DEEPSEEK_THINKING_MODE=disabled`，避免默认推理耗尽短输出预算；其他服务不注入该参数，调用方显式设置仍优先。

## 本机环境

- Windows、Python 3.11.15，项目虚拟环境位于 E 盘；此次新增依赖为 ijson，依赖兼容性检查通过。
- BGE-M3 使用已有 E 盘模型缓存和 CPU，固定 revision `5617a9f61b028005a4858fdac845db406aefb181`。
- 实际索引目录：`E:/email-agent-runtime/chroma-v2`。5000 封包内示例邮件，共 5001 个片段。
- 在校验 generation、模型 revision、配置、向量维度及归一化后，从本次失败构建的日志队列恢复 906 个向量；其余 4095 个向量重新计算。原索引保留备份。
- 构建进程退出后，独立新进程实际读取到 5001 个片段，存储核验通过。
- DeepSeek 生成模型和规划模型已通过实际调用；保留模型名称，将本机 V4 thinking 模式显式关闭。
- 前端：`http://127.0.0.1:8501`；API：`http://127.0.0.1:8000`；MCP：`http://127.0.0.1:8001/mcp`。

## 验证结果

最终全量离线回归：**977 passed、190 subtests passed、2 skipped，退出码 0**。两项跳过均因当前 Windows 账户不能创建符号链接。编译、依赖兼容性和 Git 差异空白检查通过。

以下实测使用包内示例语料、真实本地 BGE-M3/重排模型和已配置的 DeepSeek 服务：

| 功能 | 结果 |
| --- | --- |
| 完整索引及模型就绪 | 通过 |
| 邮件和线程原文读取 | 通过 |
| 检索回答与引用原文核验 | 通过 |
| 邮件摘要 | 通过 |
| 回复草稿 | 通过 |
| 全索引统计 | 通过 |
| SSE 流式回答 | 通过 |
| Self-RAG 反思工作流 | 通过 |
| 自主 Agent 统计工具调用 | 通过 |

另外通过浏览器验证了“提交自主 Agent 请求 → 后台任务 → 工具调用 → 待审批模拟草稿 → 页面拒绝”的完整流程，以及全新会话中的普通模式摘要，页面正确显示三个优先方向和证据入口。API 模拟批准、拒绝、重复批准幂等验证通过，`sent=false`。API 重启后会话和任务结果保持一致。MCP 完成实际握手，发现 10 个工具、2 个提示词和 1 个资源；实际 `email_stats` 工具返回 5000 封已索引邮件。非法 Origin/Host、空查询及不存在任务的边界响应符合预期。

## 使用与验证边界

本次为 Windows 本机部署；未执行 Docker 镜像构建或容器启动。未连接新的真实邮箱账号、未创建远端 Gmail 草稿、未发送邮件。Gmail 行为通过离线接口模拟与审批回归验证。本次功能验收不等于真实邮箱检索质量基准或长期稳定性测试；压缩包中的历史评测 JSON 不作为本次质量结论。

后续启动：在项目目录执行 `.\tasks.ps1 run`；需要独立 MCP 时另执行 `.\tasks.ps1 mcp`。当前后台服务已启动，重复启动前先停止现有实例。Windows 索引完整路径必须仅含 ASCII 字符，代码和模型缓存可以继续保留在当前 E 盘项目目录。

复验命令：

```powershell
.\.venv\Scripts\python.exe scripts/offline_tests.py tests -q -ra --work-dir E:/email-agent-checks
.\.venv\Scripts\python.exe scripts/doctor.py --profile core
```

完整本机日志、覆盖前备份与恢复证据保留在 `E:/email-agent-import-20260912-112731`，未上传到仓库。原有两份未跟踪资料 `data/prompt_injection_testset.json`、`docs/production-readiness-review-2026.md` 继续保留本地，未混入本次提交。
