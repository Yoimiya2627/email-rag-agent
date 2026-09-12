# 个人单机部署

支持边界是一个 owner、一个 API 实例、一个 worker。审批的 SQLite 去重不等于会话锁、模型缓存和后台任务能够跨进程协调。不要使用 `--workers 2`、多副本或开发热重载运行持久后台任务。API 重启后中断任务需要显式恢复；不确定的邮箱执行结果需要人工核对。

## 本机准备

Windows 的 Chroma 索引目录必须使用完整 ASCII 路径，例如在 `.env` 中设置 `CHROMA_PERSIST_DIR=E:/email-agent-runtime/chroma_db`。项目代码可以放在中文目录，但索引实际解析后的路径不能含中文等非 ASCII 字符。已实测 Chroma 1.5.8 在此类路径中写进程计数正常，却未写出 HNSW 二进制文件，重启后索引不可读；复制这样的坏库到英文目录不能补回缺失文件。请保留原库供恢复检查，在 ASCII 目录恢复或重新索引。API 启动和存储初始化会拒绝不支持的路径，`doctor` 也会报告配置错误。

`/warmup` 会实际读取已发布的索引并核对数量，再初始化模型；`/ready` 每次检查索引是否可读且数量与 manifest 一致。只有 manifest 存在不代表索引就绪。预热与就绪检查不会调用付费 LLM 或邮箱 API。

在独立 `.venv` 中安装依赖，复制 `.env.example` 为 `.env` 后填写自己的设置。`tasks.ps1 install` / `make install` 只安装 Python 依赖。`tasks.ps1 preload`、`make preload` 或 `python scripts/preload_model.py` 才显式准备嵌入模型；首次索引/预热也可能触发下载。模型体积和耗时取决于选择，不承诺固定下载大小。设置 `EMBEDDING_MODEL`、经过验证的 `EMBEDDING_MODEL_REVISION` 与 `HF_ENDPOINT`；端点应由使用者选择，默认官方地址。不同 revision/维度与旧索引不兼容时需按索引迁移流程显式处理。

`tasks.ps1 doctor` 只检查配置形状、依赖和路径存在性；不会自动下载、打开 OAuth 或证明模型/邮箱就绪。`tasks.ps1 test` 使用隔离的离线 runner。`tasks.ps1 clean` 默认预览缓存清理，`-Apply` 才删除；索引目录还必须显式指定 `-IncludeIndex`。该开关仅指项目内 `chroma_db`，不代替自定义状态路径的维护流程。先备份再清理。Linux 的 `make clean` 是旧的破坏性清理目标，不属于留存预览工具；维护应优先使用下述备份/留存脚本。

## Docker Compose

先提供 `.env` 中的 `DEEPSEEK_API_KEY` 和非空随机 `API_AUTH_TOKEN`，准备 `./data/emails.json` 与持久目录，再执行：

```text
docker compose build
docker compose up -d
```

默认构建不下载模型权重，也不将邮件数据或 `.env`/凭据/SQLite 打入镜像。若明确希望构建时准备权重，设置 `PRELOAD_EMBEDDING_MODEL=true`，同时配置模型、revision 和 `HF_ENDPOINT` 后重新构建。不要将 API key、HF token 或 OAuth secret 放入 Docker build args；本实现未添加 BuildKit secret 下载流程。模型缓存使用 `hf_cache` 卷；已有卷会保留自己的内容，不会因重建镜像自动替换所有缓存。

Dockerfile 目前从 `requirements.txt` 安装范围依赖；本地 constraints 是 Windows 环境实测版本闭包，不是已经验证的 Linux/ML 镜像锁。目标镜像需要执行 `pip check`、离线用例和索引兼容性检查后保存自己的精确版本记录。不要把镜像可构建等同于依赖或模型质量已验收。

Compose 将 API 与 Streamlit 端口分别绑定到宿主机 `127.0.0.1:8000`、`127.0.0.1:8501`，容器内 API 使用 `0.0.0.0:8000` 并显式单 worker。`/health` 仅表示进程存活；`/ready` 另外检查预热与索引状态，仍不代表远端模型网络验证成功。Compose 依赖健康检查只等待存活，不会自动预热模型。

API 服务通过 `env_file: .env` 读取运行时配置，Compose 再覆盖所有状态路径：语料/审批/会话/任务/同步/日志在 `/app/data`，索引在 `/app/chroma_db`，缓存在 `/root/.cache/huggingface`。宿主机 `.env` 中的 Windows 绝对路径不会成为这些状态文件的容器路径。挂载的是整个 `./data` 和 `./chroma_db`，请保持项目目录私有。前端容器只接收 API URL、API token 和请求时长，不接收模型 key 或 Gmail 配置。此处的 `.env` 与环境变量是明文 secret 来源，不是专门的密钥管理系统；不要把 `docker compose config` 或容器环境的输出分享出去。

当前 Compose 默认未挂载 Gmail 凭据。需要 Gmail 草稿时，先在本机运行显式授权命令 `python -m agents.mail_providers authorize`，准备有效 compose token，再使用本地 Compose override 将准备好的凭据目录只读挂载到 `/run/gmail`，并设置 `MAIL_PROVIDER=gmail`：

```yaml
services:
  api:
    volumes:
      - ./credentials:/run/gmail:ro
```

该目录需由本人预先准备且不进入构建上下文。审批阶段只使用已就绪 token，不打开浏览器、不刷新或写回 token；过期时回到本机显式授权。不要在只读挂载的容器内执行需要写入 OAuth token 的同步/授权命令。只读 Gmail 同步与 compose 草稿的 token/scopes 分离。`ENABLE_REAL_EMAIL_SEND` 不会启用真实发送；当前 provider 只创建 draft。MCP 服务未包含在默认 Compose 中，如自行部署须保持可信 owner、一致工具策略、独立服务端状态和日志，并为跨容器地址显式配置 `MCP_SERVER_URL`。

## 维护与验收

停掉 API、UI、MCP 和后台 worker 后，用 [完整状态包](state-retention-and-recovery.md) 一起保存语料、raw 目录、索引目录与审批/任务/会话 SQLite；恢复只写入新目录，原路径不会被覆盖。模型缓存与 OAuth secret 不会自动加入包。核对 manifest 与远端草稿结果后，再显式切换配置或挂载。

本次验证包括离线 Python、配置解析和临时目录 PowerShell 探针；未执行 Docker 构建/启动、托管 CI、真实邮箱/模型或 24 小时运行。目标环境应按 [离线验证](offline-validation.md) 和 [服务性能实验](service-performance-experiments.md) 分层验收，保存实际结果与未验证项。
