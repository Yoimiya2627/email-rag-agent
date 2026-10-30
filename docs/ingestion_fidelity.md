# 邮件正文格式与索引回放

`Email.body_format` 接受 `plain`（默认）或 `html`。历史 JSON 不含此字段时按 plain 处理，不从正文尖括号猜测 HTML。邮箱地址、比较符、`<p>` 示例及 plain 正文的 `&amp;` 均是字面内容。

明确 HTML 的调用入口：

```python
from core.cleaner import clean_body, clean_email
from models.schemas import Email

text = clean_body("<p>A &amp; B</p><p>下一段</p>", body_format="html")
email = Email(**{**record, "body_format": "html"})
normalized = clean_email(email)
assert normalized.body_format == "plain"
```

HTML 使用标准库解析器解码实体、去掉 script/style/title/template 与 head 元数据，并保留段落、换行和 hr 分隔。省略 `</head>` 时，遇到 body、正文标签或可见正文文本会结束元数据区域，避免整封正文被隐藏。`clean_email` 的输出标为 plain，再次清洗不重复解析 HTML 或解码实体。

清洗保留 Thanks、谢谢、Regards、祝好等开场或结束用语及其后文，不根据这些词猜测签名。只有出现在非空正文之后、单独一行的精确 `-- `（两个减号及一个空格）才触发 plain-text 签名裁剪；裸 `--` 不触发。仍进行空白规范化，因此索引详情是清洗后文本，不等同原始 MIME。

旧 JSON 中若实际储存 HTML，需要依据数据来源确认后为相应记录补 `"body_format": "html"`，再运行原有索引流程。不要给已经 Gmail 标准化的正文补 html。已被旧清洗逻辑删掉的文本无法从旧 chunks 恢复，应从原始 JSON 或邮箱重新摄取。

Gmail 转换按 MIME 树分层选择：alternative 保留优先选择的 plain 正文，并补充 HTML 备选版本中的表格，明确标记备选来源，不重复全部 HTML 正文；两种版本冲突时仍需核对原件。mixed 保留非附件正文顺序，related 仅使用指定根或第一部分；具 filename、Content-Type name 或 attachment disposition 的子树不充当正文，message/rfc822 也不替换父邮件正文。遵守 Content-Type charset，非法编码明确报错。合法正文单独放在 Gmail attachmentId 时，provider 按需读取该正文；附件文件不读取。`GmailReadOnlyProvider.message_to_email(raw_message)` 复用已获取的消息，并提供这种按需正文加载。纯转换函数 `gmail_message_to_email` 如遇单独正文，需要调用者提供 `attachment_loader`。

主题中的 RFC 2047 encoded-word 使用标准库解码，支持与普通文本混排；普通主题保持原文，遇未知字符集或无法解码的字节时保留完整原始主题，不丢失邮件。

新 chunk 都是同一清洗后 `Subject: ...\n\n正文` 的连续切片。metadata 中 source_start/source_end 是 Python 字符偏移，source_length/source_sha256 验证完整文本。短尾通过延长上一块区间合并，不把重叠文字再追加一次。详情回放使用偏移，不依赖当前 CHUNK_OVERLAP，也不依据内容相同猜测去重。

详情结果 `reconstruction_exact=true` 表示区间、重叠内容、总长度和摘要全部验证通过。旧索引缺少这些元数据或 chunk 损坏、缺失时，结果保留全部 chunk 内容，`reconstruction_exact=false`、`reindex_required=true`；此时可能仍含旧重叠，应重新索引获得精确回放。

离线验证：`python -m unittest tests.test_ingestion_fidelity tests.test_retrieval_contracts -q`。无需真实邮箱、模型或网络。

## 2026-09-10：表格和同步恢复

`html_to_structured_text(html)` 返回规范化正文与 `table_rows`。表格行文本带表头、来源 ID 和合并单元格关系；Email 的 `table_rows` 保存行区间与行列信息。调用 `clean_email` 再 `chunk_email` 会保留这些结构。只用 `clean_body` 可以得到可读文本，但拿不到结构元数据。

按行切块；超过 CHUNK_SIZE 的行分为有明确 `partial_row` 标记的片段，metadata.table_context 是 JSON 字符串。普通生成、摘要和搜索工具把来源信息带给模型；搜索短摘要和生成预算二次截断也标为部分行。片段不足以确认事实时，应按邮件 ID 读取完整索引正文。上述 exact 校验仍指规范化文本，而非 Gmail 原始字节。

解析不是浏览器排版引擎：嵌套表格会降级并标记，某些显式表头引用暂不支持。限制包括每个 HTML 部分最多 4,000,000 输入字符、1,024 张表、4,096 条规范化表格记录及 4,000,000 表格输出字符；单表另有限制。触发上限会保留可见的不完整标记和已处理部分，不冒充完整解析。

同步在开始前保存待处理 ID 队列，默认每处理 25 封检查点保存。每封邮件转换前先原子保存 Gmail full JSON，再补存选中的外置正文数据。归档格式为 `gmail-full-v1`，默认目录为 `OUTPUT.raw`，文件名为 Gmail ID 的 SHA-256。它不是 RFC822/.eml 全量导出，也不含普通附件二进制。`provider.email_from_capture(envelope)` 可在离线状态重新解析已归档的正文；缺少必要字节时明确报错。

损坏正文或结构会记录安全错误码并继续其他邮件，失败 ID 不进入 seen；相同 output/query 下，下次即使列表窗口已不包含失败邮件，也会优先重试。网络/授权等任务级故障会停止，尽可能提交已成功处理的部分。硬退出后当前批次可能重新拉取，但已提交 corpus 是去重依据。归档文件保留的内容不会自动替代后续 Gmail 重试；自动离线重建任务尚未实现。

失败清单、待处理 ID 和 complete/partial/aborted 状态存于 state JSON，CLI 输出 status_path。部分成功或任务失败时 CLI 返回非零，不执行 `--index` 或 `--clear-index`。格式不完整标记与同步转换失败是不同情况：前者可作为明确不完整的证据索引。

单页大小最多 500，`--max-results` 是本次列表总量上限，另有恢复队列，故恢复时处理量可大于列表上限。读取对 429、部分 5xx、内置超时/连接异常进行最多 3 次尝试；其他错误停止。当前仍串行拉取、整份 JSON 检查点落盘，不是完整 historyId 增量同步或大容量数据库迁移。

回到有凭据的运行环境后，可先用新文件做小范围试用（在项目根目录执行）：

```powershell
python scripts/sync_gmail_readonly.py --output data/real_emails/gmail-v2.json --state-path data/mail_sync/gmail-v2-state.json --query "newer_than:30d" --max-results 100 --checkpoint-size 25
```

这条命令只同步最近 30 天列表中的最多 100 封，并非全邮箱导入。归档包含私人邮件内容，使用上述已忽略的本地数据目录；自定义 raw_dir 时也应保存在私人数据目录。单个 corpus/raw/state 组合仅供同一邮箱账号使用。

旧 Gmail JSON 如果已经失去 HTML 结构，原地重建索引不能恢复表格；请保留旧数据，用新的 output/state 从邮箱重新同步目标范围并验收。正式替换索引前确认新 corpus 覆盖需要保留的全部范围；不要把这 100 封试用样本直接作为完整替换输入。

新增离线验证：

```powershell
python -B -m unittest tests.test_gmail_capture tests.test_sync_recovery tests.test_sync_integrity tests.test_table_fidelity tests.test_real_mail_pipeline tests.test_ingestion_fidelity -v
```

真实 Gmail、模型推理、Chroma 持久化和容量压测仍需在目标环境单独验收。
