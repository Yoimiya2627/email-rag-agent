import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests

import config.settings as cfg

API_URL = cfg.API_URL

st.set_page_config(
    page_title="邮件智能助手",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict = None, timeout: int = 90) -> dict:
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接到后端服务，请先启动 API 服务器"}
    except Exception as exc:
        return {"error": str(exc)}


def _get(endpoint: str, timeout: int = 5) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=timeout)
        return r.json()
    except Exception:
        return None


def _render_metadata(metadata: dict | None):
    """Render assistant-message metadata — agent tool-call steps, or stats."""
    if not metadata:
        return
    steps = metadata.get("steps")
    if steps:
        with st.expander(f"🛠️ Agent 工具调用（{len(steps)} 步）", expanded=False):
            for i, s in enumerate(steps, 1):
                blocked = "  ⚠️ 重复调用被拦截" if s.get("blocked") else ""
                st.markdown(f"**{i}. `{s.get('tool', '?')}`**{blocked}")
                if s.get("arguments"):
                    st.json(s["arguments"])
            if metadata.get("max_steps_reached"):
                st.caption("⚠️ 已达到最大步数上限")
        return
    if not (metadata.get("top5_senders") or metadata.get("label_distribution")
            or metadata.get("daily_counts")):
        return
    with st.expander("📊 统计数据", expanded=False):
        top5 = metadata.get("top5_senders", [])
        if top5:
            st.subheader("发件人 Top 5")
            max_count = max((i.get("count", 1) if isinstance(i, dict) else i[1]) for i in top5)
            for item in top5:
                sender = item.get("sender", "?") if isinstance(item, dict) else item[0]
                count = item.get("count", 0) if isinstance(item, dict) else item[1]
                st.progress(count / max_count, text=f"{sender}：{count} 封")
        label_dist = metadata.get("label_distribution", {})
        if label_dist:
            st.subheader("标签分布")
            st.bar_chart(label_dist)
        daily = metadata.get("daily_counts", [])
        if daily:
            st.subheader("每日邮件量")
            chart_data = {
                (d["date"] if isinstance(d, dict) else d[0]):
                (d["count"] if isinstance(d, dict) else d[1])
                for d in daily
            }
            st.bar_chart(chart_data)


INTENT_LABELS = {
    "retrieve": "🔍 检索",
    "summarize": "📝 摘要",
    "write_reply": "✉️ 回复草稿",
    "analyze": "📊 统计分析",
    "general": "💬 问答",
}

QUICK_QUESTIONS = [
    "最近有哪些重要邮件？",
    "帮我总结一下所有邮件",
    "谁给我发邮件最多？",
    "有什么待回复的邮件吗？",
    "请分析最近的邮件标签分布",
    "帮我回复关于项目进度的邮件",
]

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📧 邮件智能助手")
    st.caption("基于 RAG + Multi-Agent 架构")
    st.divider()

    # API status
    health = _get("/health")
    if health:
        st.success(f"✅ 已连接  |  模型: `{health.get('model', '?')}`")
        status = _get("/index/status")
        if status:
            ec = status.get("email_count", 0)
            cc = status.get("chunk_count", 0)
            if cc:
                st.info(f"📚 已索引 **{ec}** 封邮件 / **{cc}** 个片段")
            else:
                st.warning("⚠️ 索引为空，请先点击下方 “索引邮件”")
    else:
        st.error("❌ 后端未连接，请运行：\n`uvicorn api.main:app --reload`")

    st.divider()
    st.subheader("📂 数据管理")
    data_path = st.text_input(
        "邮件数据路径",
        value="./data/emails.json",
        help="JSON 文件路径，相对于项目根目录",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗂️ 索引邮件", use_container_width=True):
            with st.spinner("正在索引，首次运行需下载嵌入模型…"):
                result = _post("/index", {"data_path": data_path}, timeout=300)
            if result.get("error"):
                st.error(result["error"])
            elif result.get("success"):
                st.success(result["message"])
            else:
                st.error(result.get("message", "索引失败"))
    with col2:
        if st.button("🗑️ 清除索引", use_container_width=True):
            result = _post("/index/clear")
            if result.get("error"):
                st.error(result["error"])
            elif result.get("success"):
                st.success("索引已清除")

    st.divider()
    st.subheader("⚡ 快捷问题")
    for q in QUICK_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"qk_{q}"):
            st.session_state["pending_query"] = q

    st.divider()
    st.subheader("⚙️ 模式")
    mode = st.radio(
        "问答模式",
        ["普通（多 Agent 路由）", "Self-RAG（反思工作流）", "Agent（自主工具调用）"],
        help=(
            "普通=意图路由到专家 agent；"
            "Self-RAG=LangGraph 反思重试；"
            "Agent=function-calling 自主规划并多轮调用工具"
        ),
    )
    use_stream = st.toggle("流式输出", value=False, help="SSE 流式返回 token（仅普通模式）")

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state["messages"] = []
        sid = st.session_state.get("session_id", "default")
        try:
            requests.delete(f"{API_URL}/chat/history", params={"session_id": sid}, timeout=5)
        except Exception:
            pass

# ── main chat area ────────────────────────────────────────────────────────────

st.title("邮件智能问答")
st.caption("支持检索、摘要、回复撰写、统计分析")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state["session_id"] = str(uuid.uuid4())

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("intent"):
            st.caption(f"意图识别：{INTENT_LABELS.get(msg['intent'], msg['intent'])}")
        st.markdown(msg["content"])

        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📎 参考来源（{len(sources)} 条）", expanded=False):
                for i, src in enumerate(sources):
                    m = src.get("metadata", {})
                    st.markdown(
                        f"**[{i+1}]** `{m.get('sender','?')}` · "
                        f"`{m.get('date','?')}` · **{m.get('subject','?')}**"
                    )
                    body = src.get("content", "")
                    st.text(body[:300] + ("…" if len(body) > 300 else ""))
                    if i < len(sources) - 1:
                        st.divider()

        _render_metadata(msg.get("extra_metadata"))

# Handle pending quick query
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("输入你的问题，例如：最近有哪些重要邮件？") or pending

if user_input:
    session_id = st.session_state["session_id"]
    # Show user bubble
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Call API and show assistant bubble
    payload = {"query": user_input, "session_id": session_id}
    if mode.startswith("Agent"):
        endpoint = "/chat/agent"
    elif mode.startswith("Self-RAG"):
        endpoint = "/chat/graph"
    else:
        endpoint = "/chat"

    with st.chat_message("assistant"):
        if use_stream and mode.startswith("普通"):
            # SSE streaming
            import sseclient
            intent_caption = st.empty()
            answer_placeholder = st.empty()
            full_answer = ""
            intent_value = ""
            try:
                with requests.post(
                    f"{API_URL}/chat/stream",
                    json=payload,
                    stream=True,
                    timeout=120,
                ) as resp:
                    client_sse = sseclient.SSEClient(resp)
                    for event in client_sse.events():
                        if event.data == "[DONE]":
                            break
                        import json as _json
                        token_data = _json.loads(event.data)
                        if "intent" in token_data:
                            intent_value = token_data["intent"]
                            label = INTENT_LABELS.get(intent_value, intent_value)
                            intent_caption.caption(f"意图识别：{label}")
                        elif "token" in token_data:
                            full_answer += token_data["token"]
                            answer_placeholder.markdown(full_answer + "▌")
                answer_placeholder.markdown(full_answer)
                # 标记 _streamed，让下方通用渲染分支跳过重复渲染
                result = {"answer": full_answer, "sources": [], "intent": intent_value, "_streamed": True}
            except Exception as exc:
                result = {"error": str(exc)}
        else:
            with st.spinner("思考中…"):
                result = _post(endpoint, payload)

        if result.get("error"):
            st.error(result["error"])
            st.session_state["messages"].append(
                {"role": "assistant", "content": result["error"]}
            )
        else:
            intent = result.get("intent", "")
            answer = result.get("answer", "抱歉，未能生成回答。")

            # 流式模式：placeholder 已渲染答案 + intent caption，跳过下面的重复渲染；
            # 非流式模式：在这里渲染 caption 和答案。
            if not result.get("_streamed"):
                if intent:
                    st.caption(f"意图识别：{INTENT_LABELS.get(intent, intent)}")
                st.markdown(answer)

            sources = result.get("sources", [])
            if sources:
                with st.expander(f"📎 参考来源（{len(sources)} 条）", expanded=False):
                    for i, src in enumerate(sources):
                        m = src.get("metadata", {})
                        st.markdown(
                            f"**[{i+1}]** `{m.get('sender','?')}` · "
                            f"`{m.get('date','?')}` · **{m.get('subject','?')}**"
                        )
                        st.text(
                            src.get("content", "")[:300]
                            + ("…" if len(src.get("content","")) > 300 else "")
                        )
                        if i < len(sources) - 1:
                            st.divider()

            _render_metadata(result.get("metadata"))

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "intent": intent,
                    "sources": sources,
                    "extra_metadata": result.get("metadata"),
                }
            )
