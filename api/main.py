import asyncio
import json
import logging
import sys
import os
import threading
import time

# Ensure project root is on path when running as `python api/main.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from models.schemas import (
    AgentRequest,
    AgentResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
)
from core.loader import load_emails
from core.cleaner import clean_email
from core.chunker import chunk_email
from core.embedder import index_chunks, clear_collection, get_collection_stats
from core.session_store import create_session_store_from_settings
from agents.coordinator import route
from agents.approvals import ApprovalStore
from agents.mail_providers import MailProviderError, create_mail_provider_from_settings
from agents.mcp_adapter import MCPAuditLogger
import config.settings as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# In-memory session store: session_id → ConversationMemory.
# defaultdict.__getitem__ creates entries non-atomically across threads, so
# requests for an unseen session_id can race and clobber each other's memory.
# Wrap with a lock for the lookup-or-create path.
_session_store = create_session_store_from_settings()
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class FixedWindowRateLimiter:
    def __init__(self):
        self._buckets: dict[tuple[str, str], tuple[int, int]] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()

    def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        current_window = int(time.time() // max(window_seconds, 1))
        bucket_key = (key, str(current_window))
        with self._lock:
            count, _ = self._buckets.get(bucket_key, (0, current_window))
            if count >= limit:
                return False
            self._buckets[bucket_key] = (count + 1, current_window)
            return True


_rate_limiter = FixedWindowRateLimiter()


def _get_session(session_id: str, tenant_id: str | None = None):
    return _session_store.get(tenant_id or cfg.DEFAULT_TENANT_ID, session_id)


def _tenant_id_from_request(request: Request) -> str:
    tenant_id = request.headers.get("X-Tenant-ID", "").strip()
    return tenant_id or cfg.DEFAULT_TENANT_ID


def _auth_identity(request: Request) -> str:
    auth = request.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if request.client:
        return request.client.host
    return "anonymous"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Email RAG API starting up")
    yield
    logger.info("Email RAG API shut down")


app = FastAPI(
    title="Email RAG API",
    description="Intelligent email Q&A with multi-agent architecture",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def production_guardrails(request: Request, call_next):
    request.state.tenant_id = _tenant_id_from_request(request)
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    if cfg.API_AUTH_TOKEN:
        expected = f"Bearer {cfg.API_AUTH_TOKEN}"
        if request.headers.get("Authorization", "") != expected:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    if cfg.RATE_LIMIT_ENABLED:
        key = f"{request.state.tenant_id}:{_auth_identity(request)}"
        if not _rate_limiter.allow(key, cfg.RATE_LIMIT_REQUESTS, cfg.RATE_LIMIT_WINDOW_SECONDS):
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

    return await call_next(request)


@app.get("/health")
async def health():
    return {"status": "ok", "model": cfg.DEEPSEEK_MODEL}


@app.post("/index", response_model=IndexResponse)
async def index_emails(request: Optional[IndexRequest] = Body(None)):
    try:
        path = (request.data_path if request else None) or None
        emails = load_emails(path)
        all_chunks = []
        for email in emails:
            cleaned = clean_email(email)
            all_chunks.extend(chunk_email(cleaned))
        count = index_chunks(all_chunks)
        # Invalidate BM25 cache so the next query rebuilds it over the new corpus
        from core.retriever import invalidate_bm25_cache
        invalidate_bm25_cache()
        return IndexResponse(
            success=True,
            message=f"Indexed {len(emails)} emails into {count} chunks",
            count=count,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/index/clear")
async def clear_index():
    try:
        clear_collection()
        from core.retriever import invalidate_bm25_cache
        invalidate_bm25_cache()
        return {"success": True, "message": "Index cleared"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/index/status")
async def index_status():
    try:
        return get_collection_stats()
    except Exception as exc:
        logger.exception("Status fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat", response_model=AgentResponse)
async def chat(request: AgentRequest, http_request: Request):
    try:
        session_id = request.session_id or "default"
        memory = _get_session(session_id, http_request.state.tenant_id)
        response = route(request, memory=memory)
        memory.add("user", request.query)
        memory.add("assistant", response.answer)
        return response
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
async def chat_stream(request: AgentRequest, http_request: Request):
    """SSE streaming chat — yields tokens as `data: <token>\\n\\n` events.

    Routes through the Coordinator's intent classifier so summarize/write/
    analyze queries land on the right agent instead of being forced through
    the retriever pipeline. Token-level streaming applies to retrieve/general
    intents; other agents emit their final answer as a single event because
    they don't expose a streaming generator.
    """
    from core.generator import stream_generate
    from agents.retriever_agent import RetrieverAgent
    from agents.coordinator import classify_intent, route
    from models.schemas import IntentType

    session_id = request.session_id or "default"
    memory = _get_session(session_id, http_request.state.tenant_id)
    history = memory.to_messages()

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()
        loop = asyncio.get_running_loop()

        def producer():
            try:
                intent = classify_intent(request.query)
                # Tell the client which agent picked this up so the UI can
                # render an appropriate header / spinner.
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("__intent__", intent.value)
                )

                if intent in (IntentType.RETRIEVE, IntentType.GENERAL):
                    # Token-level streaming via the retriever pipeline.
                    agent = RetrieverAgent()
                    contexts = agent.prepare_contexts(request.query)
                    for token in stream_generate(
                        request.query, contexts, history=history
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                else:
                    # Non-retrieve intents (summarize/write/analyze) don't
                    # expose a streaming generator — run the agent and emit
                    # its full answer as one event.
                    response = route(request, memory=None)
                    loop.call_soon_threadsafe(queue.put_nowait, response.answer)
            except Exception as exc:
                logger.exception("Stream producer failed")
                loop.call_soon_threadsafe(queue.put_nowait, ("__error__", str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        loop.run_in_executor(None, producer)

        full_answer = ""
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if isinstance(item, tuple) and item:
                tag = item[0]
                if tag == "__error__":
                    yield f"data: {json.dumps({'error': item[1]})}\n\n"
                    return
                if tag == "__intent__":
                    yield f"data: {json.dumps({'intent': item[1]})}\n\n"
                    continue
            full_answer += item
            yield f"data: {json.dumps({'token': item}, ensure_ascii=False)}\n\n"

        memory.add("user", request.query)
        memory.add("assistant", full_answer)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/chat/history")
async def clear_history(request: Request, session_id: str = "default"):
    _session_store.clear(request.state.tenant_id, session_id)
    return {"success": True, "session_id": session_id, "tenant_id": request.state.tenant_id}


@app.get("/agent/approvals")
async def list_agent_approvals(request: Request, status: Optional[str] = None):
    """List pending/approved/rejected high-risk agent actions."""
    return {"approvals": ApprovalStore(tenant_id=request.state.tenant_id).list(status=status)}


@app.post("/agent/approvals/{approval_id}/approve")
async def approve_agent_action(
    approval_id: str,
    request: Request,
    reviewer: str = Body("human"),
    note: str = Body(""),
):
    """Approve a pending high-risk agent action.

    The default implementation records a simulated result.  When MAIL_PROVIDER
    is set to gmail, this endpoint creates a Gmail draft and does not send it.
    """
    try:
        provider = create_mail_provider_from_settings()
        return ApprovalStore(tenant_id=request.state.tenant_id).approve(
            approval_id,
            reviewer=reviewer,
            note=note,
            executor=provider.execute_approval,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except MailProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/agent/approvals/{approval_id}/reject")
async def reject_agent_action(
    approval_id: str,
    request: Request,
    reviewer: str = Body("human"),
    note: str = Body(""),
):
    """Reject a pending high-risk agent action."""
    try:
        return ApprovalStore(tenant_id=request.state.tenant_id).reject(
            approval_id,
            reviewer=reviewer,
            note=note,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/agent/mcp-audit")
async def list_mcp_audit_events(
    tool: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
):
    """List MCP tool-call audit events for operational review."""
    events = MCPAuditLogger.load_events(tool=tool, status=status, limit=limit)
    return {"count": len(events), "events": events}


@app.post("/chat/graph", response_model=AgentResponse)
async def chat_graph(request: AgentRequest, http_request: Request):
    """Self-RAG workflow via LangGraph-style state machine."""
    from agents.graph_workflow import run_graph
    try:
        session_id = request.session_id or "default"
        memory = _get_session(session_id, http_request.state.tenant_id)
        response = run_graph(request, memory=memory)
        memory.add("user", request.query)
        memory.add("assistant", response.answer)
        return response
    except Exception as exc:
        logger.exception("Graph chat failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/agent", response_model=AgentResponse)
async def chat_agent(request: AgentRequest, http_request: Request):
    """Function-calling agent loop — the planner LLM autonomously selects and
    chains tools (search / get / summarize / draft / stats) to fulfil the query."""
    from agents.agent_loop import run_agent_loop
    try:
        session_id = request.session_id or "default"
        memory = _get_session(session_id, http_request.state.tenant_id)
        response = run_agent_loop(request, memory=memory)
        memory.add("user", request.query)
        memory.add("assistant", response.answer)
        return response
    except Exception as exc:
        logger.exception("Agent chat failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Direct RAG query without multi-agent routing."""
    from core.retriever import hybrid_search
    from core.reranker import rerank
    from core.generator import generate_answer

    try:
        results = hybrid_search(request.query, top_k=request.top_k)
        reranked = rerank(request.query, results)
        answer = generate_answer(request.query, reranked)
        return QueryResponse(answer=answer, sources=reranked)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=cfg.API_HOST, port=cfg.API_PORT, reload=True)
