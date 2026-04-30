import asyncio
import json
import logging
import sys
import os
import threading
from collections import defaultdict

# Ensure project root is on path when running as `python api/main.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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
from core.memory import ConversationMemory
from agents.coordinator import route
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
_sessions: dict[str, ConversationMemory] = {}
_sessions_lock = threading.Lock()


def _get_session(session_id: str) -> ConversationMemory:
    with _sessions_lock:
        memory = _sessions.get(session_id)
        if memory is None:
            memory = ConversationMemory()
            _sessions[session_id] = memory
        return memory


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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def chat(request: AgentRequest):
    try:
        session_id = request.session_id or "default"
        memory = _get_session(session_id)
        response = route(request, memory=memory)
        memory.add("user", request.query)
        memory.add("assistant", response.answer)
        return response
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
async def chat_stream(request: AgentRequest):
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
    memory = _get_session(session_id)
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
async def clear_history(session_id: str = "default"):
    with _sessions_lock:
        memory = _sessions.get(session_id)
    if memory is not None:
        memory.clear()
    return {"success": True, "session_id": session_id}


@app.post("/chat/graph", response_model=AgentResponse)
async def chat_graph(request: AgentRequest):
    """Self-RAG workflow via LangGraph-style state machine."""
    from agents.graph_workflow import run_graph
    try:
        session_id = request.session_id or "default"
        memory = _get_session(session_id)
        response = run_graph(request, memory=memory)
        memory.add("user", request.query)
        memory.add("assistant", response.answer)
        return response
    except Exception as exc:
        logger.exception("Graph chat failed")
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
