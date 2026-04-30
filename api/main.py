import asyncio
import json
import logging
import sys
import os
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


# In-memory session store: session_id → ConversationMemory
_sessions: dict[str, ConversationMemory] = defaultdict(ConversationMemory)


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
        memory = _sessions[session_id]
        response = route(request, memory=memory)
        memory.add("user", request.query)
        memory.add("assistant", response.answer)
        return response
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
async def chat_stream(request: AgentRequest):
    """SSE streaming chat — yields tokens as `data: <token>\\n\\n` events."""
    from core.retriever import hybrid_search
    from core.reranker import rerank
    from core.generator import stream_generate
    from agents.retriever_agent import RetrieverAgent

    session_id = request.session_id or "default"
    memory = _sessions[session_id]
    history = memory.to_messages()

    async def event_generator():
        try:
            agent = RetrieverAgent()
            rewritten = agent._rewrite_query(request.query)
            filters = agent._extract_filters(rewritten)
            search_query = filters.get("query") or rewritten

            results = hybrid_search(search_query, top_k=cfg.TOP_K * 4)
            reranked = rerank(request.query, results, top_n=cfg.RERANK_TOP_N)

            full_answer = ""
            loop = asyncio.get_event_loop()
            tokens = await loop.run_in_executor(
                None,
                lambda: list(stream_generate(request.query, reranked, history=history)),
            )
            for token in tokens:
                full_answer += token
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            memory.add("user", request.query)
            memory.add("assistant", full_answer)
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("Stream chat failed")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/chat/history")
async def clear_history(session_id: str = "default"):
    if session_id in _sessions:
        _sessions[session_id].clear()
    return {"success": True, "session_id": session_id}


@app.post("/chat/graph", response_model=AgentResponse)
async def chat_graph(request: AgentRequest):
    """Self-RAG workflow via LangGraph-style state machine."""
    from agents.graph_workflow import run_graph
    try:
        session_id = request.session_id or "default"
        memory = _sessions[session_id]
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
