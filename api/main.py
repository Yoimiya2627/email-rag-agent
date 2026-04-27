import logging
import sys
import os

# Ensure project root is on path when running as `python api/main.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
from core.embedder import index_chunks, clear_collection
from agents.coordinator import route
import config.settings as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
async def index_emails(request: IndexRequest = None):
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


@app.post("/chat", response_model=AgentResponse)
async def chat(request: AgentRequest):
    try:
        return route(request)
    except Exception as exc:
        logger.exception("Chat failed")
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
