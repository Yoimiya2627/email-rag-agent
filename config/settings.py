import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent

# Load .env from project root if present (no-op if missing)
load_dotenv(BASE_DIR / ".env")

# DeepSeek API (called via OpenAI SDK)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Local embedding model (sentence-transformers, no API cost)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ChromaDB
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "emails")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "50"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))

# Data
EMAIL_DATA_PATH = os.getenv("EMAIL_DATA_PATH", str(BASE_DIR / "data" / "emails.json"))

# API server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Frontend
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Feature flags for ablation study
ENABLE_BM25 = os.getenv("ENABLE_BM25", "true").lower() == "true"
ENABLE_RRF = os.getenv("ENABLE_RRF", "true").lower() == "true"
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "false").lower() == "true"

# Timeout and degradation
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "15"))

# Agent loop
# Planning / tool-selection calls use a non-reasoning model — faster and with
# function calling verified (see docs/agent_loop_decisions.md, Step 0).
AGENT_PLANNER_MODEL = os.getenv("AGENT_PLANNER_MODEL", "deepseek-chat")
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))
