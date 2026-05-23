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
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "cross_encoder").lower()
RERANK_INPUT_CHAR_LIMIT = int(os.getenv("RERANK_INPUT_CHAR_LIMIT", "1200"))
GENERATION_CONTEXT_CHAR_LIMIT = int(os.getenv("GENERATION_CONTEXT_CHAR_LIMIT", "6000"))

# Cross-encoder reranker. Loaded lazily only when
# ENABLE_RERANKER=true and RERANKER_BACKEND=cross_encoder.
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
CROSS_ENCODER_DEVICE = os.getenv("CROSS_ENCODER_DEVICE", EMBEDDING_DEVICE)
CROSS_ENCODER_MAX_LENGTH = int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512"))

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
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# Agent loop
# Planning / tool-selection calls use a non-reasoning model — faster and with
# function calling verified (see docs/agent_loop_decisions.md, Step 0).
AGENT_PLANNER_MODEL = os.getenv("AGENT_PLANNER_MODEL", "deepseek-chat")
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))
# Loop-safety: block a tool re-called with identical args beyond this count.
AGENT_MAX_REPEAT = int(os.getenv("AGENT_MAX_REPEAT", "2"))
# Truncate any single tool result longer than this (chars) to bound context growth.
AGENT_TOOL_OUTPUT_LIMIT = int(os.getenv("AGENT_TOOL_OUTPUT_LIMIT", "4000"))
# Output-token ceiling for the agent's final-answer generation. A tool-selection
# turn only emits short tool_calls, so a long multi-step synthesis answer would
# silently truncate at a low cap. max_tokens is a ceiling, not a reservation —
# raising it costs nothing on short turns.
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4000"))
AGENT_TOOL_BACKEND = os.getenv("AGENT_TOOL_BACKEND", "local").lower()

# MCP client/server defaults. The local function-calling backend remains the
# default; this URL is only used when AGENT_TOOL_BACKEND=mcp.
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://{MCP_HOST}:{MCP_PORT}/mcp")
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "")
MCP_TOOL_SCHEMA_CACHE_SECONDS = int(os.getenv("MCP_TOOL_SCHEMA_CACHE_SECONDS", "60"))
ENABLE_MCP_AUDIT = os.getenv("ENABLE_MCP_AUDIT", "true").lower() == "true"
MCP_AUDIT_LOG_PATH = os.getenv(
    "MCP_AUDIT_LOG_PATH",
    str(BASE_DIR / "data" / "audit" / "mcp_audit.jsonl"),
)
MCP_ALLOWED_TOOLS = [
    item.strip()
    for item in os.getenv("MCP_ALLOWED_TOOLS", "").split(",")
    if item.strip()
]
MCP_READ_ONLY_MODE = os.getenv("MCP_READ_ONLY_MODE", "false").lower() == "true"

# Human-in-the-loop approval store for high-risk tools.
APPROVAL_STORE_PATH = os.getenv(
    "APPROVAL_STORE_PATH",
    str(BASE_DIR / "data" / "approvals" / "pending_actions.json"),
)

# Approved mail execution. The default remains simulated; gmail creates a real
# Gmail draft after human approval and never sends by default.
MAIL_PROVIDER = os.getenv("MAIL_PROVIDER", "simulated").lower()
GMAIL_CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH", str(BASE_DIR / "credentials" / "gmail_credentials.json"))
GMAIL_TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH", str(BASE_DIR / "credentials" / "gmail_token.json"))
GMAIL_SCOPES = [
    item.strip()
    for item in os.getenv("GMAIL_SCOPES", "https://www.googleapis.com/auth/gmail.compose").split(",")
    if item.strip()
]
GMAIL_USER_ID = os.getenv("GMAIL_USER_ID", "me")
ENABLE_REAL_EMAIL_SEND = os.getenv("ENABLE_REAL_EMAIL_SEND", "false").lower() == "true"

# Agent trace events. Disabled by default to keep local unit tests and demos
# quiet; enable when running evals or production-like debugging.
ENABLE_AGENT_TRACE = os.getenv("ENABLE_AGENT_TRACE", "false").lower() == "true"
AGENT_TRACE_LOG_PATH = os.getenv(
    "AGENT_TRACE_LOG_PATH",
    str(BASE_DIR / "data" / "traces" / "agent_traces.jsonl"),
)
