import os
import json
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent

# Load .env from project root if present (no-op if missing)
load_dotenv(BASE_DIR / ".env")

# DeepSeek API (called via OpenAI SDK)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_THINKING_MODE = os.getenv('DEEPSEEK_THINKING_MODE', 'disabled').strip().lower()
if DEEPSEEK_THINKING_MODE not in {'enabled', 'disabled'}:
    raise ValueError('DEEPSEEK_THINKING_MODE must be enabled or disabled')

# Local embedding model (sentence-transformers, no API cost)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_CPU_THREADS = int(os.getenv("EMBEDDING_CPU_THREADS", "0"))

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
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")
API_OWNER_ID = os.getenv("API_OWNER_ID", "local")
API_CORS_ORIGINS = [v.strip() for v in os.getenv("API_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if v.strip()]
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))
RETRIEVAL_TIMEZONE_OFFSET_HOURS = int(os.getenv("RETRIEVAL_TIMEZONE_OFFSET_HOURS", "8"))

# Frontend
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Feature flags for ablation study
ENABLE_BM25 = os.getenv("ENABLE_BM25", "true").lower() == "true"
ENABLE_RRF = os.getenv("ENABLE_RRF", "true").lower() == "true"
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "false").lower() == "true"

# Timeout and degradation
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
EVAL_JUDGE_TIMEOUT_SECONDS = float(os.getenv('EVAL_JUDGE_TIMEOUT_SECONDS',str(min(LLM_TIMEOUT,30))))
EVAL_JUDGE_CONTEXT_CHAR_LIMIT = int(os.getenv('EVAL_JUDGE_CONTEXT_CHAR_LIMIT','24000'))
EVIDENCE_VERIFY_CHAR_LIMIT = int(os.getenv('EVIDENCE_VERIFY_CHAR_LIMIT','60000'))

# Agent loop
# Planning / tool-selection calls use a non-reasoning model — faster and with
# function calling verified (see docs/agent_loop_decisions.md, Step 0).
AGENT_PLANNER_MODEL = os.getenv("AGENT_PLANNER_MODEL", "deepseek-chat")
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))
AGENT_MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "12"))
AGENT_RUN_TIMEOUT = float(os.getenv("AGENT_RUN_TIMEOUT", "120"))
AGENT_CONTEXT_CHAR_LIMIT = int(os.getenv("AGENT_CONTEXT_CHAR_LIMIT", "60000"))
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
MCP_OWNER_ID = os.getenv("MCP_OWNER_ID", API_OWNER_ID)
MCP_TOOL_SCHEMA_CACHE_SECONDS = int(os.getenv("MCP_TOOL_SCHEMA_CACHE_SECONDS", "60"))
ENABLE_MCP_AUDIT = os.getenv("ENABLE_MCP_AUDIT", "true").lower() == "true"
MCP_AUDIT_LOG_PATH = os.getenv(
    "MCP_AUDIT_LOG_PATH",
    str(BASE_DIR / "data" / "audit" / "mcp_audit.jsonl"),
)
MCP_SERVER_AUDIT_LOG_PATH = os.getenv('MCP_SERVER_AUDIT_LOG_PATH',str(Path(MCP_AUDIT_LOG_PATH).with_name('mcp_server.jsonl')))
MCP_ALLOWED_TOOLS = [
    item.strip()
    for item in os.getenv("MCP_ALLOWED_TOOLS", "").split(",")
    if item.strip()
]
MCP_READ_ONLY_MODE = os.getenv("MCP_READ_ONLY_MODE", "false").lower() == "true"

# Human-in-the-loop approval store for high-risk tools.
APPROVAL_TTL_SECONDS = int(os.getenv("APPROVAL_TTL_SECONDS", "86400"))
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

# Gmail read-only ingestion. Kept separate from the draft provider so read-only
# sync never needs compose/send scopes.
GMAIL_READONLY_TOKEN_PATH = os.getenv(
    "GMAIL_READONLY_TOKEN_PATH",
    str(BASE_DIR / "credentials" / "gmail_readonly_token.json"),
)
GMAIL_READONLY_SCOPES = [
    item.strip()
    for item in os.getenv("GMAIL_READONLY_SCOPES", "https://www.googleapis.com/auth/gmail.readonly").split(",")
    if item.strip()
]
GMAIL_SYNC_QUERY = os.getenv("GMAIL_SYNC_QUERY", "newer_than:30d")
GMAIL_SYNC_MAX_RESULTS = int(os.getenv("GMAIL_SYNC_MAX_RESULTS", "100"))
GMAIL_SYNC_OUTPUT_PATH = os.getenv(
    "GMAIL_SYNC_OUTPUT_PATH",
    str(BASE_DIR / "data" / "real_emails" / "gmail_emails.json"),
)
GMAIL_SYNC_STATE_PATH = os.getenv(
    "GMAIL_SYNC_STATE_PATH",
    str(BASE_DIR / "data" / "mail_sync" / "gmail_sync_state.json"),
)

# Agent trace events. Disabled by default to keep local unit tests and demos
# quiet; enable when running evals or production-like debugging.
ENABLE_AGENT_TRACE = os.getenv("ENABLE_AGENT_TRACE", "false").lower() == "true"
AGENT_TRACE_LOG_PATH = os.getenv(
    "AGENT_TRACE_LOG_PATH",
    str(BASE_DIR / "data" / "traces" / "agent_traces.jsonl"),
)

# Explicit boundaries for the single-owner deployment.
FILTER_METADATA_SCAN_LIMIT = int(os.getenv('FILTER_METADATA_SCAN_LIMIT', '100000'))
FILTER_METADATA_PAGE_SIZE = int(os.getenv('FILTER_METADATA_PAGE_SIZE', '500'))
FILTER_VECTOR_BATCH_SIZE = int(os.getenv('FILTER_VECTOR_BATCH_SIZE', '100'))
GMAIL_SYNC_PAGE_SIZE = int(os.getenv('GMAIL_SYNC_PAGE_SIZE', '100'))
GMAIL_SYNC_MAX_PAGES_PER_RUN = int(os.getenv('GMAIL_SYNC_MAX_PAGES_PER_RUN', '1000'))
GMAIL_SYNC_CURSOR_RESET_LIMIT = int(os.getenv('GMAIL_SYNC_CURSOR_RESET_LIMIT', '1'))
RERANKER_COOLDOWN_SECONDS = float(os.getenv('RERANKER_COOLDOWN_SECONDS', '30'))
SELF_RAG_GRADE_CHAR_LIMIT = int(os.getenv('SELF_RAG_GRADE_CHAR_LIMIT', '1200'))
MAIL_APPROVAL_TIMEOUT_SECONDS = float(os.getenv('MAIL_APPROVAL_TIMEOUT_SECONDS', '30'))
FILTER_LEXICAL_MAX_CHUNKS = int(os.getenv('FILTER_LEXICAL_MAX_CHUNKS', '10000'))
FILTER_LEXICAL_CHAR_LIMIT = int(os.getenv('FILTER_LEXICAL_CHAR_LIMIT', '10000000'))
SESSION_STORE_PATH = os.getenv('SESSION_STORE_PATH', str(BASE_DIR / 'data' / 'sessions' / 'sessions.sqlite3'))
SESSION_CONTEXT_TURNS = int(os.getenv('SESSION_CONTEXT_TURNS', '30'))
MODEL_CONTEXT_TOKENS = int(os.getenv('MODEL_CONTEXT_TOKENS', '32000'))
MODEL_OUTPUT_RESERVE_TOKENS = int(os.getenv('MODEL_OUTPUT_RESERVE_TOKENS', '4000'))
JOB_STORE_PATH = os.getenv('JOB_STORE_PATH', str(BASE_DIR / 'data' / 'jobs' / 'jobs.sqlite3'))
MAX_ACTIVE_REQUESTS = int(os.getenv('MAX_ACTIVE_REQUESTS', '4'))
MAX_BACKGROUND_JOBS = int(os.getenv('MAX_BACKGROUND_JOBS', '2'))
WARMUP_ON_START = os.getenv('WARMUP_ON_START', 'false').lower() == 'true'
SESSION_CACHE_BYTES = int(os.getenv('SESSION_CACHE_BYTES', '16777216'))
MODEL_RUN_TOKEN_LIMIT = int(os.getenv('MODEL_RUN_TOKEN_LIMIT', '0'))
MODEL_RUN_COST_LIMIT = float(os.getenv('MODEL_RUN_COST_LIMIT', '0'))
MODEL_INPUT_COST_PER_MILLION = os.getenv('MODEL_INPUT_COST_PER_MILLION') or None
MODEL_OUTPUT_COST_PER_MILLION = os.getenv('MODEL_OUTPUT_COST_PER_MILLION') or None
EMBEDDING_MODEL_REVISION = os.getenv('EMBEDDING_MODEL_REVISION') or None
EMBEDDING_DIMENSION = int(os.environ['EMBEDDING_DIMENSION']) if os.getenv('EMBEDDING_DIMENSION') else None
BM25_MAX_CHUNKS = int(os.getenv('BM25_MAX_CHUNKS', '100000'))
BM25_CHAR_LIMIT = int(os.getenv('BM25_CHAR_LIMIT', '10000000'))
STATS_METADATA_SCAN_LIMIT = int(os.getenv('STATS_METADATA_SCAN_LIMIT', '100000'))
MODEL_TOKEN_PRICES = json.loads(os.getenv('MODEL_TOKEN_PRICES', '{}'))
if not isinstance(MODEL_TOKEN_PRICES,dict):
    raise ValueError('MODEL_TOKEN_PRICES must be a JSON object keyed by model identifier')
MODEL_STREAM_INCLUDE_USAGE = os.getenv('MODEL_STREAM_INCLUDE_USAGE', 'true').lower() == 'true'
LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', '5242880'))
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '3'))
MIME_PART_LIMIT = int(os.getenv('MIME_PART_LIMIT', '1000'))
MIME_DEPTH_LIMIT = int(os.getenv('MIME_DEPTH_LIMIT', '32'))
MIME_ENCODED_BYTE_LIMIT = int(os.getenv('MIME_ENCODED_BYTE_LIMIT', '16000000'))
MIME_DECODED_BYTE_LIMIT = int(os.getenv('MIME_DECODED_BYTE_LIMIT', '8000000'))
MIME_HEADER_CHAR_LIMIT = int(os.getenv('MIME_HEADER_CHAR_LIMIT', '64000'))
MIME_OUTPUT_CHAR_LIMIT = int(os.getenv('MIME_OUTPUT_CHAR_LIMIT', '8000000'))
MIME_PARSE_SECONDS = float(os.getenv('MIME_PARSE_SECONDS', '15'))
GMAIL_READ_TIMEOUT = float(os.getenv('GMAIL_READ_TIMEOUT', '30'))
RETRIEVAL_TIMEZONE = os.getenv('RETRIEVAL_TIMEZONE', '')
MAX_EMAIL_RECORD_BYTES = int(os.getenv('MAX_EMAIL_RECORD_BYTES', '16000000'))
MAX_EMAIL_JSON_DEPTH = int(os.getenv('MAX_EMAIL_JSON_DEPTH', '64'))
MAX_INDEX_INPUT_BYTES = int(os.getenv('MAX_INDEX_INPUT_BYTES', '2000000000'))
MAX_INDEX_INPUT_EMAILS = int(os.getenv('MAX_INDEX_INPUT_EMAILS', '100000'))
MAX_EMAIL_CHUNKS = int(os.getenv('MAX_EMAIL_CHUNKS', '10000'))
MAX_INDEX_INPUT_CHUNKS = int(os.getenv('MAX_INDEX_INPUT_CHUNKS', '250000'))

# Context rollout. Semantic generation remains an explicit experiment until
# the selected live model passes the fixed context quality cases.
ENABLE_CONTEXT_OPTIMIZATION = os.getenv('ENABLE_CONTEXT_OPTIMIZATION', 'true').lower() == 'true'
ENABLE_CONTEXT_SUMMARY = os.getenv('ENABLE_CONTEXT_SUMMARY', 'false').lower() == 'true'
MODEL_CONTEXT_SAFETY_TOKENS = int(os.getenv('MODEL_CONTEXT_SAFETY_TOKENS', '128'))
MODEL_REVISION = os.getenv('MODEL_REVISION') or None
MODEL_CONTEXT_PROFILES = json.loads(os.getenv('MODEL_CONTEXT_PROFILES', '{}'))
if not isinstance(MODEL_CONTEXT_PROFILES, dict):
    raise ValueError('MODEL_CONTEXT_PROFILES must be a JSON object keyed by model identifier')
CONTEXT_PURPOSE_WEIGHTS = json.loads(os.getenv('CONTEXT_PURPOSE_WEIGHTS', '{}'))
if not isinstance(CONTEXT_PURPOSE_WEIGHTS, dict):
    raise ValueError('CONTEXT_PURPOSE_WEIGHTS must be a JSON object keyed by call stage')
ENABLE_CONTEXT_CANDIDATES = os.getenv('ENABLE_CONTEXT_CANDIDATES', 'false').lower() == 'true'
CONTEXT_CANDIDATE_MAX_OUTPUT_TOKENS = int(os.getenv('CONTEXT_CANDIDATE_MAX_OUTPUT_TOKENS', '1000'))
CONTEXT_HISTORY_LIMIT = int(os.getenv('CONTEXT_HISTORY_LIMIT', '12'))
CONTEXT_MATERIAL_CHAR_LIMIT = int(os.getenv('CONTEXT_MATERIAL_CHAR_LIMIT', '4000'))
CONTEXT_MATERIAL_TOKEN_LIMIT = int(os.getenv('CONTEXT_MATERIAL_TOKEN_LIMIT', '2000'))
CONTEXT_SUMMARY_MIN_TURNS = int(os.getenv('CONTEXT_SUMMARY_MIN_TURNS', '20'))
CONTEXT_SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv('CONTEXT_SUMMARY_MAX_OUTPUT_TOKENS', '1500'))
CONTEXT_SUMMARY_MAX_ATTEMPTS = int(os.getenv('CONTEXT_SUMMARY_MAX_ATTEMPTS', '2'))
TOOL_RESULT_STORE_PATH = os.getenv('TOOL_RESULT_STORE_PATH', str(BASE_DIR / 'data' / 'sessions' / 'tool_results.sqlite3'))
CONTEXT_TOOL_RESULTS_ENABLED = os.getenv('CONTEXT_TOOL_RESULTS_ENABLED', 'true').lower() == 'true'
CONTEXT_TOOL_COMPACTION_ENABLED = os.getenv('CONTEXT_TOOL_COMPACTION_ENABLED', 'true').lower() == 'true'
