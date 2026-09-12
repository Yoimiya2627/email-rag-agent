from __future__ import annotations
from pydantic import AfterValidator, BaseModel, Field
from typing import Annotated, Optional, List, Dict, Any, Literal
from enum import Enum


class Email(BaseModel):
    id: str
    subject: str
    sender: str
    recipients: List[str]
    date: str
    body: str
    # Historical JSON and Gmail-normalized bodies are plain text. HTML input
    # must declare its format; literal angle brackets are not format evidence.
    body_format: Literal["plain", "html"] = "plain"
    table_rows: List[Dict[str, Any]] = Field(default_factory=list)
    labels: List[str] = []
    thread_id: Optional[str] = None
    sender_name: str = ""
    cc: List[str] = Field(default_factory=list)
    message_id: str = ""
    in_reply_to: str = ""
    references: List[str] = Field(default_factory=list)
    label_names: List[str] = Field(default_factory=list)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    source: Dict[str, Any] = Field(default_factory=dict)
    decode_quality: Dict[str, Any] = Field(default_factory=dict)


class EmailChunk(BaseModel):
    chunk_id: str
    email_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = {}


class SearchResult(BaseModel):
    chunk_id: str
    email_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}


class IntentType(str, Enum):
    RETRIEVE = "retrieve"
    SUMMARIZE = "summarize"
    WRITE_REPLY = "write_reply"
    ANALYZE = "analyze"
    GENERAL = "general"


def _nonblank_query(value: str) -> str:
    if not value.strip():
        raise ValueError("query must contain non-whitespace text")
    return value


QueryText = Annotated[str, Field(min_length=1, max_length=20000), AfterValidator(_nonblank_query)]


class AgentRequest(BaseModel):
    query: QueryText
    user_email: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    operation_key: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r'^[A-Za-z0-9._:-]+$')


class AgentResponse(BaseModel):
    intent: Optional[IntentType] = None
    answer: str
    sources: List[SearchResult] = []
    metadata: Optional[Dict[str, Any]] = None


class IndexRequest(BaseModel):
    data_path: Optional[str] = None


class IndexResponse(BaseModel):
    success: bool
    message: str
    count: int = 0
    index_metrics: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    query: QueryText
    top_k: int = Field(default=5, ge=1, le=100)


class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchResult] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)
