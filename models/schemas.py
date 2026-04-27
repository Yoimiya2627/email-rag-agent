from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class Email(BaseModel):
    id: str
    subject: str
    sender: str
    recipients: List[str]
    date: str
    body: str
    labels: List[str] = []
    thread_id: Optional[str] = None


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


class AgentRequest(BaseModel):
    query: str
    user_email: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


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


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchResult] = []
