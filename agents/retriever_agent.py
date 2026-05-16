"""
RetrieverAgent: thin wrapper over the shared retrieval pipeline
(core.pipeline.retrieve) plus grounded answer generation.

The rewrite / filter-extraction / hybrid-search / post-filter / rerank logic
now lives in core/pipeline.py so the agent, the streaming endpoint and the
evaluation scripts all exercise the exact same pipeline.
"""
import logging

from models.schemas import AgentRequest, AgentResponse
from core.pipeline import retrieve
from core.generator import generate_answer

logger = logging.getLogger(__name__)


class RetrieverAgent:
    def prepare_contexts(self, query: str):
        """Public: run the full retrieval pipeline and return the reranked
        SearchResult list.

        Kept as a public method so callers (e.g. the streaming endpoint) can
        plug the contexts into a streaming generator without going through
        run().
        """
        return retrieve(query)

    def run(self, request: AgentRequest, memory=None) -> AgentResponse:
        reranked = retrieve(request.query)
        history = memory.to_messages() if memory else None
        answer = generate_answer(request.query, reranked, history=history)
        return AgentResponse(answer=answer, sources=reranked)
