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
from core.model_outcomes import display_text, outcome_metadata

logger = logging.getLogger(__name__)


class RetrieverAgent:
    def prepare_contexts(self, query: str, *, filters=None, history=None):
        """Public: run the full retrieval pipeline and return the reranked
        SearchResult list.

        Kept as a public method so callers (e.g. the streaming endpoint) can
        plug the contexts into a streaming generator without going through
        run().
        """
        return retrieve(query, filters=filters, history=history)

    def run(self, request: AgentRequest, memory=None) -> AgentResponse:
        history = memory.to_messages() if memory else None
        reranked = retrieve(request.query, history=history)
        answer = generate_answer(request.query, reranked, history=history)
        return AgentResponse(answer=display_text(answer), sources=reranked,
                             metadata=outcome_metadata(answer))
