"""
Sliding-window conversation memory for multi-turn chat.
Keeps the last max_turns (user, assistant) pairs in context.
"""
import json
import logging
import threading
import copy
import uuid
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List
from core.context_budget import measure_context, within_budget, model_profile

logger = logging.getLogger(__name__)


def _object_bytes(value, seen=None):
    """Estimate retained Python objects without serializing/copying private text."""
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_object_bytes(k, seen)+_object_bytes(v, seen) for k,v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(_object_bytes(item, seen) for item in value)
    elif isinstance(value, Turn):
        size += _object_bytes(value.__dict__, seen)
    return size


class HistoryMessages(list):
    """Messages with backend turn identities kept out of provider payloads."""
    def __init__(self, rows=(), turn_ids=()):
        super().__init__(rows)
        self.turn_ids = list(turn_ids)


def conversation_pairs(history) -> list[dict]:
    """Copy complete user/assistant turns; never import system/tool authority."""
    output, user = HistoryMessages(), None
    seen = set()
    ids = getattr(history, "turn_ids", [])
    current_id = None
    for index, row in enumerate(history or []):
        if not isinstance(row, dict) or not isinstance(row.get('content'), str):
            user = None
            continue
        role = row.get('role')
        if role == 'user':
            user = {'role': 'user', 'content': row['content']}
            current_id = row.get('turn_id') or (ids[index // 2] if index // 2 < len(ids) else None)
        elif role == 'assistant' and user is not None:
            if not current_id or current_id not in seen:
                output.extend([user, {'role': 'assistant', 'content': row['content']}])
                output.turn_ids.append(current_id)
                if current_id:
                    seen.add(current_id)
            user = None
        else:
            user = None
    return output


def fit_messages_to_budget(messages: list[dict], history_message_count: int = 0, *,
                           schemas=None, char_limit=None, token_limit=None,
                           output_reserve=None, token_counter=None, model=None, model_revision=None,
                           stage=None, max_output_tokens=None, safety_margin=None) -> tuple[list[dict], int]:
    """Drop oldest complete prior turns, preserving the current run verbatim.

    Historical turns occupy the prefix after the system message. Current user
    input and all assistant/tool messages from this run are never cut or split.
    Count JSON exactly as RunContext.check_context does, including tool schemas.
    """
    from agents.runtime import ContextBudgetExceeded, current_run
    import config.settings as cfg
    if history_message_count < 0 or history_message_count % 2 or history_message_count > max(0, len(messages)-2):
        raise ValueError('history_message_count must identify complete prior turns')
    run = current_run()
    if char_limit is None:
        char_limit = run.context_char_limit if run else cfg.AGENT_CONTEXT_CHAR_LIMIT
    if token_limit is None:
        token_limit = getattr(run, "context_token_limit", None)
        if token_limit is None:
            token_limit = int(getattr(cfg, "MODEL_CONTEXT_TOKENS", 0))
    if output_reserve is None:
        output_reserve = getattr(run, "output_token_reserve", None)
        if output_reserve is None:
            output_reserve = int(getattr(cfg, "MODEL_OUTPUT_RESERVE_TOKENS", 0))
    from core.context_budget import model_profile
    profile = model_profile(model, stage)
    model_revision = profile['revision'] or model_revision
    if profile['context_tokens']:
        token_limit = min(token_limit, profile['context_tokens']) if token_limit else profile['context_tokens']
    if max_output_tokens is not None:
        safety_margin = int(getattr(cfg, "MODEL_CONTEXT_SAFETY_TOKENS", 128)) if safety_margin is None else safety_margin
        if type(max_output_tokens) is not int or max_output_tokens < 0 or type(safety_margin) is not int or safety_margin < 0:
            raise ValueError("invalid output tokens or safety margin")
        output_reserve = max_output_tokens + safety_margin
    token_counter = token_counter or getattr(run, "token_counter", None)
    kept = list(messages)
    original_count = history_message_count
    while True:
        usage = measure_context(kept, schemas, token_counter=token_counter, model=model, model_revision=model_revision, stage=stage)
        if within_budget(usage, char_limit=char_limit, token_limit=token_limit, output_reserve=output_reserve):
            break
        if history_message_count == 0:
            if run is not None:
                run.context_metrics.update(usage, budget_failure='required_input_over_budget',
                    token_capacity=token_limit, output_token_reserve=output_reserve)
            raise ContextBudgetExceeded('current input, evidence or tool results exceed the context budget')
        del kept[1:3]
        history_message_count -= 2
    if run is not None:
        run.context_metrics.pop("budget_failure", None)
        run.context_metrics = {**run.context_metrics, **usage, "dropped_history_turns": (original_count-history_message_count)//2,
                               "token_capacity": token_limit, "output_token_reserve": output_reserve}
    if original_count != history_message_count:
        logger.info('Dropped oldest conversation turns count=%s', (original_count-history_message_count)//2)
    return kept, history_message_count


def build_model_messages(system: str, current: str, history=None, *, schemas=None,
                         stage=None, model=None, model_revision=None, max_output_tokens=None,
                         safety_margin=None, original_request=None) -> list[dict]:
    from agents.runtime import current_run
    from core.session_context import apply_session_context, without_recent_excerpts, context_for_stage
    run = current_run()
    from core.context_contracts import ContextRequest, ContextSelection, TrustedScope
    request = ContextRequest(TrustedScope(run.owner_id, run.session_id, run.run_id) if run else None,
                             stage, current if original_request is None else original_request)
    if not isinstance(current, str):
        raise ValueError('current model input must be text')
    task_context = getattr(run, "task_context", None)
    if task_context:
        task_context = context_for_stage(task_context, stage, model, model_revision)
        task_context = without_recent_excerpts(task_context, [], stage=stage)
    prior = conversation_pairs(history)
    count = len(prior)
    from agents.runtime import ContextBudgetExceeded
    while True:
        retained_ids = prior.turn_ids[len(prior.turn_ids)-count//2:] if count else []
        deduped = without_recent_excerpts(task_context, retained_ids) if task_context else None
        final_system, final_current = apply_session_context(system, current, deduped)
        tail = prior[len(prior)-count:] if count else []
        messages = [{'role':'system','content':final_system}, *tail, {'role':'user','content':final_current}]
        try:
            # Recalculate duplicates for each candidate window: if a turn is
            # evicted, its retrieved excerpt becomes eligible again immediately.
            fitted, _ = fit_messages_to_budget(messages, 0, schemas=schemas, stage=stage,
                model=model, model_revision=model_revision, max_output_tokens=max_output_tokens, safety_margin=safety_margin)
            break
        except ContextBudgetExceeded:
            if not count:
                raise
            count -= 2
    if run is not None:
        run.context_metrics['dropped_history_turns'] = (len(prior)-count)//2
    if run is not None and task_context:
        final_material = deduped if count else task_context
        from collections import Counter
        final_usage = measure_context(fitted, schemas, token_counter=getattr(run, 'token_counter', None),
            model=model, model_revision=model_profile(model, stage)['revision'] or model_revision, stage=stage)
        run.context_metrics.update(final_usage)
        run.context_metrics.update(material_counts=dict(Counter(row['kind'] for row in final_material.get('material_manifest', []))),
            omission_counts=dict(Counter(row['reason'] for row in final_material.get('omissions', []))), material_manifest=final_material.get('material_manifest', []),
            omissions=final_material.get('omissions', []), selection_policy='purpose_v1', stage=stage)
    if run is not None:
        from core.context_contracts import ContextMaterial, TrustedScope
        scope = TrustedScope(run.owner_id, run.session_id, run.run_id)
        method = run.context_metrics.get('token_estimation_method', 'utf8_bytes_upper_bound')
        core_manifest = [ContextMaterial('current_request', 'current_user_request', scope,
            run.run_id, 'request_v1', current if original_request is None else original_request,
            coverage='complete', required=True).manifest(position='current_user', method=method)]
        ids = prior.turn_ids[len(prior.turn_ids)-count//2:] if count else []
        for index, row in enumerate(fitted[1:1+count]):
            turn_id = ids[index//2] or 'legacy_window:'+str(index//2)
            kind = 'user_statement' if row['role'] == 'user' else 'historical_assistant_claim'
            core_manifest.append(ContextMaterial('recent:'+turn_id+':'+row['role'], kind, scope,
                turn_id+':'+row['role'], 'turn_v1', row['content'], coverage='complete').manifest(
                    position='recent_window', method=method))
        from collections import Counter
        manifests = [*core_manifest, *(run.context_metrics.get('material_manifest', []) if task_context else [])]
        selection = ContextSelection(request, fitted, manifests, run.context_metrics.get('omissions', []) if task_context else [])
        run.context_metrics.update(selection.metrics())
        return selection.messages
    return ContextSelection(request, fitted, [], []).messages


@dataclass
class Turn:
    role: str   # "user" or "assistant"
    content: str
    turn_id: str | None = None


class ConversationMemory:
    """Thread-safe sliding window. FastAPI runs requests on a thread pool, and
    the same session_id can be touched concurrently (e.g. background streaming
    persisting `assistant` while a new `user` request reads history). Without
    a lock the underlying list can be torn during slice-rebind."""

    def __init__(self, max_turns: int | None = 5):
        if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
            raise ValueError("history window must be a positive integer or None")
        self.max_turns = max_turns
        self._turns: List[Turn] = []
        self._lock = threading.Lock()
        self._transcript = []
        self._pending_turns = []
        self._legacy_user = None
        self.cache_byte_limit = None

    def _trim(self):
        # None selects a bounded cache; explicit legacy windows stay compatible.
        max_messages = (self.max_turns or 1000) * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def _record(self, query, answer, metadata, include_in_context, evidence_refs, turn_id=None):
        metadata = copy.deepcopy(metadata or {})
        from core.model_outcomes import ModelText, outcome_metadata
        if isinstance(answer, ModelText):
            outcome = outcome_metadata(answer)
            if outcome["status"] != "success":
                metadata.update(outcome)
        status = metadata.get("status", "success")
        include = (bool(include_in_context) and status == "success" and bool(answer.strip())
                   and metadata.get("completion_status", "complete") == "complete")
        record = {"turn_id": turn_id or str(uuid.uuid4()), "query": query, "answer": answer,
                  "metadata": metadata, "include_in_context": include,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "evidence_refs": copy.deepcopy(evidence_refs or [])}
        self._transcript.append(record)
        self._pending_turns.append(record)
        return record

    def append_turn(self, query: str, answer: str, metadata=None, *, include_in_context=True,
                    evidence_refs=None, turn_id=None):
        """Stage an atomic pair; SessionStore commits it on successful turn exit.

        Failures/partial answers remain in transcript, never in successful model
        history. Callers explicitly record an error inside the turn if desired;
        an exception escaping the turn rolls the entire staged change back.
        """
        if not isinstance(query, str) or not isinstance(answer, str):
            raise ValueError("turn query and answer must be strings")
        with self._lock:
            record = self._record(query, answer, metadata, include_in_context, evidence_refs, turn_id)
            if record["include_in_context"]:
                self._turns.extend([Turn("user", query, record["turn_id"]), Turn("assistant", answer, record["turn_id"])])
                self._trim()
            return record["turn_id"]

    def add(self, role: str, content: str) -> None:
        with self._lock:
            self._turns.append(Turn(role=role, content=content))
            self._trim()
            if role == "user":
                self._legacy_user = content
            elif role == "assistant" and self._legacy_user is not None:
                self._record(self._legacy_user, content, {}, True, [])
                self._legacy_user = None
            else:
                self._legacy_user = None

    def snapshot(self):
        with self._lock:
            return copy.deepcopy((self._turns, self._transcript, self._pending_turns, self._legacy_user))

    def restore(self, snapshot):
        with self._lock:
            self._turns, self._transcript, self._pending_turns, self._legacy_user = copy.deepcopy(snapshot)

    def pending_turns(self):
        with self._lock:
            return copy.deepcopy(self._pending_turns)

    def committed(self, *, persistent=False, base_snapshot=None):
        with self._lock:
            if persistent:
                # Build from the pre-turn successful cache plus committed pairs,
                # so legacy add(user) without an assistant cannot leak a half turn.
                self._turns = copy.deepcopy(base_snapshot[0]) if base_snapshot else []
                self._turns.extend(Turn(role, row[name], row.get("turn_id")) for row in self._pending_turns
                                   if row["include_in_context"]
                                   for role, name in (("user", "query"), ("assistant", "answer")))
                self._trim()
                self._legacy_user = None
                self._transcript = []
            self._pending_turns = []

    def load_context(self, records):
        with self._lock:
            self._turns = [Turn(role, row[name], row.get("turn_id")) for row in records if row.get("include_in_context")
                           for role, name in (("user", "query"), ("assistant", "answer"))]
            self._trim()
            self._transcript, self._pending_turns, self._legacy_user = [], [], None

    def transcript(self):
        with self._lock:
            return [{"seq": index, **copy.deepcopy(row)} for index, row in enumerate(self._transcript, 1)]

    def to_messages(self) -> List[dict]:
        with self._lock:
            if self.cache_byte_limit is not None and _object_bytes(self._turns) > self.cache_byte_limit:
                from agents.runtime import ContextBudgetExceeded
                raise ContextBudgetExceeded("Session history exceeds its cache budget; reduce the history window")
            return HistoryMessages([{"role": t.role, "content": t.content} for t in self._turns],
                                   [t.turn_id for t in self._turns if t.role == "user"])

    def cache_size(self) -> int:
        with self._lock:
            return _object_bytes((self._turns, self._transcript, self._pending_turns, self._legacy_user)) + 256

    def clear(self) -> None:
        with self._lock:
            self._turns = []
            self._transcript, self._pending_turns, self._legacy_user = [], [], None

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)
