import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from models.schemas import AgentRequest, AgentResponse, IndexRequest, IndexResponse, QueryRequest, QueryResponse
from core.loader import load_emails
from core.cleaner import clean_email
from core.chunker import chunk_email
from core.embedder import index_chunks, clear_collection, get_collection_stats, verify_collection_readiness
from core.storage_paths import validate_chroma_path
from agents.coordinator import route
from agents.general_agent import direct_general_response
from agents.approvals import ApprovalStore
from agents.mail_providers import MailProviderError, create_mail_provider_from_settings
from agents.mcp_adapter import MCPAuditLogger
from api.security import Identity, require_identity
from api.sessions import SessionStore, SessionBusyError
from agents.runtime import RunContext, RunCancelled, ContextBudgetExceeded, use_run_context, remaining_timeout
from core.model_outcomes import ModelOutputError, display_text, outcome_metadata
from api.contracts import can_commit_answer, finalize_response
from agents.execution_scope import ExecutionScope, use_execution_scope
from typing import Literal
import config.settings as cfg
from core.jobs import AdmissionGate, CapacityExceeded, JobStore, JobManager
from api.readiness import Readiness
from core.model_clients import ModelBudgetExceeded,model_metrics_snapshot

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)
sessions = SessionStore(cfg.MAX_SESSIONS, cfg.SESSION_TTL_SECONDS,
                        path=cfg.SESSION_STORE_PATH, max_history_turns=cfg.SESSION_CONTEXT_TURNS,
                        max_cached_bytes=cfg.SESSION_CACHE_BYTES)
admission = AdmissionGate(cfg.MAX_ACTIVE_REQUESTS)
_jobs = None
_jobs_lock = threading.Lock()
readiness = Readiness()


@asynccontextmanager
async def lifespan(app):
    validate_chroma_path(cfg.CHROMA_PERSIST_DIR)
    logger.info('Email RAG API starting up')
    if cfg.WARMUP_ON_START:
        readiness.start(_warm_components)
    yield
    logger.info('Email RAG API shut down')
    if _jobs is not None:
        _jobs.stop()
        await asyncio.to_thread(_jobs.wait_idle,5)
    if admission.active == 0:
        from core.model_clients import close_model_clients
        close_model_clients()


app = FastAPI(title='Email RAG API', version='1.1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=cfg.API_CORS_ORIGINS,
                   allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Authorization', 'Content-Type'])


def _safe_error(exc):
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, PermissionError):
        return HTTPException(403, 'Resource access denied')
    if isinstance(exc, KeyError):
        return HTTPException(404, 'Resource not found')
    if isinstance(exc, SessionBusyError):
        return HTTPException(409, str(exc))
    if isinstance(exc, CapacityExceeded):
        return HTTPException(429, 'All execution slots are busy; wait for an active task to finish', headers={'Retry-After':'2'})
    if isinstance(exc, RunCancelled):
        return HTTPException(409, 'Task cancelled; inspect completed steps before continuing')
    if isinstance(exc, ModelOutputError):
        return HTTPException(502, str(exc))
    if isinstance(exc, ContextBudgetExceeded):
        return HTTPException(413, 'Current input or evidence exceeds the context budget; reduce the request size')
    if isinstance(exc, ModelBudgetExceeded):
        return HTTPException(413, 'Model token or cost budget exhausted; inspect completed work before continuing')
    if isinstance(exc, ValueError):
        return HTTPException(409, 'Invalid input or operation conflicts with current state')
    if isinstance(exc, TimeoutError):
        return HTTPException(504, 'Operation timed out; check action status before retrying')
    if isinstance(exc, MailProviderError):
        return HTTPException(502, 'Mail provider failed; inspect approval state before retrying')
    logger.error('API operation failed error_type=%s', type(exc).__name__)
    return HTTPException(500, 'Operation failed; inspect its run or approval status')


@app.get('/health')
def health():
    """Liveness only; does not certify model, mailbox or index readiness."""
    return {'status': 'ok'}


def _warm_components():
    # Explicit warmup may load local/downloaded embedding weights; it does not
    # send prompts to the model provider or open an OAuth authorization flow.
    from core.embedder import embed_texts
    from core.generator import _get_client
    with admission.slot():
        verify_collection_readiness()
        embed_texts(['warmup'])
        _get_client()
    return {'embedding':'initialized','model_client':'initialized','index':'verified'}


@app.get('/ready')
def ready(identity: Identity = Depends(require_identity)):
    state = readiness.snapshot()
    try:
        index = verify_collection_readiness()
        state['index'] = index
        usable = bool(index.get('chunk_count',0)) and not index.get('requires_rebuild',False)
    except Exception as exc:
        state['index'] = {'status':'unavailable','error_code':type(exc).__name__}
        usable = False
    state['ready'] = state['status']=='ready' and usable
    state['active_requests'] = admission.active
    return JSONResponse(state,status_code=200 if state['ready'] else 503)


@app.post('/warmup',status_code=202)
def warmup(identity: Identity = Depends(require_identity)):
    try:
        readiness.start(_warm_components)
        return readiness.snapshot()
    except Exception as exc:
        raise _safe_error(exc) from None


def _session_material(owner, sid, query, *, memory=None, context=None):
    from core.session_context import assemble_session_context
    from core.context_contracts import TrustedScope
    started = time.perf_counter()
    state = sessions.context_state(owner, sid) if sessions.repository else {}
    # The user request is kept verbatim in the model call. Derived history is
    # supplied only from this backend-bound session; request.context cannot
    # select an owner, history repository or remembered execution permission.
    if not cfg.ENABLE_CONTEXT_OPTIMIZATION:
        import re
        terms = list(dict.fromkeys(re.findall(r'[A-Za-z0-9_-]{3,}|[\u4e00-\u9fff]{2,8}',query)))[:3]
        matches = {}
        for term in terms:
            for row in sessions.search_history(owner,sid,term,limit=5):
                matches[row['turn_id']] = row
        return assemble_session_context(state.get('facts', []), list(matches.values()))
    matches = sessions.search_history(owner, sid, query[:500], limit=cfg.CONTEXT_HISTORY_LIMIT,
                                      task_id=state.get('task_id') if sessions.repository else None)
    recent_ids = getattr(memory.to_messages(), 'turn_ids', []) if memory is not None else []
    material = assemble_session_context(state.get('facts', []), matches,
        scope=TrustedScope(owner, sid, context.run_id if context else None),
        task_state=state.get('task_state'), summary=state.get('summary'),
        recent_turn_ids=recent_ids, user_events=state.get('user_events', []), current_request=query,
        char_limit=cfg.CONTEXT_MATERIAL_CHAR_LIMIT, token_limit=cfg.CONTEXT_MATERIAL_TOKEN_LIMIT)
    material.update(session_revision=state.get('revision', 0), deletion_epoch=state.get('deletion_epoch', 0),
                    task_id=state.get('task_id'), history_search_ms=round((time.perf_counter()-started)*1000, 3))
    if state.get('required_omissions'):
        material['external_required_omissions']=list(state['required_omissions'])
        material['required_omissions']=[*material.get('required_omissions',[]),*state['required_omissions']]
        material['omissions']=[*material.get('omissions',[]),*state['required_omissions']]
        material['omitted_count']=len(material['omissions'])
    if sessions.repository:
        material['history_retrieval']=dict(getattr(sessions.repository,'last_history_diagnostics',{}))
    return material


def _summary_generate(*, payload, budget=None):
    """Optional semantic call shares the active request deadline and usage cap."""
    from core.model_clients import create_completion, get_model_client
    from core.model_outcomes import text_from_choice
    from core.session_summary import SUMMARY_PROMPT
    response = create_completion(get_model_client(), stage='session_summary',
        model=cfg.DEEPSEEK_MODEL, temperature=0,
        max_tokens=cfg.CONTEXT_SUMMARY_MAX_OUTPUT_TOKENS, timeout=cfg.LLM_TIMEOUT,
        messages=[{'role':'system', 'content':SUMMARY_PROMPT},
                  {'role':'user', 'content':json.dumps(payload, ensure_ascii=False)}])
    return _complete_derived_text(response)


def _prepare_session_context(owner, sid, query, memory, context, scope):
    if sessions.repository:
        context.session_repository = scope.session_repository = sessions.repository
        state = sessions.context_state(owner, sid)
        context.context_epoch = scope.context_epoch = state['deletion_epoch']
        scope.session_id = sid
        if cfg.ENABLE_CONTEXT_OPTIMIZATION and not getattr(context,'resuming',False):
            sessions.record_current_request(owner, sid, text=query, request_id=context.run_id)
        if cfg.ENABLE_CONTEXT_SUMMARY and not getattr(context,'resuming',False):
            outcome = sessions.generate_summary(owner, sid, generate=_summary_generate, budget=context,
                min_turns=cfg.CONTEXT_SUMMARY_MIN_TURNS, max_attempts=cfg.CONTEXT_SUMMARY_MAX_ATTEMPTS,
                model_id=cfg.DEEPSEEK_MODEL,model_revision=cfg.MODEL_REVISION or 'unknown')
            context.context_metrics['summary'] = {key:outcome.get(key) for key in
                ('status','reason','reused','attempts') if key in outcome}
        if cfg.ENABLE_CONTEXT_CANDIDATES and not getattr(context,'resuming',False):
            outcome=sessions.extract_candidates(owner,sid,generate=_candidate_generate,budget=context,
                model_id=cfg.DEEPSEEK_MODEL,model_revision=cfg.MODEL_REVISION or 'unknown')
            context.context_metrics['memory_candidates']={key:outcome.get(key) for key in ('status','reason') if key in outcome}
        if cfg.CONTEXT_TOOL_RESULTS_ENABLED:
            from core.tool_results import ToolResultStore
            import sqlite3
            try:
                store = ToolResultStore(cfg.TOOL_RESULT_STORE_PATH)
                context.tool_result_store = scope.tool_result_store = store
            except (OSError, ValueError, sqlite3.Error) as exc:
                context.context_metrics['result_storage_degraded'] = type(exc).__name__
    context.task_context = _session_material(owner, sid, query, memory=memory, context=context)


def _candidate_generate(*, payload, budget=None):
    from core.model_clients import create_completion,get_model_client
    from core.model_outcomes import text_from_choice
    from core.session_candidates import CANDIDATE_PROMPT
    response=create_completion(get_model_client(),stage='context_candidates',model=cfg.DEEPSEEK_MODEL,
        temperature=0,max_tokens=cfg.CONTEXT_CANDIDATE_MAX_OUTPUT_TOKENS,timeout=cfg.LLM_TIMEOUT,
        messages=[{'role':'system','content':CANDIDATE_PROMPT},
                  {'role':'user','content':json.dumps(payload,ensure_ascii=False)}])
    return _complete_derived_text(response)


def _complete_derived_text(response):
    from core.model_outcomes import text_from_choice
    value=text_from_choice(response.choices[0] if response.choices else None)
    if outcome_metadata(value)['completion_status']!='complete':
        raise ModelOutputError(str(value),finish_reason=getattr(value,'finish_reason',None))
    return str(value)


def _bounded_endpoint(function):
    from functools import wraps
    @wraps(function)
    def bounded(*args,**kwargs):
        try:
            with admission.slot():
                return function(*args,**kwargs)
        except Exception as exc:
            raise _safe_error(exc) from None
    return bounded


@app.post('/index', response_model=IndexResponse)
@_bounded_endpoint
def index_emails(request: Optional[IndexRequest] = Body(None), identity: Identity = Depends(require_identity)):
    try:
        from core.ingestion import prepare_email_chunks
        from core.index_metrics import collect_index_metrics
        with collect_index_metrics() as report:
            emails, chunks = prepare_email_chunks(request.data_path if request else None)
            try:
                email_count = len(emails)
                count = index_chunks(chunks)
            finally:
                close = getattr(emails,'close',None)
                if close:
                    close()
        if report.outcome != 'unchanged':
            from core.retriever import invalidate_bm25_cache
            invalidate_bm25_cache()
        message = ('Index unchanged' if report.outcome == 'unchanged'
                   else f'Indexed {email_count} emails into {count} chunks')
        return IndexResponse(success=True, message=message, count=count, index_metrics=report.to_dict())
    except FileNotFoundError:
        raise HTTPException(404, 'Email data file not found')
    except Exception as exc:
        raise _safe_error(exc) from None


@app.post('/index/clear')
@_bounded_endpoint
def clear_index(identity: Identity = Depends(require_identity)):
    try:
        from core.index_metrics import collect_index_metrics
        with collect_index_metrics() as report:
            clear_collection()
        if report.outcome != 'unchanged':
            from core.retriever import invalidate_bm25_cache
            invalidate_bm25_cache()
        return {'success':True,'message':'Index already empty' if report.outcome=='unchanged' else 'Index cleared',
                'index_metrics':report.to_dict()}
    except Exception as exc:
        raise _safe_error(exc) from None


@app.get('/index/status')
def index_status(identity: Identity = Depends(require_identity)):
    try:
        return get_collection_stats()
    except Exception as exc:
        raise _safe_error(exc) from None


def _recorded_evidence(metadata, context, answer):
    """Keep actual prompt ranges distinct from retrieved candidate documents."""
    refs=[]
    for ref in [*(metadata.get('model_visible_evidence') or []),*context.generation_evidence_refs]:
        if ref not in refs:
            refs.append(ref)
    metadata['model_visible_evidence']=refs
    metadata.setdefault('sources_kind','retrieved_candidates')
    metadata.setdefault('cited_evidence',[ref for ref in refs
        if f"[{ref['email_id']}#{ref['chunk_id']}]" in answer])
    return refs


def _chat_turn(request, identity, runner, *, context=None, admitted=False):
    session_id = request.session_id or (str(uuid.uuid5(uuid.NAMESPACE_URL,identity.owner_id+':'+request.operation_key))
        if request.operation_key else str(uuid.uuid4()))
    request = request.model_copy(update={'session_id': session_id})
    try:
        context = context or RunContext(owner_id=identity.owner_id, session_id=session_id,
                             deadline=time.monotonic()+cfg.AGENT_RUN_TIMEOUT,
                             context_char_limit=cfg.AGENT_CONTEXT_CHAR_LIMIT)
        scope = ExecutionScope(owner_id=identity.owner_id, operation_key=request.operation_key)
        from contextlib import nullcontext
        failure = None
        with (nullcontext() if admitted else admission.slot()), use_execution_scope(scope), use_run_context(context), sessions.turn(identity.owner_id, session_id) as memory:
            try:
                remaining_timeout(cfg.LLM_TIMEOUT)
                direct = None if context.resuming else direct_general_response(request.query)
                if direct is None:
                    _prepare_session_context(identity.owner_id,session_id,request.query,memory,context,scope)
                    remaining_timeout(cfg.LLM_TIMEOUT)
                response = finalize_response(direct if direct is not None else runner(request, memory))
                if context.cancel_event is not None and context.cancel_event.is_set():
                    response.metadata.update(status='cancelled', completion_status='incomplete')
            except Exception as exc:
                failure = exc
                response = AgentResponse(answer=str(getattr(exc,'partial_text','')),
                    metadata=getattr(exc,'metadata',None) or {'status':'cancelled' if isinstance(exc,RunCancelled) else 'error','completion_status':'error'})
            response.metadata = {**(response.metadata or {}), 'session_id':session_id,'run_id':response.metadata.get('run_id',context.run_id)}
            response.metadata['session_context'] = {key:value for key,value in (context.task_context or {}).items() if key!='text'}
            response.metadata.setdefault('context_metrics',dict(context.context_metrics))
            response.metadata['model_usage'] = model_metrics_snapshot(context)
            refs = _recorded_evidence(response.metadata,context,response.answer)
            turn_id = memory.append_turn(request.query,response.answer,response.metadata,
                include_in_context=can_commit_answer(response.answer,response.metadata),evidence_refs=refs)
            response.metadata['turn_id'] = turn_id
        if failure is not None:
            raise failure
        return response
    except Exception as exc:
        raise _safe_error(exc) from None


@app.post('/chat', response_model=AgentResponse)
def chat(request: AgentRequest, identity: Identity = Depends(require_identity)):
    return _chat_turn(request, identity, lambda req, memory: route(req, memory=memory))


@app.post('/chat/agent', response_model=AgentResponse)
def chat_agent(request: AgentRequest, identity: Identity = Depends(require_identity)):
    from agents.agent_loop import run_agent_loop
    return _chat_turn(request, identity, lambda req, memory: run_agent_loop(
        req, memory=memory, owner_id=identity.owner_id, session_id=req.session_id))


@app.post('/chat/graph', response_model=AgentResponse)
def chat_graph(request: AgentRequest, identity: Identity = Depends(require_identity)):
    def invoke(req, memory):
        try:
            from agents.graph_workflow import run_graph
            return run_graph(req, memory=memory)
        except ImportError:
            raise HTTPException(503, 'Graph mode requires the optional graph dependencies') from None
    return _chat_turn(request, identity, invoke)


@app.post('/chat/stream')
async def chat_stream(request: AgentRequest, identity: Identity = Depends(require_identity)):
    """Bounded legacy RAG stream; a provider already running may finish after disconnect."""
    from core.generator import stream_generate
    from core.pipeline import retrieve
    from agents.coordinator import classify_intent
    from models.schemas import IntentType
    session_id = request.session_id or (str(uuid.uuid5(uuid.NAMESPACE_URL,identity.owner_id+':'+request.operation_key))
        if request.operation_key else str(uuid.uuid4()))
    request = request.model_copy(update={'session_id': session_id})
    stopped = threading.Event()
    events = queue.Queue(maxsize=128)
    deadline = time.monotonic() + cfg.AGENT_RUN_TIMEOUT

    def emit(event):
        while not stopped.is_set() and time.monotonic() < deadline:
            try:
                events.put(event, timeout=.1)
                return True
            except queue.Full:
                pass
        stopped.set()
        return False

    def producer():
        try:
            context = RunContext(owner_id=identity.owner_id, session_id=session_id, deadline=deadline,
                                 context_char_limit=cfg.AGENT_CONTEXT_CHAR_LIMIT,cancel_event=stopped)
            scope = ExecutionScope(owner_id=identity.owner_id, operation_key=request.operation_key)
            with admission.slot(), use_execution_scope(scope), use_run_context(context), sessions.turn(identity.owner_id, session_id) as memory:
                answer, sources, error = '', [], None
                metadata = {'status':'incomplete','completion_status':'incomplete'}
                try:
                    remaining_timeout(cfg.LLM_TIMEOUT)
                    direct = direct_general_response(request.query)
                    if direct is None:
                        _prepare_session_context(identity.owner_id,session_id,request.query,memory,context,scope)
                        history = memory.to_messages()
                        remaining_timeout(cfg.LLM_TIMEOUT)
                        intent = classify_intent(request.query, history=history)
                    else:
                        intent = direct.intent
                    if not emit({'intent': intent.value, 'session_id': session_id}):
                        return
                    if intent == IntentType.RETRIEVE:
                        contexts = retrieve(request.query, history=history)
                        sources = [item.model_dump() for item in contexts]
                        remaining_timeout(cfg.LLM_TIMEOUT)
                        stream = stream_generate(request.query, contexts, history=history)
                        try:
                            for token in stream:
                                answer += token
                                if not emit({'token': token}):
                                    return
                        finally:
                            close = getattr(stream,'close',None)
                            if close:
                                close()
                        metadata = outcome_metadata(answer)
                    else:
                        remaining_timeout(cfg.LLM_TIMEOUT)
                        response = finalize_response(direct if direct is not None else route(request,memory=memory,intent=intent))
                        answer, metadata = response.answer,response.metadata
                        sources = [item.model_dump() for item in response.sources]
                        if not emit({'token':answer}):
                                return
                except ModelOutputError as exc:
                    answer = str(exc.partial_text) or answer
                    metadata, error = exc.metadata,str(exc)
                except Exception as exc:
                    metadata = {'status':'cancelled' if isinstance(exc,RunCancelled) else 'error','completion_status':'error'}
                    error = _safe_error(exc).detail
                finally:
                    if stopped.is_set() or time.monotonic() >= deadline:
                        metadata.update(status='incomplete',completion_status='incomplete',error_code='stream_interrupted')
                    refs = _recorded_evidence(metadata,context,answer)
                    metadata.update(session_id=session_id,run_id=context.run_id)
                    metadata.update(context_metrics=dict(context.context_metrics),model_usage=model_metrics_snapshot(context))
                    metadata['session_context'] = {key:value for key,value in (context.task_context or {}).items() if key!='text'}
                    turn_id = memory.append_turn(request.query,answer,metadata,
                        include_in_context=can_commit_answer(answer,metadata),evidence_refs=refs)
                    metadata['turn_id'] = turn_id
            event = {'sources':sources,'session_id':session_id,'metadata':metadata}
            if error:
                event['error'] = error
            emit(event)
            emit('[DONE]')
        except Exception as exc:
            emit({'error': _safe_error(exc).detail,
                  'metadata': {'status': 'error', 'completion_status': 'error'}, 'session_id': session_id})
            emit('[DONE]')

    async def event_generator():
        worker = threading.Thread(target=producer, daemon=True, name='email-rag-stream')
        worker.start()
        try:
            while time.monotonic() < deadline:
                try:
                    item = await asyncio.to_thread(events.get, True, .2)
                except queue.Empty:
                    if not worker.is_alive():
                        break
                    continue
                if item == '[DONE]':
                    yield 'data: [DONE]\n\n'
                    return
                yield f'data: {json.dumps(item, ensure_ascii=False)}\n\n'
            yield 'data: {"error":"Stream stopped or timed out"}\n\n'
        finally:
            stopped.set()
    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.delete('/chat/history')
def clear_history(session_id: str = Query(..., min_length=1, max_length=128), identity: Identity = Depends(require_identity)):
    try:
        # Tombstone first. Cross-database cleanup is deliberately idempotent;
        # a partial cleanup cannot grant access to an old epoch or restart it.
        sessions.clear(identity.owner_id, session_id, invalidate_active=True)
        cleanup=[]
        try:
            if _jobs is not None:
                _jobs.invalidate_session(identity.owner_id,session_id)
            elif os.path.isfile(cfg.JOB_STORE_PATH):
                JobStore(cfg.JOB_STORE_PATH).invalidate_session(identity.owner_id,session_id)
        except Exception as exc:
            cleanup.append({'component':'jobs','error_code':type(exc).__name__})
        try:
            if os.path.isfile(cfg.TOOL_RESULT_STORE_PATH):
                from core.tool_results import ToolResultStore
                ToolResultStore(cfg.TOOL_RESULT_STORE_PATH).delete_session(identity.owner_id,session_id)
        except Exception as exc:
            cleanup.append({'component':'tool_results','error_code':type(exc).__name__})
        return {'success': not cleanup, 'session_id': session_id, 'history_inaccessible':True,
                'cleanup_pending':cleanup}
    except Exception as exc:
        raise _safe_error(exc) from None


@app.get('/chat/sessions')
def list_sessions(limit: int = Query(50,ge=1,le=100), offset: int = Query(0,ge=0),
                  identity: Identity = Depends(require_identity)):
    return sessions.list_sessions_page(identity.owner_id,limit=limit,offset=offset)


@app.get('/chat/history')
def get_history(session_id: str = Query(...,min_length=1,max_length=128),
                after: int = Query(0,ge=0), limit: int = Query(50,ge=1,le=100),
                identity: Identity = Depends(require_identity)):
    return sessions.history(identity.owner_id,session_id,after=after,limit=limit)


@app.get('/chat/history/search')
def search_history(session_id: str = Query(...,min_length=1,max_length=128),
                   query: str = Query(...,min_length=1,max_length=200),
                   limit: int = Query(20,ge=1,le=100), identity: Identity = Depends(require_identity)):
    return {'turns':sessions.search_history(identity.owner_id,session_id,query,limit=limit)}


@app.get('/chat/history/summary')
def history_summary(session_id: str = Query(...,min_length=1,max_length=128),
                    after: int = Query(0,ge=0), limit: int = Query(20,ge=1,le=100),
                    identity: Identity = Depends(require_identity)):
    return sessions.summary(identity.owner_id,session_id,after=after,limit=limit)


@app.get('/chat/facts')
def get_task_facts(session_id: str = Query(...,min_length=1,max_length=128),
                   include_history: bool = Query(False),
                   identity: Identity = Depends(require_identity)):
    return {'facts':sessions.task_facts(identity.owner_id,session_id,include_history=include_history)}


@app.post('/chat/facts')
def set_task_fact(session_id: str = Body(...,min_length=1,max_length=128),
                  key: str = Body(...,min_length=1,max_length=100),
                  value: str = Body(...,max_length=2000),
                  source_turn_id: str = Body(...,min_length=1,max_length=128),
                  expected_version: int = Body(0,ge=0),
                  scope: Literal['task','session'] = Body('task'),
                  task_id: Optional[str] = Body(None,max_length=128), identity: Identity = Depends(require_identity)):
    try:
        return sessions.set_task_fact(identity.owner_id,session_id,key,value,
                    source_turn_id=source_turn_id,expected_version=expected_version,explicit_user=True,
                    scope=scope,task_id=task_id)
    except Exception as exc:
        raise _safe_error(exc) from None


@app.get('/chat/evidence')
def history_evidence(session_id: str = Query(...,min_length=1,max_length=128),
                     turn_id: Optional[str] = Query(None,max_length=128),
                     identity: Identity = Depends(require_identity)):
    return {'evidence':sessions.evidence_refs(identity.owner_id,session_id,turn_id=turn_id)}


# Read-only evidence endpoints share admission limits with model/index work.
from api.evidence_routes import evidence_router
app.include_router(evidence_router(admission, _safe_error))


def _run_background_job(job, cancel_event, progress, checkpoint_callback):
    request = job['request']
    sid = request.get('session_id')
    if job['kind']=='agent' and sessions.repository is not None:
        epoch=sessions.repository.context_epoch(job['owner'],sid)
        stored_epoch=request.get('_session_epoch',(job.get('checkpoint') or {}).get('context_epoch',0))
        if stored_epoch!=epoch:
            raise ValueError('Session was deleted after this task was submitted')
    context = RunContext(owner_id=job['owner'],session_id=sid,run_id=job['id'],
        deadline=time.monotonic()+cfg.AGENT_RUN_TIMEOUT if job['kind']=='agent' else None,
        context_char_limit=cfg.AGENT_CONTEXT_CHAR_LIMIT,cancel_event=cancel_event,
        progress_callback=progress,checkpoint_callback=checkpoint_callback)
    context.resuming = bool(job.get('checkpoint'))
    if job['kind'] == 'agent':
        from agents.agent_loop import run_agent_loop
        req = AgentRequest(**request)
        return _chat_turn(req,Identity(job['owner']),lambda r,m:run_agent_loop(r,m,
            owner_id=job['owner'],session_id=sid,checkpoint=job['checkpoint']),
            context=context,admitted=True).model_dump()
    from core.ingestion import prepare_email_chunks
    from core.index_metrics import collect_index_metrics
    with use_run_context(context):
        with collect_index_metrics() as report:
            if job['checkpoint'] and job['checkpoint'].get('generation_id'):
                from core.embedder import resume_index_generation
                count = resume_index_generation(job['checkpoint']['generation_id'])
                result = {'chunk_count':count,'metadata':{'status':'success'}}
            else:
                context.progress('validating')
                emails,chunks = prepare_email_chunks(request.get('data_path'))
                try:
                    email_count = len(emails)
                    count = index_chunks(chunks)
                finally:
                    close = getattr(emails,'close',None)
                    if close:
                        close()
                result = {'email_count':email_count,'chunk_count':count,'metadata':{'status':'success'}}
        if report.outcome != 'unchanged':
            from core.retriever import invalidate_bm25_cache
            invalidate_bm25_cache()
        result['index_metrics'] = report.to_dict()
        return result


def _job_manager():
    global _jobs
    with _jobs_lock:
        if _jobs is None:
            _jobs = JobManager(JobStore(cfg.JOB_STORE_PATH),_run_background_job,
                gate=admission,max_workers=cfg.MAX_BACKGROUND_JOBS)
    return _jobs


@app.post('/jobs/agent', status_code=202)
def submit_agent_job(request: AgentRequest, identity: Identity = Depends(require_identity)):
    try:
        # Persist identity and logical operation before any model/tool execution.
        operation_key = request.operation_key or str(uuid.uuid4())
        req = request.model_copy(update={'session_id':request.session_id or str(uuid.uuid5(uuid.NAMESPACE_URL,identity.owner_id+':'+operation_key)),
                                         'operation_key':operation_key})
        payload=req.model_dump()
        if sessions.repository is not None:
            payload['_session_epoch']=sessions.repository.context_epoch(identity.owner_id,req.session_id)
        return _job_manager().submit(identity.owner_id,'agent',payload,req.operation_key)
    except Exception as exc:
        raise _safe_error(exc) from None


@app.post('/jobs/index',status_code=202)
def submit_index_job(request: Optional[IndexRequest] = Body(None), identity: Identity = Depends(require_identity)):
    try:
        return _job_manager().submit(identity.owner_id,'index',request.model_dump() if request else {})
    except Exception as exc:
        raise _safe_error(exc) from None


@app.get('/jobs')
def list_jobs(limit: int = Query(25,ge=1,le=100), offset: int = Query(0,ge=0), identity: Identity = Depends(require_identity)):
    page=_job_manager().store.list(identity.owner_id,limit=limit,offset=offset)
    page['jobs']=[_visible_job(identity.owner_id,job['id']) for job in page['jobs']]
    return page


@app.get('/jobs/{job_id}')
def get_job(job_id: str, identity: Identity = Depends(require_identity)):
    try:
        return _visible_job(identity.owner_id,job_id)
    except Exception as exc:
        raise _safe_error(exc) from None


@app.get('/jobs/operations/{operation_key}')
def find_job_operation(operation_key: str, identity: Identity = Depends(require_identity)):
    try:
        job=_job_manager().store.find_operation(identity.owner_id,operation_key)
        return _visible_job(identity.owner_id,job['id'])
    except Exception as exc:
        raise _safe_error(exc) from None


@app.post('/jobs/{job_id}/cancel')
def cancel_job(job_id: str, identity: Identity = Depends(require_identity)):
    try:
        return _job_manager().cancel(identity.owner_id,job_id)
    except Exception as exc:
        raise _safe_error(exc) from None


@app.post('/jobs/{job_id}/resume',status_code=202)
def resume_job(job_id: str, identity: Identity = Depends(require_identity)):
    try:
        if _visible_job(identity.owner_id,job_id).get('error_code')=='session_invalidated':
            raise ValueError('Session memory was deleted; this job cannot resume')
        return _job_manager().resume(identity.owner_id,job_id)
    except Exception as exc:
        raise _safe_error(exc) from None


@app.get('/agent/approvals')
def list_agent_approvals(status: Optional[str] = None, limit: int = Query(50,ge=1,le=200),
                        cursor: Optional[str] = Query(None,max_length=1024),
                        include_payload: bool = Query(True), identity: Identity = Depends(require_identity)):
    try:
        page = ApprovalStore().list_page(status=status,owner_id=identity.owner_id,limit=limit,
                                        cursor=cursor,include_payload=include_payload)
        return {'approvals':page['items'],'next_cursor':page['next_cursor'],'limit':page['limit']}
    except Exception as exc:
        raise _safe_error(exc) from None


@app.post('/agent/approvals/{approval_id}/approve')
@_bounded_endpoint
def approve_agent_action(approval_id: str, reviewer: str = Body('human'), note: str = Body('', max_length=2000),
                         identity: Identity = Depends(require_identity)):
    """Reviewer is accepted for compatibility; the trusted identity is persisted."""
    try:
        provider = create_mail_provider_from_settings()
        return ApprovalStore().approve(approval_id, reviewer=identity.owner_id, note=note,
                                       owner_id=identity.owner_id, executor=provider.execute_approval)
    except Exception as exc:
        raise _safe_error(exc) from None


@app.post('/agent/approvals/{approval_id}/reject')
def reject_agent_action(approval_id: str, reviewer: str = Body('human'), note: str = Body('', max_length=2000),
                        identity: Identity = Depends(require_identity)):
    try:
        return ApprovalStore().reject(approval_id, reviewer=identity.owner_id, note=note, owner_id=identity.owner_id)
    except Exception as exc:
        raise _safe_error(exc) from None


@app.post('/agent/approvals/{approval_id}/reconcile')
def reconcile_agent_action(approval_id: str,
        outcome: Literal['succeeded', 'not_executed', 'unresolved'] = Body(...),
        evidence: str = Body(..., min_length=1, max_length=2000),
        expected_payload_hash: str = Body(..., pattern=r'^[a-f0-9]{64}$'),
        result: Optional[dict] = Body(None), execution_stopped: bool = Body(False),
        identity: Identity = Depends(require_identity)):
    try:
        return ApprovalStore().reconcile(approval_id, outcome, evidence,
            reviewer=identity.owner_id, owner_id=identity.owner_id,
            expected_payload_hash=expected_payload_hash, result=result,
            execution_stopped=execution_stopped)
    except Exception as exc:
        raise _safe_error(exc) from None


@app.get('/agent/mcp-audit')
def list_mcp_audit_events(tool: Optional[str] = None, status: Optional[str] = None,
                          limit: int = Query(100, ge=1, le=500), identity: Identity = Depends(require_identity)):
    events = MCPAuditLogger.load_events(tool=tool, status=status, limit=limit)
    return {'count': len(events), 'events': events}


@app.post('/query', response_model=QueryResponse)
@_bounded_endpoint
def query(request: QueryRequest, identity: Identity = Depends(require_identity)):
    from core.pipeline import retrieve
    from core.generator import generate_answer
    try:
        context = RunContext(owner_id=identity.owner_id, deadline=time.monotonic()+cfg.AGENT_RUN_TIMEOUT,
                             context_char_limit=cfg.AGENT_CONTEXT_CHAR_LIMIT)
        with use_run_context(context):
            results = retrieve(request.query, top_n=request.top_k)
            answer = generate_answer(request.query, results)
            metadata={**outcome_metadata(answer),'run_id':context.run_id,
                'context_metrics':dict(context.context_metrics),'model_usage':model_metrics_snapshot(context)}
            _recorded_evidence(metadata,context,display_text(answer))
            return QueryResponse(answer=display_text(answer), sources=results, metadata=metadata)
    except Exception as exc:
        raise _safe_error(exc) from None


def _visible_job(owner,job_id,*,store=None):
    """Tombstones prevent stale private results leaking after partial cleanup."""
    private=(store if store is not None else _job_manager().store).get(owner,job_id,private=True)
    public={key:value for key,value in private.items() if key not in ('request','checkpoint','request_hash','owner')}
    sid=(private.get('request') or {}).get('session_id')
    if sid and sessions.repository is not None:
        epoch=sessions.repository.context_epoch(owner,sid)
        original=(private.get('request') or {}).get('_session_epoch',
                    (private.get('checkpoint') or {}).get('context_epoch',0))
        if original!=epoch:
            public.update(status='cancelled',result=None,progress={},resumable=False,
                          error_code='session_invalidated',resume_block_reason='session_invalidated')
    return public


def _context_progress(owner, sid):
    store=_jobs.store if _jobs is not None else (JobStore(cfg.JOB_STORE_PATH) if os.path.isfile(cfg.JOB_STORE_PATH) else None)
    if store is None:return []
    return [{key:row.get(key) for key in ('id','kind','status','progress','error_code')}
            for row in (_visible_job(owner,item['id'],store=store) for item in store.session_jobs(owner,sid))]


def _rebuild_summary(owner, sid):
    if not cfg.ENABLE_CONTEXT_SUMMARY:
        raise HTTPException(409, 'Semantic summary experiment is disabled; deterministic history remains available')
    context=RunContext(owner_id=owner,session_id=sid,deadline=time.monotonic()+cfg.AGENT_RUN_TIMEOUT,
                       context_char_limit=cfg.AGENT_CONTEXT_CHAR_LIMIT)
    with admission.slot(), use_run_context(context):
        result=sessions.generate_summary(owner,sid,generate=_summary_generate,budget=context,force=True,
            min_turns=1,max_attempts=cfg.CONTEXT_SUMMARY_MAX_ATTEMPTS,
            model_id=cfg.DEEPSEEK_MODEL,model_revision=cfg.MODEL_REVISION or 'unknown')
        return {**result,'model_usage':model_metrics_snapshot(context)}


def _read_stored_result(owner,sid,run_id,result_id,*,start,limit):
    from core.tool_results import ToolResultStore
    if sessions.repository is None or not os.path.isfile(cfg.TOOL_RESULT_STORE_PATH):
        raise KeyError('tool result store unavailable')
    epoch=sessions.repository.context_epoch(owner,sid)
    page=ToolResultStore(cfg.TOOL_RESULT_STORE_PATH).page(result_id,owner=owner,session=sid,run=run_id,
                                                       epoch=epoch,start=start,limit=limit)
    sessions.repository.validate_context_epoch(owner,sid,epoch)
    return page


from api.context_routes import context_router
app.include_router(context_router(lambda:sessions,_safe_error,progress_provider=_context_progress,
                                  summary_rebuild=_rebuild_summary,result_reader=_read_stored_result))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api.main:app', host=cfg.API_HOST, port=cfg.API_PORT)
