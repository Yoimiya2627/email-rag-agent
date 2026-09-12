"""Owner-bound inspection and explicit user edits. Never registered as LLM writes."""
from typing import Literal
from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from api.security import Identity, require_identity


class TaskUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1,max_length=128)
    task_id: str = Field(min_length=1,max_length=128)
    goal: str = Field(default='',max_length=2000)
    objects: list[str] = Field(default_factory=list,max_length=30)
    open_questions: list[str] = Field(default_factory=list,max_length=30)
    source_turn_id: str = Field(min_length=1,max_length=128)
    expected_revision: int = Field(ge=0)


class FactAction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1,max_length=128)
    key: str = Field(min_length=1,max_length=100)
    source_turn_id: str = Field(min_length=1,max_length=128)
    expected_version: int = Field(ge=1)
    scope: Literal['task','session'] = 'task'
    task_id: str | None = Field(default=None,max_length=128)


def context_router(session_provider, safe_error, *, progress_provider=None, summary_rebuild=None, result_reader=None):
    router = APIRouter()

    def invoke(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise safe_error(exc) from None

    @router.get('/chat/context')
    def context_state(session_id: str=Query(...,min_length=1,max_length=128),
                      identity: Identity=Depends(require_identity)):
        state=invoke(session_provider().context_state,identity.owner_id,session_id)
        state['execution_progress']=invoke(progress_provider,identity.owner_id,session_id) if progress_provider else []
        state['execution_authority']=False
        return state

    @router.get('/chat/tasks')
    def tasks(session_id: str=Query(...,min_length=1,max_length=128),identity: Identity=Depends(require_identity)):
        return {'tasks':invoke(session_provider().list_tasks,identity.owner_id,session_id)}

    @router.post('/chat/tasks')
    def update_task(request: TaskUpdate,identity: Identity=Depends(require_identity)):
        data=request.model_dump(); sid=data.pop('session_id')
        return invoke(session_provider().update_task,identity.owner_id,sid,**data,explicit_user=True)

    @router.post('/chat/tasks/select')
    def select_task(session_id: str=Body(...,min_length=1,max_length=128),
                    task_id: str=Body(...,min_length=1,max_length=128),expected_revision: int=Body(...,ge=0),
                    identity: Identity=Depends(require_identity)):
        return invoke(session_provider().select_task,identity.owner_id,session_id,task_id=task_id,
                      expected_revision=expected_revision,explicit_user=True)

    @router.post('/chat/facts/{action}')
    def fact_action(action: Literal['revoke','confirm','delete'],request: FactAction,
                    identity: Identity=Depends(require_identity)):
        data=request.model_dump();sid=data.pop('session_id');key=data.pop('key')
        method=getattr(session_provider(),{'revoke':'revoke_task_fact','confirm':'confirm_task_fact',
                                          'delete':'delete_task_fact'}[action])
        return invoke(method,identity.owner_id,sid,key,**data,explicit_user=True)

    @router.get('/chat/history/turn')
    def turn_page(session_id: str=Query(...,min_length=1,max_length=128),
                  turn_id: str=Query(...,min_length=1,max_length=128),
                  field: Literal['query','answer']='answer',offset: int=Query(0,ge=0),
                  limit: int=Query(1200,ge=1,le=4000),epoch: int|None=Query(None,ge=0),
                  identity: Identity=Depends(require_identity)):
        return invoke(session_provider().get_turn_page,identity.owner_id,session_id,turn_id,
                      field=field,offset=offset,limit=limit,expected_epoch=epoch)

    @router.get('/chat/summary')
    def summary(session_id: str=Query(...,min_length=1,max_length=128),identity: Identity=Depends(require_identity)):
        return {'summary':invoke(session_provider().get_semantic_summary,identity.owner_id,session_id),
                'kind':'derived_summary_not_source_evidence'}

    @router.post('/chat/summary/rebuild')
    def rebuild(session_id: str=Body(...,embed=True,min_length=1,max_length=128),
                identity: Identity=Depends(require_identity)):
        return invoke(summary_rebuild,identity.owner_id,session_id)

    @router.get('/chat/tool-result')
    def tool_result(session_id: str=Query(...,min_length=1,max_length=128),
                    run_id: str=Query(...,min_length=1,max_length=128),
                    result_id: str=Query(...,min_length=1,max_length=128),
                    start: int=Query(0,ge=0),limit: int=Query(1200,ge=1,le=4000),
                    identity: Identity=Depends(require_identity)):
        return invoke(result_reader,identity.owner_id,session_id,run_id,result_id,start=start,limit=limit)

    return router
