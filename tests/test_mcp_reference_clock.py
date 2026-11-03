"""Offline MCP reference-clock protocol tests at real adapter/server boundaries."""
import asyncio
import contextvars
import copy
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from agents.mcp_adapter import (REFERENCE_CLOCK_META, MCPReferenceClockError,
    MCPToolBackend, StreamableHttpMCPClient, normalize_mcp_result)
from agents.runtime import RunCancelled, current_run
from core.run_reference import prepare_reference, run_reference

T0=datetime(2026,4,28,12,tzinfo=timezone.utc)


@pytest.fixture
def protocol(monkeypatch):
    import mcp
    import mcp.client.streamable_http as http
    import agents.tools as tools
    import mcp_server as server
    import core.pipeline as pipeline
    monkeypatch.setattr(pipeline.cfg,'RETRIEVAL_TIMEZONE','')
    monkeypatch.setattr(pipeline.cfg,'RETRIEVAL_TIMEZONE_OFFSET_HOURS',8)
    monkeypatch.setattr(pipeline.cfg,'MCP_OWNER_ID','server-owner')
    state=NS(tool_name='email_stats', capability=1, calls=[], observed=[], request_meta=None, alter_echo=None, tool_error=None)
    def business(name,args):
        state.observed.append({'time':pipeline._now(),'owner':current_run().owner_id,
                               'deadline':current_run().deadline})
        if state.tool_error:raise state.tool_error
        return {'value':1}
    monkeypatch.setattr(tools,'call_tool',business)
    wrapped=server._trusted_tool('email_stats',tools.email_stats,
        context_provider=lambda:NS(request_context=NS(meta=state.request_meta)))
    class Session:
        def __init__(self,*args):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def initialize(self):pass
        async def list_tools(self):
            return NS(tools=[NS(name=state.tool_name,description='stats',inputSchema=({'type':'object','properties':{}} if state.tool_name=='email_stats' else tools.TOOL_REGISTRY[state.tool_name].parameters),
                meta={} if state.capability is None else {REFERENCE_CLOCK_META:state.capability})])
        async def call_tool(self,name,arguments=None,**kwargs):
            state.calls.append({'name':name,'arguments':copy.deepcopy(arguments),'kwargs':copy.deepcopy(kwargs)})
            state.request_meta=mcp.types.RequestParams.Meta(**kwargs['meta']) if 'meta' in kwargs else None
            # Model a separate server process: client ContextVars do NOT transfer.
            target=wrapped if name=='email_stats' else server._trusted_tool(name,getattr(tools,name),
                context_provider=lambda:NS(request_context=NS(meta=state.request_meta)))
            result=contextvars.Context().run(lambda:target(**(arguments or {})))
            if state.alter_echo:state.alter_echo(result)
            return NS(isError=False,structuredContent=result)
    @asynccontextmanager
    async def transport(*args,**kwargs):yield None,None,None
    monkeypatch.setattr(mcp,'ClientSession',Session)
    monkeypatch.setattr(http,'streamablehttp_client',transport)
    state.transport=StreamableHttpMCPClient('http://offline.invalid/mcp')
    state.backend=MCPToolBackend(state.transport)
    state.pipeline=pipeline;state.wrapper=wrapped
    return state


def contract():
    from core.pipeline import _fixed_reference_time
    with _fixed_reference_time(T0):return prepare_reference()[1]


def test_frozen_transport_applies_and_echoes_clock_without_business_parameters(protocol):
    p=protocol; clock=contract()
    schema=p.backend.tool_schemas()
    assert schema[0]['function']['parameters']=={'type':'object','properties':{}}
    p.backend.validate_reference_clock(clock)
    with run_reference({'reference_clock':clock}):
        first=normalize_mcp_result(p.transport.call_tool('email_stats',{}))
        second=normalize_mcp_result(p.transport.call_tool('email_stats',{}))
    assert first['reference_clock']==second['reference_clock']==clock
    assert [row['time'] for row in p.observed]==[T0,T0]
    assert all(row['owner']=='server-owner' for row in p.observed)
    assert all(call['arguments']=={} and call['kwargs']=={'meta':{REFERENCE_CLOCK_META:clock}} for call in p.calls)


@pytest.mark.parametrize('capability',[None,True,2,'1'])
def test_old_or_wrong_server_capability_is_rejected_before_business_call(protocol,capability):
    p=protocol;p.capability=capability;p.backend.tool_schemas()
    with pytest.raises(MCPReferenceClockError):p.backend.validate_reference_clock(contract())
    with run_reference({'reference_clock':contract()}),pytest.raises(MCPReferenceClockError):
        p.transport.call_tool('email_stats',{})
    assert p.calls==p.observed==[]


def test_unfrozen_direct_transport_keeps_legacy_live_compatibility(protocol):
    p=protocol;p.capability=None
    result=normalize_mcp_result(p.transport.call_tool('email_stats',{}))
    assert result['status']=='success'
    assert p.calls[0]['kwargs']=={} and 'reference_clock' not in result
    assert p.observed[0]['time']!=T0


@pytest.mark.parametrize('change',['missing','different_instant','extra_field'])
def test_server_echo_must_match_actual_requested_contract(protocol,change):
    p=protocol
    def alter(payload):
        if change=='missing':payload.pop('reference_clock',None)
        elif change=='different_instant':payload['reference_clock']['instant_utc']=(T0+timedelta(days=1)).isoformat()
        else:payload['reference_clock']['owner']='attacker'
    p.alter_echo=alter
    with run_reference({'reference_clock':contract()}),pytest.raises(MCPReferenceClockError) as caught:
        p.transport.call_tool('email_stats',{})
    assert caught.value.error_code=='mcp_reference_clock_mismatch'


@pytest.mark.parametrize('bad',[None,{}, {'version':True}, {'version':1,'instant_utc':'2026-01-01','timezone':{} }])
def test_malformed_server_clock_metadata_never_executes_tool(protocol,bad):
    p=protocol;p.request_meta={REFERENCE_CLOCK_META:bad}
    result=contextvars.Context().run(p.wrapper)
    assert result['error_code']=='mcp_reference_clock_invalid'
    assert p.observed==[]


def test_server_exception_restores_clock_and_does_not_freeze_deadline(protocol):
    import time
    p=protocol;p.request_meta={REFERENCE_CLOCK_META:contract()};p.tool_error=RunCancelled('fixture')
    def isolated():
        before=p.pipeline._now()
        with pytest.raises(RunCancelled):p.wrapper()
        after=p.pipeline._now()
        assert before!=T0 and after!=T0
    contextvars.Context().run(isolated)
    assert p.observed[0]['time']==T0
    assert p.observed[0]['deadline']>time.monotonic()


def test_real_sdk_tool_metadata_advertises_clock_without_argument_schema_changes():
    import mcp_server
    from mcp.server.fastmcp import FastMCP
    server=FastMCP('offline')
    mcp_server.register_tools(server)
    listed=asyncio.run(server.list_tools())
    assert listed
    for tool in listed:
        assert tool.meta[REFERENCE_CLOCK_META]==1
        assert 'reference_clock' not in tool.inputSchema.get('properties',{})
        assert 'owner_id' not in tool.inputSchema.get('properties',{})


def test_context_provider_failure_is_not_silently_live(monkeypatch):
    import mcp_server
    import agents.tools as tools
    seen=[]
    monkeypatch.setattr(tools,'call_tool',lambda *args:seen.append(True))
    def broken():raise RuntimeError('private context failure')
    wrapped=mcp_server._trusted_tool('email_stats',tools.email_stats,context_provider=broken)
    result=wrapped()
    assert result['error_code']=='mcp_reference_clock_invalid' and seen==[]



def test_real_loop_rejects_legacy_server_before_model_client(protocol, monkeypatch):
    import agents.agent_loop as loop
    from models.schemas import AgentRequest
    p=protocol;p.capability=None
    monkeypatch.setattr(loop.cfg,'AGENT_TOOL_BACKEND','mcp')
    monkeypatch.setattr(loop,'_get_tool_backend',lambda:p.backend)
    model_access=[]
    def forbidden():model_access.append(True);raise AssertionError('model client must not be reached')
    monkeypatch.setattr(loop,'_get_client',forbidden)
    with pytest.raises(MCPReferenceClockError):
        loop.run_agent_loop(AgentRequest(query='retrieve recent messages'))
    assert model_access==[] and p.calls==p.observed==[]


def test_real_fastmcp_get_context_routes_sdk_metadata_and_restores_clock(monkeypatch):
    import mcp_server
    import agents.tools as tools
    import core.pipeline as pipeline
    from mcp.server.fastmcp import FastMCP
    from mcp.server.lowlevel.server import request_ctx
    from mcp.shared.context import RequestContext
    from mcp.types import RequestParams
    clock=contract(); observed=[]
    monkeypatch.setattr(tools,'call_tool',lambda name,args:observed.append((pipeline._now(),current_run().owner_id)) or {'value':1})
    monkeypatch.setattr(pipeline.cfg,'MCP_OWNER_ID','fixed-owner')
    server=FastMCP('offline-metadata')
    mcp_server.register_tools(server)
    async def invoke():
        token=request_ctx.set(RequestContext(request_id='fixture',session=None,lifespan_context=None,
            meta=RequestParams.Meta(**{REFERENCE_CLOCK_META:clock,'owner_id':'forged-owner'})))
        try:return await server.call_tool('email_stats',{})
        finally:request_ctx.reset(token)
    blocks,structured=asyncio.run(invoke())
    assert structured['reference_clock']==clock
    assert observed==[(T0,'fixed-owner')]
    assert pipeline._now()!=T0



def test_missing_echo_after_remote_approval_is_unknown_cached_and_audited(protocol, monkeypatch):
    from agents.runtime import RunContext, use_run_context
    import config.settings as cfg
    p=protocol;p.tool_name='send_email'
    monkeypatch.setattr(cfg,'MCP_READ_ONLY_MODE',False)
    monkeypatch.setattr(cfg,'MCP_ALLOWED_TOOLS',[])
    p.alter_echo=lambda payload:payload.pop('reference_clock',None)
    events=[]
    p.backend.audit_logger=NS(record=lambda **event:events.append(event))
    p.backend.tool_schemas()
    clock=contract();p.backend.validate_reference_clock(clock)
    arguments={'to':['recipient@example.test'],'subject':'fixture','body':'fixture','rationale':'fixture'}
    with use_run_context(RunContext(owner_id='server-owner',run_id='same-run')),run_reference({'reference_clock':clock}):
        first=p.backend.call_tool('send_email',arguments)
        second=p.backend.call_tool('send_email',arguments)
    assert first==second
    assert first['status']=='unknown' and first['error_code']=='mcp_reference_clock_mismatch'
    assert first['side_effect_state']=='unknown'
    assert len(p.calls)==len(p.observed)==1
    assert len(events)==2
    assert events[0]['status']=='unknown' and events[0]['error_type']=='MCPReferenceClockError'
    assert events[1]['status']=='unknown' and events[1]['cache_hit'] is True
    assert all('arguments' not in event and 'reference_clock' not in event for event in events)
