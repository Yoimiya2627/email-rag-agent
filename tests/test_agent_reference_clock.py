import copy
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, Mock

import pytest
from core import pipeline
from core.run_reference import run_reference, restore_reference, timezone_policy
from core.tool_results import ToolResultUnavailable
from agents.execution_scope import ExecutionScope, use_execution_scope
from tests.test_agent_checkpoints import run, client, reply, setup_backend
from models.schemas import AgentRequest
import agents.agent_loop as loop

T = datetime(2026,4,28,23,59,59,tzinfo=timezone.utc)


def capture():
    snapshots=[]
    req=AgentRequest(query="read",session_id="s")
    with pipeline._fixed_reference_time(T):
        run(req,client([reply("prefix",finish="length")]),setup_backend(),snapshots)
    return req,snapshots[-1]


def test_resume_answer_only_uses_saved_instant_and_resets():
    req,cp=capture()
    assert cp["version"] == 1
    assert cp["reference_clock"]["instant_utc"] == T.isoformat()
    observed=[]
    model=client([reply("suffix")])
    create=model.chat.completions.create
    def respond(**kwargs):
        observed.append(pipeline._now())
        return create(**kwargs)
    model.chat.completions.create=respond
    out=run(req,model,setup_backend(),[],checkpoint=cp)
    assert out.answer == "prefix\nsuffix"
    assert observed == [T]
    assert pipeline._reference_time.get() is None


@pytest.mark.parametrize("mutation,code",[
    (lambda c:c.pop("reference_clock"),"checkpoint_reference_requires_restart"),
    (lambda c:c["reference_clock"].update(version=99),"checkpoint_reference_version_unsupported"),
    (lambda c:c["reference_clock"].update(instant_utc="2026-04-28"),"checkpoint_reference_invalid"),
    (lambda c:c.update(context_fingerprint="changed"),"checkpoint_context_policy_changed"),
    (lambda c:c.update(owner_id="other"),"checkpoint_scope_mismatch"),
    (lambda c:c["reference_clock"].update(timezone={"kind":"offset","microseconds":99}),"checkpoint_reference_timezone_changed"),
])
def test_invalid_restore_never_initializes_clients_or_backend(mutation,code):
    req,cp=capture();mutation(cp)
    with patch.object(loop,"_get_client") as model,patch.object(loop,"_get_tool_backend") as backend:
        with pytest.raises(ToolResultUnavailable,match=code):
            loop.run_agent_loop(req,session_id="s",checkpoint=cp)
        model.assert_not_called();backend.assert_not_called()
    assert pipeline._reference_time.get() is None


def test_scope_conflict_rejected_and_same_instant_accepted():
    req,cp=capture()
    def scope(value):
        return ExecutionScope(owner_id="eval:test",evaluation=True,approval_store_path="unused",run_dir="unused",evaluation_reference_time=value)
    with use_execution_scope(scope(T.astimezone(timezone(timedelta(hours=-7))))):
        with run_reference(cp):
            assert pipeline._now() == T
    with use_execution_scope(scope(T+timedelta(seconds=1))):
        with pytest.raises(ToolResultUnavailable,match="checkpoint_reference_conflict"):
            with run_reference(cp):
                pass
    assert pipeline._reference_time.get() is None


def test_fresh_run_reads_real_clock_only_once_across_tools():
    from types import SimpleNamespace as NS
    # Keep datetime type validation intact; mock only the initial _now read.
    original=pipeline._now
    reads=[]
    def clock():
        if pipeline._reference_time.get() is None:
            reads.append(1)
            return T+timedelta(days=len(reads)-1)
        return original()
    seen=[]
    backend=setup_backend()
    backend.call_tool.side_effect=lambda *a:seen.append(pipeline._now()) or {"status":"success","data":{}}
    call=NS(id="c",function=NS(name="send_email",arguments="{}"))
    with patch.object(pipeline,"_now",side_effect=clock):
        result=run(AgentRequest(query="read",session_id="s"),client([reply("",[call],"tool_calls"),reply("done")]),backend,[])
    assert result.metadata["status"] == "success"
    assert len(reads)==1 and seen == [T]
    assert pipeline._reference_time.get() is None


def test_dst_policy_and_nested_exception_restore():
    from zoneinfo import ZoneInfo
    tz=ZoneInfo("America/Los_Angeles")
    assert timezone_policy(tz) != timezone_policy(ZoneInfo("America/Vancouver"))
    with pipeline._fixed_reference_time(T):
        with pytest.raises(RuntimeError):
            with run_reference():
                raise RuntimeError("fixture")
        assert pipeline._now()==T
    assert pipeline._reference_time.get() is None


@pytest.mark.parametrize("policy", [{"kind":"offset","microseconds":True}, {"kind":"iana","name":"UTC","extra":1}, {"kind":"offset","microseconds":0.1}])
def test_malformed_timezone_policy_rejected(policy):
    _,cp=capture()
    cp["reference_clock"]["timezone"]=policy
    with pytest.raises(ToolResultUnavailable,match="checkpoint_reference_invalid"):
        restore_reference(cp,timezone.utc)


def test_fractional_offset_is_preserved_exactly():
    tz=timezone(timedelta(microseconds=1234567))
    assert timezone_policy(tz)=={"kind":"offset","microseconds":1234567}


def test_changed_valid_reference_rejected_before_clients_without_scope():
    req,cp=capture()
    cp["reference_clock"]["instant_utc"]=(T+timedelta(hours=1)).isoformat()
    with patch.object(loop,"_get_client") as model,patch.object(loop,"_get_tool_backend") as backend:
        with pytest.raises(ToolResultUnavailable,match="checkpoint_context_policy_changed"):
            loop.run_agent_loop(req,session_id="s",checkpoint=cp)
        model.assert_not_called();backend.assert_not_called()


def test_equivalent_offset_encoding_normalizes_before_fingerprint():
    req,cp=capture()
    cp["reference_clock"]["instant_utc"]=T.astimezone(timezone(timedelta(hours=-7))).isoformat()
    assert run(req,client([reply("suffix")]),setup_backend(),[],checkpoint=cp).answer == "prefix\nsuffix"


def test_missing_outer_version_rejected_before_clients():
    req,cp=capture();cp.pop("version")
    with patch.object(loop,"_get_client") as model,patch.object(loop,"_get_tool_backend") as backend:
        with pytest.raises(ToolResultUnavailable,match="checkpoint_policy_mismatch"):
            loop.run_agent_loop(req,session_id="s",checkpoint=cp)
        model.assert_not_called();backend.assert_not_called()


def test_actual_legacy_v1_shape_requires_restart_before_owner_validation():
    req,cp=capture()
    for field in ("reference_clock", "owner_id", "session_id", "context_fingerprint"):
        cp.pop(field)
    with patch.object(loop,"_get_client") as model,patch.object(loop,"_get_tool_backend") as backend:
        with pytest.raises(ToolResultUnavailable,match="checkpoint_reference_requires_restart"):
            loop.run_agent_loop(req,session_id="s",checkpoint=cp)
        model.assert_not_called();backend.assert_not_called()


def test_reference_error_code_is_a_strict_safe_allowlist():
    from core.run_reference import reference_error_code
    for code in ("checkpoint_reference_requires_restart", "checkpoint_reference_invalid",
                 "checkpoint_reference_version_unsupported", "checkpoint_reference_timezone_changed",
                 "checkpoint_reference_conflict", "run_reference_conflict"):
        assert reference_error_code(ToolResultUnavailable(code)) == code
        assert reference_error_code(ValueError(code)) is None
    assert reference_error_code(ToolResultUnavailable("private response")) is None
    assert reference_error_code(ToolResultUnavailable("checkpoint_reference_invalid", "private")) is None
