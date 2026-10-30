"""MCP server exposing the Email RAG Agent tool layer.

This module is deliberately thin: tool metadata lives in
``agents.tool_registry`` and implementations remain in ``agents.tools``.  The
server registers those tools with FastMCP so external MCP hosts can discover
and call the same capabilities that `/chat/agent` currently uses locally.
"""
from __future__ import annotations

import argparse
import functools
import hmac
import inspect
import json
import hashlib
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable

from pydantic import Field

import config.settings as cfg
from agents.tool_policy import ToolPolicy
from agents.tool_registry import TOOL_REGISTRY, ToolSpec, tool_dispatch
from agents.tools import email_stats, get_email
from agents.runtime import RunContext, use_run_context, normalize_tool_result, tool_error

_AUDIT_LOCK = threading.Lock()


def _server_audit(event: str, *, request_id: str, target: str, status: str,
                  started: float, run_id: str | None = None) -> None:
    """Server-boundary scalars only; do not copy tool arguments or mail text."""
    if not getattr(cfg, 'ENABLE_MCP_AUDIT', True):
        return
    owner = str(getattr(cfg, 'MCP_OWNER_ID', getattr(cfg, 'API_OWNER_ID', 'local')))
    row = {'event': event, 'boundary': 'server', 'request_id': request_id, 'run_id': run_id,
           'owner_ref': hashlib.sha256(owner.encode('utf-8')).hexdigest(),
           'target': target, 'status': status,
           'latency_ms': round((time.perf_counter() - started) * 1000, 2),
           'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        default = Path(cfg.MCP_AUDIT_LOG_PATH).with_name('mcp_server.jsonl')
        path = Path(getattr(cfg, 'MCP_SERVER_AUDIT_LOG_PATH', default))
        from agents.log_storage import append_jsonl
        append_jsonl(path,row,max_bytes=getattr(cfg,'LOG_MAX_BYTES',5_000_000),
                     backups=getattr(cfg,'LOG_BACKUP_COUNT',3))
    except (OSError, TypeError, ValueError):
        logging.getLogger(__name__).warning('MCP server audit write failed')


class StaticBearerTokenVerifier:
    """Minimal bearer-token verifier for local/portfolio MCP deployments."""

    def __init__(self, token: str, client_id: str = "email-rag-agent"):
        self.token = token
        self.client_id = client_id

    async def verify_token(self, token: str):
        started, request_id = time.perf_counter(), str(uuid.uuid4())
        if not hmac.compare_digest(token.encode('utf-8'), self.token.encode('utf-8')):
            _server_audit('authentication', request_id=request_id, target='bearer', status='denied', started=started)
            return None
        from mcp.server.auth.provider import AccessToken

        return AccessToken(
            token=token,
            client_id=self.client_id,
            scopes=["tools:read", "tools:call"],
            expires_at=int(time.time()) + 3600,
            resource=f"http://{cfg.MCP_HOST}:{cfg.MCP_PORT}/mcp",
        )


def _auth_kwargs() -> dict[str, Any]:
    """Return FastMCP auth kwargs when bearer-token auth is configured."""
    if not cfg.MCP_AUTH_TOKEN:
        return {}

    from mcp.server.auth.settings import AuthSettings

    base_url = f"http://{cfg.MCP_HOST}:{cfg.MCP_PORT}"
    return {
        "token_verifier": StaticBearerTokenVerifier(cfg.MCP_AUTH_TOKEN),
        "auth": AuthSettings(
            issuer_url=base_url,
            resource_server_url=f"{base_url}/mcp",
            required_scopes=["tools:read", "tools:call"],
        ),
    }


def _tool_decorator(server: Any, spec: ToolSpec) -> Callable:
    """Return a FastMCP tool decorator, with compatibility for older SDKs."""
    try:
        return server.tool(name=spec.name, description=spec.description)
    except TypeError:
        return server.tool()


def register_tools(server: Any, policy: ToolPolicy | None = None) -> list[str]:
    """Register every local email-agent tool on a FastMCP-compatible server."""
    dispatch = tool_dispatch()
    visible_specs = (policy or ToolPolicy.from_settings()).visible_specs(TOOL_REGISTRY)
    registered = []
    for name, spec in visible_specs.items():
        fn = dispatch[name]
        _tool_decorator(server, spec)(_trusted_tool(name, fn))
        registered.append(name)
    return registered


def _trusted_tool(name: str, function: Callable) -> Callable:
    """This MCP server has one configured owner, not caller-selected tenants."""
    @functools.wraps(function)
    def execute(*args, **kwargs):
        from agents.tools import call_tool
        started, request_id = time.perf_counter(), str(uuid.uuid4())
        try:
            arguments = dict(inspect.signature(function).bind(*args, **kwargs).arguments)
        except TypeError:
            _server_audit('tool_call', request_id=request_id, target=name, status='validation_error', started=started)
            return tool_error("validation_error", "Invalid tool arguments.")
        context = RunContext(
            owner_id=getattr(cfg, "MCP_OWNER_ID", getattr(cfg, "API_OWNER_ID", "local")),
            session_id="mcp",
            deadline=time.monotonic() + float(getattr(cfg, "AGENT_RUN_TIMEOUT", 120)),
            context_char_limit=int(getattr(cfg, "AGENT_CONTEXT_CHAR_LIMIT", 60000)),
        )
        with use_run_context(context):
            try:
                result = normalize_tool_result(call_tool(name, arguments))
            except BaseException:
                _server_audit('tool_call', request_id=request_id, target=name, status='error',
                              started=started, run_id=context.run_id)
                raise
            result["evidence_refs"] = [{"email_id": email, "chunk_id": chunk}
                                       for email, chunk in context.tool_evidence]
            # Sources are actual retrieval candidates, not model-verified citations.
            if context.tool_evidence and result["status"] == "success":
                result["candidate_sources"] = list(context.tool_evidence.values())
            result['server_request_id'] = request_id
            _server_audit('tool_call', request_id=request_id, target=name, status=result['status'],
                          started=started, run_id=context.run_id)
            return result
    # wraps() otherwise exposes the business function's list/str return type,
    # while this boundary always returns a result envelope. FastMCP validates
    # both signatures, including defaults, before and after execution.
    spec = TOOL_REGISTRY[name]
    signature = inspect.signature(function, eval_str=True)
    parameters = [parameter.replace(
        annotation=_parameter_annotation(spec.parameters["properties"][parameter.name]),
        default=(inspect.Parameter.empty if parameter.name in spec.parameters.get("required", [])
                 else parameter.default),
    ) for parameter in signature.parameters.values()]
    execute.__signature__ = signature.replace(parameters=parameters, return_annotation=dict[str, Any])
    execute.__annotations__ = {parameter.name: parameter.annotation for parameter in parameters}
    execute.__annotations__["return"] = dict[str, Any]
    return execute


def _parameter_annotation(schema: dict) -> Any:
    """Project registry constraints into the SDK's inferred input model."""
    kind = schema["type"]
    nullable = isinstance(kind, list) and "null" in kind
    if isinstance(kind, list):
        kind = next(item for item in kind if item != "null")
    annotation = {"string": str, "integer": int, "boolean": bool, "number": float}.get(kind)
    if kind == "array":
        annotation = list[_parameter_annotation(schema["items"])]
    constraints = {"strict": True}
    for source, target in (("minLength", "min_length"), ("maxLength", "max_length"),
                           ("minItems", "min_length"), ("maxItems", "max_length"),
                           ("minimum", "ge"), ("maximum", "le"), ("description", "description")):
        if source in schema:
            constraints[target] = schema[source]
    annotation = Annotated[annotation, Field(**constraints)]
    return annotation | None if nullable else annotation


def read_email_resource(email_id: str) -> str:
    """Return one full email as a JSON resource."""
    return _read_resource('email', lambda: get_email(email_id))


def read_email_corpus_stats_resource() -> str:
    """Return corpus-level email statistics as a JSON resource."""
    return _read_resource('corpus_stats', email_stats)


def _read_resource(target: str, read: Callable) -> str:
    started, request_id = time.perf_counter(), str(uuid.uuid4())
    status = 'error'
    try:
        result = json.dumps(read(), ensure_ascii=False)
        status = 'success'
        return result
    finally:
        _server_audit('resource_read', request_id=request_id, target=target, status=status, started=started)


def draft_reply_prompt(email_id: str = "", instruction: str = "") -> str:
    """Build a reusable prompt for drafting a reply to an email."""
    target = f"邮件 ID：{email_id}" if email_id else "请先根据用户描述定位目标邮件"
    requirement = instruction or "保持礼貌、简洁，并明确下一步行动"
    return f"请根据邮件内容起草一封中文回复。\n{target}\n回复要求：{requirement}"


def summarize_emails_prompt(query: str) -> str:
    """Build a reusable prompt for summarizing matching emails."""
    return (
        "请检索并总结与下列主题相关的邮件，输出：核心结论、关键事实、"
        f"涉及人员和待办事项。\n主题：{query}"
    )


def register_resources(server: Any) -> list[str]:
    """Register read-only MCP resources for email content and corpus stats."""
    resources = [
        ("email://{email_id}", read_email_resource),
        ("email-corpus://stats", read_email_corpus_stats_resource),
    ]
    for uri, fn in resources:
        server.resource(uri)(fn)
    return [uri for uri, _ in resources]


def _prompt_decorator(server: Any, name: str, description: str) -> Callable:
    try:
        return server.prompt(name=name, description=description)
    except TypeError:
        return server.prompt()


def register_prompts(server: Any) -> list[str]:
    """Register reusable MCP prompt helpers."""
    prompts = [
        (
            "draft_reply_prompt",
            "生成一段用于起草邮件回复的提示词。",
            draft_reply_prompt,
        ),
        (
            "summarize_emails_prompt",
            "生成一段用于总结相关邮件的提示词。",
            summarize_emails_prompt,
        ),
    ]
    for name, description, fn in prompts:
        _prompt_decorator(server, name, description)(fn)
    return [name for name, _, _ in prompts]


def build_server(
    server_factory: Callable[..., Any] | None = None,
    policy: ToolPolicy | None = None,
) -> Any:
    """Build a FastMCP server and register the email tools.

    ``server_factory`` is injectable so tests can verify registration without
    requiring the MCP SDK or opening a transport.
    """
    if not cfg.MCP_AUTH_TOKEN and cfg.MCP_HOST not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("MCP_AUTH_TOKEN is required when binding outside loopback")
    if server_factory is None:
        from mcp.server.fastmcp import FastMCP

        server_factory = FastMCP

    server = server_factory(
        "Email RAG Agent",
        json_response=True,
        host=cfg.MCP_HOST,
        port=cfg.MCP_PORT,
        **_auth_kwargs(),
    )
    register_tools(server, policy=policy)
    register_resources(server)
    register_prompts(server)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Email RAG Agent MCP server")
    parser.add_argument(
        "--transport",
        default="streamable-http",
        choices=["stdio", "streamable-http", "sse"],
        help="MCP transport to use",
    )
    args = parser.parse_args()

    server = build_server()
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
