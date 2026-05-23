"""MCP server exposing the Email RAG Agent tool layer.

This module is deliberately thin: tool metadata lives in
``agents.tool_registry`` and implementations remain in ``agents.tools``.  The
server registers those tools with FastMCP so external MCP hosts can discover
and call the same capabilities that `/chat/agent` currently uses locally.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import config.settings as cfg
from agents.tool_policy import ToolPolicy
from agents.tool_registry import TOOL_REGISTRY, ToolSpec, tool_dispatch
from agents.tools import email_stats, get_email


class StaticBearerTokenVerifier:
    """Minimal bearer-token verifier for local/portfolio MCP deployments."""

    def __init__(self, token: str, client_id: str = "email-rag-agent"):
        self.token = token
        self.client_id = client_id

    async def verify_token(self, token: str):
        if token != self.token:
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
        _tool_decorator(server, spec)(fn)
        registered.append(name)
    return registered


def read_email_resource(email_id: str) -> str:
    """Return one full email as a JSON resource."""
    return json.dumps(get_email(email_id), ensure_ascii=False)


def read_email_corpus_stats_resource() -> str:
    """Return corpus-level email statistics as a JSON resource."""
    return json.dumps(email_stats(), ensure_ascii=False)


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
