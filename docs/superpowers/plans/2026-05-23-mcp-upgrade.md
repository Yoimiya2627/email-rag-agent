# MCP Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing function-calling email agent into an MCP-ready architecture without breaking the current `/chat/agent` path.

**Architecture:** Keep the existing local agent loop stable while introducing a shared tool registry. The local DeepSeek `tools` schema and the MCP server both derive from that registry, so tool definitions stay single-sourced.

**Tech Stack:** Python, FastAPI, Streamlit, OpenAI-compatible function calling, MCP Python SDK/FastMCP, pytest.

---

### Task 1: Phase 0 Baseline And Tool Registry

**Files:**
- Create: `agents/tool_registry.py`
- Modify: `agents/tools.py`
- Test: `tests/test_tool_registry.py`

- [ ] Write tests proving registry tool names match existing dispatch and schemas.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_tool_registry.py -q` and confirm the new tests fail because `agents.tool_registry` does not exist yet.
- [ ] Implement a minimal registry with `ToolSpec`, `TOOL_REGISTRY`, `openai_tool_schemas()`, and `tool_dispatch()`.
- [ ] Refactor `agents/tools.py` so `TOOL_SCHEMAS` and `TOOL_DISPATCH` come from the registry while existing tool functions and `call_tool()` behavior remain unchanged.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_tool_registry.py tests/test_tools.py tests/test_agent_loop.py -q`.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/ -q` before moving to Phase 1.

### Task 2: Phase 1 MCP Server

**Files:**
- Create: `mcp_server.py`
- Modify: `requirements.txt`
- Test: `tests/test_mcp_server.py`

- [ ] Write tests proving the MCP server module can register every registry tool with a fake FastMCP object.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_mcp_server.py -q` and confirm failure because `mcp_server.py` does not exist yet.
- [ ] Add `mcp>=1.0.0` to `requirements.txt`.
- [ ] Implement `mcp_server.py` with `build_server()`, `register_tools(server)`, and a `__main__` entrypoint.
- [ ] Keep imports lazy so the unit tests can run without a network call or real MCP client.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_mcp_server.py tests/test_tool_registry.py tests/test_tools.py tests/test_agent_loop.py -q`.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/ -q`.

### Task 3: Phase 2 MCP Resources And Prompts

**Files:**
- Modify: `mcp_server.py`
- Test: `tests/test_mcp_server.py`

- [ ] Add tests for resource registration: `email://{email_id}` and `email-corpus://stats`.
- [ ] Add tests for prompt registration: draft reply and summary prompt helpers.
- [ ] Implement resources and prompts by wrapping the existing tool functions.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_mcp_server.py tests/test_tools.py -q`.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/ -q`.

### Task 4: Phase 3 MCP Client Adapter

**Files:**
- Create: `agents/mcp_adapter.py`
- Modify: `agents/agent_loop.py`
- Modify: `config/settings.py`
- Test: `tests/test_mcp_adapter.py`
- Test: `tests/test_agent_loop.py`

- [ ] Add `AGENT_TOOL_BACKEND=local|mcp` configuration with default `local`.
- [ ] Write tests for converting MCP `tools/list` entries into OpenAI-compatible `tools` schemas.
- [ ] Write tests for calling a selected MCP tool and preserving the current `{"error": ...}` failure style.
- [ ] Refactor `run_agent_loop()` to get schemas and execute calls through a backend object while defaulting to the current local backend.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_mcp_adapter.py tests/test_agent_loop.py -q`.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/ -q`.

### Task 5: Phase 4 Docs And Operational Hardening

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `.env.example`
- Test: `.\.venv\Scripts\python.exe -m pytest tests/ -q`

- [ ] Document `mcp_server.py`, `AGENT_TOOL_BACKEND`, and local MCP startup commands.
- [ ] Update architecture docs to show function-calling local backend and MCP backend side by side.
- [ ] Keep the resume wording honest: MCP server is only listed as implemented after Phase 1; MCP-backed agent loop is only listed after Phase 3.
- [ ] Run full tests after doc/config changes to catch accidental import or formatting regressions.
