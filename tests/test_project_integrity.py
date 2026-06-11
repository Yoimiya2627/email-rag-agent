"""Repository-level checks for install, CI, and deployment hygiene."""

from __future__ import annotations

import tomllib
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_pyproject_declares_dependency_groups_and_tooling():
    pyproject = tomllib.loads(_read("pyproject.toml"))

    dependencies = "\n".join(pyproject["project"]["dependencies"])
    optional = pyproject["project"]["optional-dependencies"]

    assert "fastapi" in dependencies
    assert "sentence-transformers" in dependencies
    assert {"dev", "gmail", "eval"}.issubset(optional)
    assert "pytest" in "\n".join(optional["dev"])
    assert "ruff" in "\n".join(optional["dev"])
    assert pyproject["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    assert pyproject["tool"]["ruff"]["lint"]["select"]


def test_ci_workflow_runs_tests_compile_and_evalops_gates():
    workflow = _read(".github/workflows/ci.yml")

    assert "python -m compileall -q api agents core config frontend models scripts mcp_server.py" in workflow
    assert "python -m pytest tests/ -q" in workflow
    assert "scripts/check_agent_eval_gate.py" in workflow
    assert "--min-tasks 1" in workflow
    assert "--min-tasks 100" in workflow
    assert "workflow_dispatch" in workflow


def test_task_shortcuts_expose_smoke_and_full_eval_gates():
    makefile = _read("Makefile")
    tasks = _read("tasks.ps1")

    for command in (
        "agent-eval-smoke",
        "agent-eval-full",
        "agent-eval-real",
        "agent-eval-real-gate",
        "gmail-agent-testset",
        "gmail-gold-quality",
        "verify",
    ):
        assert command in makefile
        assert command in tasks


def test_powershell_test_task_keeps_pytest_basetemp_inside_workspace():
    tasks = _read("tasks.ps1")

    assert "--basetemp" in tasks
    assert ".pytest_tmp" in tasks


def test_real_context_recall_task_uses_explicit_100_case_window():
    makefile = _read("Makefile")
    tasks = _read("tasks.ps1")

    for content in (makefile, tasks):
        assert "--versions V2" in content
        assert "--top-n 10" in content
        assert "--fetch-k 80" in content


def test_ignore_files_exclude_local_browser_profiles_and_private_outputs():
    gitignore = _read(".gitignore")
    dockerignore = _read(".dockerignore")

    for pattern in (
        ".edge-tmp/",
        ".chrome_pdf_profile*/",
        "edge-netlog.json",
        "tmp_ascii/",
        "credentials/",
        "_private/",
        "*.log",
        "data/eval_results/*.real*.json",
        "data/eval_results/*.real*.md",
    ):
        assert pattern in gitignore
        assert pattern in dockerignore


def test_compose_frontend_uses_project_image_instead_of_runtime_pip_install():
    compose = _read("docker-compose.yml")

    assert "target: frontend" in compose
    assert "pip install streamlit requests" not in compose


def test_powershell_full_eval_gate_passes_current_full_result():
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        pytest.skip("PowerShell is not available")

    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "tasks.ps1"),
            "agent-eval-full",
        ],
        cwd=ROOT,
        encoding="utf-8",
        errors="replace",
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "result                : PASS" in result.stdout
