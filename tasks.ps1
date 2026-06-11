# Windows mirror of the Makefile. Usage: .\tasks.ps1 <command>
#
# If you hit "running scripts is disabled on this system", lift the policy for
# the current shell only:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

param(
    [Parameter(Position = 0)]
    [string]$Cmd = "help"
)

$ErrorActionPreference = "Stop"
# Prefer the project venv so deps installed there are found without needing to
# activate it first; $env:PYTHON still overrides, and bare "python" is the
# fallback when no .venv exists.
$Python = if ($env:PYTHON) {
    $env:PYTHON
} elseif (Test-Path ".venv\Scripts\python.exe") {
    (Resolve-Path ".venv\Scripts\python.exe").Path
} else {
    "python"
}

function Assert-LastCommandSucceeded {
    param([string]$Message)
    if ($LASTEXITCODE -ne 0) { throw $Message }
}

function Show-Help {
    Write-Host "Email RAG Agent - common tasks"
    Write-Host ""
    Write-Host "  .\tasks.ps1 install     pip install + preload bge-m3 (~5 min first time, ~570MB)"
    Write-Host "  .\tasks.ps1 install-lock pip install from requirements.lock"
    Write-Host "  .\tasks.ps1 index       index data/emails.json into ChromaDB"
    Write-Host "  .\tasks.ps1 run         start API on :8000 + Streamlit on :8501 (Ctrl-C stops both)"
    Write-Host "  .\tasks.ps1 mcp         start standalone MCP server on :8001"
    Write-Host "  .\tasks.ps1 api         API only"
    Write-Host "  .\tasks.ps1 ui          Streamlit only"
    Write-Host "  .\tasks.ps1 eval        run RAGAS on V2 only (~3 min, recommended config)"
    Write-Host "  .\tasks.ps1 agent-eval run agent task evaluation"
    Write-Host "  .\tasks.ps1 agent-eval-smoke run offline gate for current smoke eval results"
    Write-Host "  .\tasks.ps1 agent-eval-full  run strict 100+ task offline gate"
    Write-Host "  .\tasks.ps1 trace-summary summarize agent trace JSONL"
    Write-Host "  .\tasks.ps1 eval-all    run all 7 ablation versions (~30 min)"
    Write-Host "  .\tasks.ps1 latency     measure end-to-end latency"
    Write-Host "  .\tasks.ps1 reranker-latency measure reranker-only latency"
    Write-Host "  .\tasks.ps1 gmail-preflight check local readiness for real Gmail data"
    Write-Host "  .\tasks.ps1 gmail-sync  sync Gmail read-only messages to local ignored JSON"
    Write-Host "  .\tasks.ps1 gmail-sync-index sync Gmail read-only messages and rebuild index"
    Write-Host "  .\tasks.ps1 gmail-gold-template build real-mail gold annotation template"
    Write-Host "  .\tasks.ps1 gmail-gold-quality check real-mail gold label quality"
    Write-Host "  .\tasks.ps1 context-recall evaluate retrieval against gold chunk labels"
    Write-Host "  .\tasks.ps1 context-recall-real evaluate V2 top10 retrieval against real-mail gold labels"
    Write-Host "  .\tasks.ps1 test        run unit tests"
    Write-Host "  .\tasks.ps1 compile     compile Python modules"
    Write-Host "  .\tasks.ps1 verify      compile, test, and run smoke EvalOps gate"
    Write-Host "  .\tasks.ps1 clean       remove chroma_db and __pycache__"
}

function Invoke-Install {
    & $Python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    & $Python scripts/preload_model.py
    if ($LASTEXITCODE -ne 0) { throw "model preload failed" }
}

function Invoke-InstallLock {
    & $Python -m pip install -r requirements.lock
    Assert-LastCommandSucceeded "locked dependency install failed"
}

function Invoke-Index {
    & $Python scripts/index_emails.py
    Assert-LastCommandSucceeded "indexing failed"
}

function Invoke-Api {
    & $Python -m api.main
    Assert-LastCommandSucceeded "api exited with an error"
}

function Invoke-Ui {
    & $Python -m streamlit run frontend/app.py
    Assert-LastCommandSucceeded "ui exited with an error"
}

function Invoke-Run {
    Write-Host "Starting API (:8000) + Streamlit (:8501). Ctrl-C stops both."
    $procs = @()
    try {
        $procs += Start-Process -FilePath $Python -ArgumentList "-m", "api.main" `
            -PassThru -NoNewWindow
        $procs += Start-Process -FilePath $Python -ArgumentList "-m", "streamlit", "run", "frontend/app.py" `
            -PassThru -NoNewWindow

        while ($true) {
            foreach ($p in $procs) {
                if ($p.HasExited) { return }
            }
            Start-Sleep -Milliseconds 500
        }
    }
    finally {
        foreach ($p in $procs) {
            if ($p -and -not $p.HasExited) {
                try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
            }
        }
    }
}

function Invoke-Mcp {
    & $Python mcp_server.py --transport streamable-http
    Assert-LastCommandSucceeded "mcp server exited with an error"
}

function Invoke-Eval {
    & $Python scripts/run_ragas_eval.py --versions V2
    Assert-LastCommandSucceeded "RAG eval failed"
}

function Invoke-AgentEval {
    & $Python scripts/run_agent_eval.py
    Assert-LastCommandSucceeded "agent eval failed"
}

function Invoke-AgentEvalSmoke {
    & $Python scripts/check_agent_eval_gate.py `
        --input data/eval_results/agent_eval.json `
        --min-tasks 1 `
        --min-task-success-rate 0.80 `
        --min-tool-accuracy 0.80 `
        --max-forbidden-tool-violation-rate 0.01 `
        --max-max-steps-reached-rate 0.05
    Assert-LastCommandSucceeded "agent eval smoke gate failed"
}

function Invoke-AgentEvalFull {
    & $Python scripts/check_agent_eval_gate.py `
        --input data/eval_results/agent_eval.json `
        --min-tasks 100 `
        --min-task-success-rate 0.80 `
        --min-tool-accuracy 0.80 `
        --max-forbidden-tool-violation-rate 0.01 `
        --max-max-steps-reached-rate 0.05
    Assert-LastCommandSucceeded "agent eval full gate failed"
}

function Invoke-TraceSummary {
    & $Python scripts/summarize_agent_traces.py
    Assert-LastCommandSucceeded "trace summary failed"
}

function Invoke-EvalAll {
    & $Python scripts/run_ragas_eval.py
    Assert-LastCommandSucceeded "full RAG eval failed"
}

function Invoke-Latency {
    & $Python scripts/measure_latency.py
    Assert-LastCommandSucceeded "latency benchmark failed"
}

function Invoke-RerankerLatency {
    & $Python scripts/measure_reranker_latency.py
    Assert-LastCommandSucceeded "reranker latency benchmark failed"
}

function Invoke-GmailPreflight {
    & $Python scripts/gmail_phase2a_preflight.py
    Assert-LastCommandSucceeded "Gmail real-data preflight failed"
}

function Invoke-GmailSync {
    & $Python scripts/sync_gmail_readonly.py
    Assert-LastCommandSucceeded "Gmail sync failed"
}

function Invoke-GmailSyncIndex {
    & $Python scripts/sync_gmail_readonly.py --index --clear-index
    Assert-LastCommandSucceeded "Gmail sync + index failed"
}

function Invoke-GmailGoldTemplate {
    & $Python scripts/build_real_gmail_gold_template.py
    Assert-LastCommandSucceeded "real Gmail gold template generation failed"
}

function Invoke-GmailGoldQuality {
    & $Python scripts/check_real_gold_quality.py
    Assert-LastCommandSucceeded "real Gmail gold quality gate failed"
}

function Invoke-ContextRecall {
    & $Python scripts/evaluate_context_recall.py
    Assert-LastCommandSucceeded "context recall evaluation failed"
}

function Invoke-ContextRecallReal {
    Invoke-GmailGoldQuality
    & $Python scripts/evaluate_context_recall.py `
        --gold data/real_emails/gold_chunks.real.json `
        --output data/eval_results/context_recall.real.json `
        --versions V2 `
        --top-n 10 `
        --fetch-k 80
    Assert-LastCommandSucceeded "real context recall evaluation failed"
}

function Invoke-Test {
    & $Python -m pytest tests/ -v --basetemp .pytest_tmp
    Assert-LastCommandSucceeded "unit tests failed"
}

function Invoke-Compile {
    & $Python -m compileall -q api agents core config frontend models scripts mcp_server.py
    Assert-LastCommandSucceeded "compile verification failed"
}

function Invoke-Verify {
    Invoke-Compile
    Invoke-Test
    Invoke-AgentEvalSmoke
}

function Invoke-Clean {
    if (Test-Path "chroma_db") {
        Remove-Item -Recurse -Force "chroma_db"
        Write-Host "Removed chroma_db/"
    }
    $pyCaches = Get-ChildItem -Path . -Include "__pycache__", ".pytest_cache" `
        -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "\\\.venv\\" }
    foreach ($d in $pyCaches) {
        Remove-Item -Recurse -Force $d.FullName -ErrorAction SilentlyContinue
    }
    Write-Host "Cleanup done."
}

switch ($Cmd.ToLower()) {
    "help"     { Show-Help }
    "install"  { Invoke-Install }
    "install-lock" { Invoke-InstallLock }
    "index"    { Invoke-Index }
    "api"      { Invoke-Api }
    "ui"       { Invoke-Ui }
    "run"      { Invoke-Run }
    "mcp"      { Invoke-Mcp }
    "eval"     { Invoke-Eval }
    "agent-eval" { Invoke-AgentEval }
    "agent-eval-smoke" { Invoke-AgentEvalSmoke }
    "agent-eval-full" { Invoke-AgentEvalFull }
    "trace-summary" { Invoke-TraceSummary }
    "eval-all" { Invoke-EvalAll }
    "latency"  { Invoke-Latency }
    "reranker-latency" { Invoke-RerankerLatency }
    "gmail-preflight" { Invoke-GmailPreflight }
    "gmail-sync" { Invoke-GmailSync }
    "gmail-sync-index" { Invoke-GmailSyncIndex }
    "gmail-gold-template" { Invoke-GmailGoldTemplate }
    "gmail-gold-quality" { Invoke-GmailGoldQuality }
    "context-recall" { Invoke-ContextRecall }
    "context-recall-real" { Invoke-ContextRecallReal }
    "test"     { Invoke-Test }
    "compile"  { Invoke-Compile }
    "verify"   { Invoke-Verify }
    "clean"    { Invoke-Clean }
    default {
        Write-Host "Unknown command: $Cmd" -ForegroundColor Red
        Write-Host ""
        Show-Help
        exit 1
    }
}
