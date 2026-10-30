# Windows mirror of the Makefile. Usage: .\tasks.ps1 <command>
#
# If you hit "running scripts is disabled on this system", lift the policy for
# the current shell only:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

param(
    [Parameter(Position = 0)]
    [string]$Cmd = "help",
    [switch]$Apply,
    [switch]$IncludeIndex
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$script:TaskExitCode = 0
Push-Location -LiteralPath $ProjectRoot
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

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArguments)
    & $Python @PythonArguments
    $script:TaskExitCode = $LASTEXITCODE
    if ($script:TaskExitCode -ne 0) {
        throw "Python task failed with exit code $script:TaskExitCode"
    }
}

function Assert-ProjectTarget {
    param([string]$Target)
    $resolved = [IO.Path]::GetFullPath($Target)
    $prefix = $ProjectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target must be inside the project directory"
    }
    $node = Get-Item -LiteralPath $resolved -Force
    while ($node -and $node.FullName -ne $ProjectRoot) {
        if ($node.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Cleanup refuses linked directories"
        }
        $node = $node.Parent
    }
    return $resolved
}

function Show-Help {
    Write-Host "Email RAG Agent - common tasks"
    Write-Host ""
    Write-Host "  .\tasks.ps1 install     install dependencies only"
    Write-Host "  .\tasks.ps1 preload     explicitly prepare/download configured embedding model"
    Write-Host "  .\tasks.ps1 doctor      inspect configuration and installed libraries without loading models"
    Write-Host "  .\tasks.ps1 benchmark-check inspect local indexing benchmark readiness without downloads"
    Write-Host "  .\tasks.ps1 benchmark-index run the explicit local indexing benchmark"
    Write-Host "  .\tasks.ps1 index       index data/emails.json into ChromaDB"
    Write-Host "  .\tasks.ps1 run         start API on :8000 + Streamlit on :8501 (Ctrl-C stops both)"
    Write-Host "  .\tasks.ps1 mcp         start standalone MCP server on :8001"
    Write-Host "  .\tasks.ps1 api         API only"
    Write-Host "  .\tasks.ps1 ui          Streamlit only"
    Write-Host "  .\tasks.ps1 eval        run RAGAS on V2 only"
    Write-Host "  .\tasks.ps1 agent-eval run agent task evaluation"
    Write-Host "  .\tasks.ps1 trace-summary summarize agent trace JSONL"
    Write-Host "  .\tasks.ps1 eval-all    run all configured ablation versions"
    Write-Host "  .\tasks.ps1 latency     measure end-to-end latency"
    Write-Host "  .\tasks.ps1 reranker-latency measure reranker-only latency"
    Write-Host "  .\tasks.ps1 gmail-sync  sync Gmail read-only messages to local ignored JSON"
    Write-Host "  .\tasks.ps1 context-recall evaluate retrieval against gold chunk labels"
    Write-Host "  .\tasks.ps1 test        run unit tests"
    Write-Host "  .\tasks.ps1 clean       preview cache cleanup; -Apply deletes caches; -IncludeIndex also selects chroma_db"
}

function Invoke-Install {
    Invoke-Python -m pip install -r requirements.txt
}

function Invoke-Index {
    Invoke-Python scripts/index_emails.py
}

function Invoke-Api {
    Invoke-Python -m api.main
}

function Invoke-Ui {
    Invoke-Python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
}

function Invoke-Run {
    Write-Host "Starting API (:8000) + Streamlit (:8501). Ctrl-C stops both."
    $procs = @()
    try {
        $procs += Start-Process -FilePath $Python -ArgumentList "-m", "api.main" `
            -PassThru -WindowStyle Hidden -WorkingDirectory $ProjectRoot
        $procs += Start-Process -FilePath $Python -ArgumentList "-m", "streamlit", "run", "frontend/app.py", "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true" `
            -PassThru -WindowStyle Hidden -WorkingDirectory $ProjectRoot

        while ($true) {
            foreach ($p in $procs) {
                if ($p.HasExited) {
                    $script:TaskExitCode = $p.ExitCode
                    return
                }
            }
            Start-Sleep -Milliseconds 500
        }
    }
    finally {
        foreach ($p in $procs) {
            if ($p -and -not $p.HasExited) {
                # Windows venv python.exe may be a launcher whose child owns the
                # server. Stop the tree rooted at the process this task started.
                try { & "$env:SystemRoot\System32\taskkill.exe" /PID $p.Id /T /F *> $null } catch {}
            }
        }
    }
}

function Invoke-Mcp {
    Invoke-Python mcp_server.py --transport streamable-http
}

function Invoke-Eval {
    Invoke-Python scripts/run_ragas_eval.py --versions V2
}

function Invoke-AgentEval {
    Invoke-Python scripts/run_agent_eval.py
}

function Invoke-TraceSummary {
    Invoke-Python scripts/summarize_agent_traces.py
}

function Invoke-EvalAll {
    Invoke-Python scripts/run_ragas_eval.py
}

function Invoke-Latency {
    Invoke-Python scripts/measure_latency.py
}

function Invoke-RerankerLatency {
    Invoke-Python scripts/measure_reranker_latency.py
}

function Invoke-GmailSync {
    Invoke-Python scripts/sync_gmail_readonly.py
}

function Invoke-ContextRecall {
    Invoke-Python scripts/evaluate_context_recall.py
}

function Invoke-Test {
    Invoke-Python scripts/offline_tests.py tests/ -v
}

function Invoke-Clean {
    $indexPath = Join-Path $ProjectRoot 'chroma_db'
    if ($IncludeIndex -and (Test-Path -LiteralPath $indexPath)) {
        $indexTarget = Assert-ProjectTarget $indexPath
        if ($Apply) { Remove-Item -LiteralPath $indexTarget -Recurse -Force }
        Write-Host "Selected index directory: $indexTarget (apply=$Apply)"
    }
    $pyCaches = Get-ChildItem -Path . -Include "__pycache__", ".pytest_cache" `
        -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "\\\.venv\\" }
    foreach ($d in ($pyCaches | Sort-Object { $_.FullName.Length } -Descending)) {
        if (-not (Test-Path -LiteralPath $d.FullName)) { continue }
        $cacheTarget = Assert-ProjectTarget $d.FullName
        if ($Apply) { Remove-Item -LiteralPath $cacheTarget -Recurse -Force }
        Write-Host "Selected cache: $cacheTarget (apply=$Apply)"
    }
    Write-Host "Cleanup done."
}

try {
switch ($Cmd.ToLower()) {
    "help"     { Show-Help }
    "install"  { Invoke-Install }
    "preload"  { Invoke-Python scripts/preload_model.py }
    "doctor"   { Invoke-Python scripts/doctor.py }
    "index"    { Invoke-Index }
    "benchmark-check" { Invoke-Python scripts/benchmark_index.py --check --input data/emails.json --max-emails 10000 }
    "benchmark-index" { Invoke-Python scripts/benchmark_index.py --run --input data/emails.json --max-emails 10000 }
    "api"      { Invoke-Api }
    "ui"       { Invoke-Ui }
    "run"      { Invoke-Run }
    "mcp"      { Invoke-Mcp }
    "eval"     { Invoke-Eval }
    "agent-eval" { Invoke-AgentEval }
    "trace-summary" { Invoke-TraceSummary }
    "eval-all" { Invoke-EvalAll }
    "latency"  { Invoke-Latency }
    "reranker-latency" { Invoke-RerankerLatency }
    "gmail-sync" { Invoke-GmailSync }
    "context-recall" { Invoke-ContextRecall }
    "test"     { Invoke-Test }
    "clean"    { Invoke-Clean }
    default {
        Write-Host "Unknown command: $Cmd" -ForegroundColor Red
        Write-Host ""
        Show-Help
        $script:TaskExitCode = 1
    }
}
} catch {
    if ($script:TaskExitCode -eq 0) { $script:TaskExitCode = 1 }
    Write-Error $_ -ErrorAction Continue
} finally {
    Pop-Location
}
exit $script:TaskExitCode
