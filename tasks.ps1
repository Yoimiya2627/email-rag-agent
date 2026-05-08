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
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Show-Help {
    Write-Host "Email RAG Agent - common tasks"
    Write-Host ""
    Write-Host "  .\tasks.ps1 install     pip install + preload bge-m3 (~5 min first time, ~570MB)"
    Write-Host "  .\tasks.ps1 index       index data/emails.json into ChromaDB"
    Write-Host "  .\tasks.ps1 run         start API on :8000 + Streamlit on :8501 (Ctrl-C stops both)"
    Write-Host "  .\tasks.ps1 api         API only"
    Write-Host "  .\tasks.ps1 ui          Streamlit only"
    Write-Host "  .\tasks.ps1 eval        run RAGAS on V2 only (~3 min, recommended config)"
    Write-Host "  .\tasks.ps1 eval-all    run all 6 ablation versions (~30 min)"
    Write-Host "  .\tasks.ps1 latency     measure end-to-end latency"
    Write-Host "  .\tasks.ps1 test        run unit tests"
    Write-Host "  .\tasks.ps1 clean       remove chroma_db and __pycache__"
}

function Invoke-Install {
    & $Python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    & $Python scripts/preload_model.py
    if ($LASTEXITCODE -ne 0) { throw "model preload failed" }
}

function Invoke-Index {
    & $Python scripts/index_emails.py
}

function Invoke-Api {
    & $Python -m api.main
}

function Invoke-Ui {
    & $Python -m streamlit run frontend/app.py
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

function Invoke-Eval {
    & $Python scripts/run_ragas_eval.py --versions V2
}

function Invoke-EvalAll {
    & $Python scripts/run_ragas_eval.py
}

function Invoke-Latency {
    & $Python scripts/measure_latency.py
}

function Invoke-Test {
    & $Python -m pytest tests/ -v
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
    "index"    { Invoke-Index }
    "api"      { Invoke-Api }
    "ui"       { Invoke-Ui }
    "run"      { Invoke-Run }
    "eval"     { Invoke-Eval }
    "eval-all" { Invoke-EvalAll }
    "latency"  { Invoke-Latency }
    "test"     { Invoke-Test }
    "clean"    { Invoke-Clean }
    default {
        Write-Host "Unknown command: $Cmd" -ForegroundColor Red
        Write-Host ""
        Show-Help
        exit 1
    }
}
