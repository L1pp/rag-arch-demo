# The API only schedules durable tasks. Run scripts/start_worker.ps1 separately.
# API-only by default. Serve the independently built frontend separately.
param([int]$Port = 8000, [int]$Workers = 1)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$webPython = Join-Path $repoRoot '.venv-web\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $webPython)) {
    throw '请先创建 .venv-web 并安装 requirements/web.txt。'
}
if ($Workers -lt 1) {
    throw 'Workers必须为正整数。'
}
Push-Location -LiteralPath $repoRoot
try {
    & $webPython -m uvicorn rag_arch.web.app:app --host 127.0.0.1 --port $Port --workers $Workers `
        --proxy-headers --forwarded-allow-ips 127.0.0.1 --timeout-graceful-shutdown 5
} finally {
    Pop-Location
}
