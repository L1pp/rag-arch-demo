# React/FastAPI default entry; credentials, accounts and corpus are operator supplied.
param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$webPython = Join-Path $PSScriptRoot '.venv-web\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $webPython)) {
    throw '请先创建 .venv-web 并安装 requirements.txt。'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend\dist\index.html'))) {
    throw '请先在 frontend 目录执行 npm ci 和 npm run build，或另用 Vite 开发服务器。'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $webPython -m uvicorn rag_arch.web.app:app --host 127.0.0.1 --port $Port --workers 1 `
        --proxy-headers --forwarded-allow-ips 127.0.0.1 --timeout-graceful-shutdown 5
} finally {
    Pop-Location
}
