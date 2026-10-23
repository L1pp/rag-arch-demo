param(
    [int]$Concurrency = 3,
    [int]$MaxTasks = 0,
    [switch]$Simulation
)
$ErrorActionPreference = 'Stop'
$workerPython = Join-Path $PSScriptRoot '.venv-web\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $workerPython)) {
    throw '请先创建 .venv-web 并安装 requirements.txt。'
}
if ($Concurrency -lt 1) {
    throw 'Concurrency必须为正整数。'
}
if ($MaxTasks -lt 0) {
    throw 'MaxTasks不能为负数。'
}

Push-Location -LiteralPath $PSScriptRoot
try {
    $workerArgs = @('-m', 'rag_arch.web.worker_process', 'run',
                    '--concurrency', "$Concurrency")
    if ($MaxTasks -gt 0) {
        $workerArgs += @('--max-tasks', "$MaxTasks")
    }
    if ($Simulation) {
        $workerArgs += '--simulation'
    }
    & $workerPython @workerArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
