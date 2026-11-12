$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create the environment first: py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt' }
Push-Location $projectRoot
try {
    & $python -m task2.communication_experiment.runner @args
    if ($LASTEXITCODE -ne 0) { throw 'Communication experiment failed.' }
} finally { Pop-Location }
