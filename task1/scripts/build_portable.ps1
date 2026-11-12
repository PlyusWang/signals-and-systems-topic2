param([string]$DistPath = (Join-Path $PSScriptRoot '..\release\portable'))
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $projectRoot '.venv-pack\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv-pack with the packaging dependencies before building.' }
Push-Location $projectRoot
$previousPath = $env:Path
$previousPythonPath = $env:PYTHONPATH
try {
    # Keep unrelated development DLL directories out of the portable bundle.
    $env:PYTHONPATH = ''
    $env:Path = "$(Split-Path -Parent $python);$env:SystemRoot\System32;$env:SystemRoot"
& $python -m PyInstaller --noconfirm --clean --distpath $DistPath --workpath (Join-Path $projectRoot '.codex\portable-build') (Join-Path $projectRoot 'task1\SignalScope.spec')
if ($LASTEXITCODE -ne 0) { throw 'Portable build failed.' }
$portableRoot = Join-Path $DistPath 'SignalScope'
$sampleRoot = Join-Path $portableRoot 'samples'
New-Item -ItemType Directory -Force -Path $sampleRoot | Out-Null
$audioRoot = Join-Path $projectRoot 'task2\audio'
foreach ($sample in @('clean.wav', 'noise_N1.wav', 'noise_N2.wav', 'clean_plus_N1_N2.wav', 'clean_plus_N1_N2_N3.wav', 'wavelet_soft_N1_N2.wav', 'psd_wiener_N1_N2.wav', 'message_demo_received_N1_N2.wav', 'message_demo_decoded_regenerated.wav')) {
    $source = Join-Path $audioRoot $sample
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination $sampleRoot -Force }
}
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot '..\release') | Out-Null
Compress-Archive -Path (Join-Path $portableRoot '*') -DestinationPath (Join-Path $PSScriptRoot '..\release\SignalScope-portable-win-x64.zip') -Force
} finally {
    $env:Path = $previousPath
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
