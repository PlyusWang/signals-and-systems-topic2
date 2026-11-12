$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$archivePath = Join-Path $projectRoot 'documents\topic2_review_bundle.zip'
$temporaryPath = Join-Path $projectRoot '.codex\review-bundle-building.zip'
$files = [System.Collections.Generic.SortedDictionary[string,string]]::new()

function Add-ProjectFile([string]$relativePath) {
    $source = Join-Path $projectRoot $relativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required review file is missing: $relativePath"
    }
    $files[$relativePath.Replace('\', '/')] = $source
}

function Add-DirectoryFiles([string]$relativeDirectory, [string[]]$extensions) {
    Get-ChildItem -LiteralPath (Join-Path $projectRoot $relativeDirectory) -Recurse -File |
        Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() -and $_.FullName -notmatch '[\\/]__pycache__[\\/]' } |
        ForEach-Object { Add-ProjectFile $_.FullName.Substring($projectRoot.Length + 1) }
}

foreach ($file in @(
    '.gitignore', 'README.md', 'requirements.txt', 'conftest.py',
    'task1\__init__.py', 'task1\SignalScope.spec', 'task1\requirements-pack.txt', 'task2\__init__.py',
    'documents\REVIEW_README.md',
    'documents\report\topic2_report.tex', 'documents\report\topic2_report.pdf', 'documents\report\references.bib',
    'documents\presentation\topic2_presentation.pptx', 'documents\presentation\topic2_presentation.pdf',
    'task2\audio\clean_plus_N1_N2.wav', 'task2\audio\clean.wav', 'task2\audio\psd_wiener_N1_N2.wav'
)) { Add-ProjectFile $file }

foreach ($directory in @('task1\analyzer', 'task1\tests', 'task2\communication_experiment', 'task2\tests', 'shared')) {
    Add-DirectoryFiles $directory @('.py')
}
foreach ($directory in @('task1\scripts', 'task2\scripts', 'documents\scripts', 'documents\presentation')) {
    Add-DirectoryFiles $directory @('.py', '.ps1')
}
Add-DirectoryFiles 'task1\screenshots' @('.png')
Add-DirectoryFiles 'documents\report\images' @('.png')
Add-DirectoryFiles 'task2\figures' @('.png', '.pdf')
Add-DirectoryFiles 'task2\experiments' @('.csv', '.json')

New-Item -ItemType Directory -Path (Split-Path -Parent $temporaryPath) -Force | Out-Null
$archive = $null
try {
    $archive = [System.IO.Compression.ZipFile]::Open($temporaryPath, [System.IO.Compression.ZipArchiveMode]::Create)
    foreach ($file in $files.GetEnumerator()) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive, $file.Value, $file.Key, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
    $archive.Dispose()
    $archive = $null
    Move-Item -LiteralPath $temporaryPath -Destination $archivePath -Force
    Write-Output "Review bundle: $archivePath"
    Write-Output "Files: $($files.Count); size: $([math]::Round((Get-Item -LiteralPath $archivePath).Length / 1MB, 2)) MiB"
} finally {
    if ($null -ne $archive) { $archive.Dispose() }
    if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) { Remove-Item -LiteralPath $temporaryPath }
}
