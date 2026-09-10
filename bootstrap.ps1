param([switch]$AI)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.13 -m venv .venv }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { & python -m venv .venv }
    else { throw 'Please install Python 3.13 for Windows and run bootstrap.ps1 again.' }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the independent Python environment.' }
}
& $taskPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; rerun to resume.' }
if ($AI) {
    & $taskPython -m pip install -r requirements-ai.txt
    if ($LASTEXITCODE -ne 0) { throw 'AI dependency installation failed.' }
    & $taskPython prepare_models.py
    if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed.' }
}
Write-Output 'Vision Workbench is ready. Open run.bat.'
