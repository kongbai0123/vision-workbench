param([switch]$AI,[switch]$Training)
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
if ($Training) {
    $trainingPython = Join-Path $PSScriptRoot '.venv-training\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $trainingPython)) {
        if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.13 -m venv .venv-training }
        elseif (Get-Command python -ErrorAction SilentlyContinue) { & python -m venv .venv-training }
        else { throw 'Please install Python 3.13 for the independent training runtime.' }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the independent training environment.' }
    }
    & $trainingPython -m pip install -r requirements-training.txt
    if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed; rerun to resume.' }
    & $trainingPython -c "import torch, torchvision; print('Mask R-CNN runtime ready:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw 'Training runtime verification failed.' }
}
Write-Output 'Vision Workbench is ready. Open vision-workbench.bat.'
