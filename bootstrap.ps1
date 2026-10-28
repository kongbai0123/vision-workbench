param([switch]$AI, [switch]$Training, [switch]$TrainingOnly)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$taskRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
Set-Location -LiteralPath $taskRoot
. (Join-Path $taskRoot 'scripts\runtime.ps1')
if ($TrainingOnly) { $Training = $true }
if ($TrainingOnly -and $AI) { throw 'TrainingOnly cannot be combined with AI.' }
$taskLogs = Join-Path $taskRoot 'data\logs'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskLog = Join-Path $taskLogs ('bootstrap-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + $PID + '.log')
Start-Transcript -LiteralPath $taskLog -Append | Out-Null
$taskSetupLock = $null
try {
    try { $taskSetupLock = [System.IO.File]::Open((Join-Path $taskLogs 'setup.lock'), [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None) }
    catch [System.IO.IOException] { throw 'Another setup is running or setup.lock is inaccessible. Wait for setup to finish, then check folder permissions and retry.' }
    if (-not $TrainingOnly) {
        $taskPython = Initialize-WorkbenchEnvironment -ProjectRoot $taskRoot
        Write-Host "Installing desktop dependencies with $taskPython"
        $taskCode = Invoke-WorkbenchNative -Command $taskPython -Arguments @('-m', 'pip', 'install', '--retries', '2', '--timeout', '30', '-r', (Join-Path $taskRoot 'requirements.txt'))
        if ($taskCode -ne 0) { throw 'Dependency installation failed. Check the log for SSL, network, or package errors; rerun bootstrap.ps1 to resume.' }
        if ($AI) {
            $taskCode = Invoke-WorkbenchNative -Command $taskPython -Arguments @('-m', 'pip', 'install', '--retries', '2', '--timeout', '30', '-r', (Join-Path $taskRoot 'requirements-ai.txt'))
            if ($taskCode -ne 0) { throw 'AI dependency installation failed.' }
            $taskCode = Invoke-WorkbenchNative -Command $taskPython -Arguments @((Join-Path $taskRoot 'prepare_models.py'))
            if ($taskCode -ne 0) { throw 'Model preparation failed.' }
        }
        $taskCode = Invoke-WorkbenchNative -Command $taskPython -Arguments @('-B', (Join-Path $taskRoot 'scripts\check_runtime.py'), '--report', (Join-Path $taskLogs 'startup-check.json'))
        if ($taskCode -ne 0) { throw 'The desktop runtime failed verification. See startup-check.json for the exact cause.' }
    }
    if ($Training) {
        $trainingPython = Initialize-WorkbenchEnvironment -ProjectRoot $taskRoot -Name '.venv-training'
        $taskCode = Invoke-WorkbenchNative -Command $trainingPython -Arguments @('-m', 'pip', 'install', '--retries', '2', '--timeout', '30', '-r', (Join-Path $taskRoot 'requirements-training.txt'))
        if ($taskCode -ne 0) { throw 'Training dependency installation failed; rerun to resume.' }
        $taskCode = Invoke-WorkbenchNative -Command $trainingPython -Arguments @('-c', "import torch, torchvision; print('Mask R-CNN runtime ready:', torch.__version__, 'CUDA:', torch.cuda.is_available())")
        if ($taskCode -ne 0) { throw 'Training runtime verification failed.' }
    }
    Write-Host 'Vision Workbench is ready. Open vision-workbench.bat.'
}
catch {
    Write-Host ("Setup failed: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host "Diagnostic log: $taskLog"
    exit 1
}
finally {
    if ($taskSetupLock) { $taskSetupLock.Dispose() }
    Stop-Transcript | Out-Null
}
