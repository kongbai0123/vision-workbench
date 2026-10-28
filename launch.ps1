param([switch]$Diagnose, [switch]$Repair, [switch]$NoAutoRepair, [string]$DataRoot)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$taskRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
. (Join-Path $taskRoot 'scripts\runtime.ps1')
$taskLogs = Join-Path $taskRoot 'data\logs'
try {
    $taskData = if ($DataRoot) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DataRoot) } else { Join-Path $taskRoot 'data' }
    $taskLogs = Join-Path $taskData 'logs'
    $taskReportPath = Join-Path $taskLogs 'startup-check.json'
    if ($Diagnose -and $Repair) { throw 'Choose either -Diagnose (inspect only) or -Repair.' }
    New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
    if ($Repair) {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $taskRoot 'bootstrap.ps1')
        if ($LASTEXITCODE -ne 0) { throw 'Setup failed. Review the bootstrap log and rerun launch.ps1 -Repair.' }
    }
    $report = Get-WorkbenchRuntimeReport -ProjectRoot $taskRoot
    if (-not $report.ok -and -not $Diagnose -and -not $Repair -and -not $NoAutoRepair) {
        Write-Host 'Runtime is incomplete. Attempting setup once; this may download dependencies.'
        foreach ($issue in $report.issues) { Write-Host " - $issue" }
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $taskRoot 'bootstrap.ps1')
        if ($LASTEXITCODE -ne 0) { throw 'Automatic setup failed. Review the bootstrap log and rerun launch.ps1 -Repair.' }
        $report = Get-WorkbenchRuntimeReport -ProjectRoot $taskRoot
    }
    if ($Diagnose) {
        $markers = foreach ($name in @('vision-workbench.bat', 'bootstrap.ps1', 'launch.ps1', 'scripts\runtime.ps1')) {
            $file = Join-Path $taskRoot $name
            $zone = Get-Content -LiteralPath $file -Stream Zone.Identifier -ErrorAction SilentlyContinue
            [pscustomobject]@{ file = $name; internet_mark = [bool]($zone -match '^ZoneId=3$') }
        }
        $report | Add-Member -NotePropertyName download_markers -NotePropertyValue @($markers) -Force
    }
    $report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $taskReportPath -Encoding UTF8
    foreach ($issue in $report.issues) { Write-Host "ERROR: $issue" -ForegroundColor Red }
    foreach ($warning in $report.warnings) { Write-Warning $warning }
    Write-Host "Runtime report: $taskReportPath"
    if (-not $report.ok) { throw 'Runtime is not ready. Run launch.ps1 -Repair; see docs/startup-troubleshooting.md.' }
    Write-Host 'Runtime check passed.'
    if ($Diagnose -or $Repair) { exit 0 }
    $taskPython = Join-Path $taskRoot '.venv\Scripts\pythonw.exe'
    $taskStamp = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + $PID
    $taskStdout = Join-Path $taskLogs "startup-$taskStamp.stdout.log"
    $taskStderr = Join-Path $taskLogs "startup-$taskStamp.stderr.log"
    $taskArguments = @((Join-Path $taskRoot 'main.py'), '--data-root', $taskData) | ForEach-Object { ConvertTo-WorkbenchArgument -Value $_ }
    Write-Host 'Starting Vision Workbench. Startup errors are saved in the logs directory.'
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList ($taskArguments -join ' ') -WorkingDirectory $taskRoot -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru -Wait
    if ($taskProcess.ExitCode -ne 0) { throw "Application exited with code $($taskProcess.ExitCode). Details: $taskStderr" }
    exit 0
}
catch {
    Write-Host ("Vision Workbench could not start: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host "Logs: $taskLogs"
    Write-Host 'Diagnostics: powershell -NoProfile -ExecutionPolicy Bypass -File .\launch.ps1 -Diagnose'
    exit 1
}
