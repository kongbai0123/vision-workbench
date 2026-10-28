# Shared runtime discovery. Dot-sourcing this file does not change the machine.
function Test-WorkbenchPython {
    param([string]$Command, [string[]]$PrefixArguments = @(), [switch]$Recommended)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $versionRule = if ($Recommended) { 'sys.version_info[:2] == (3, 13)' } else { '(3, 12) <= sys.version_info[:2] < (3, 15)' }
        & $Command @PrefixArguments -B -c "import sys,struct; sys.exit(0 if ($versionRule) and struct.calcsize('P') == 8 else 1)" 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    }
    catch { return $false }
    finally { $ErrorActionPreference = $previousPreference }
}

function Invoke-WorkbenchNative {
    param([string]$Command, [string[]]$Arguments = @())
    $previousPreference = $ErrorActionPreference
    try {
        # Route native streams through the host so PS 5.1 transcripts capture them.
        $ErrorActionPreference = 'Continue'
        & $Command @Arguments 2>&1 | Out-Host
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousPreference }
    return $code
}

function ConvertTo-WorkbenchArgument {
    param([AllowEmptyString()][string]$Value)
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Find-WorkbenchPython {
    $candidates = @()
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { $candidates += [pscustomobject]@{ Command = $launcher.Source; Prefix = @('-3.13') } }
    foreach ($parent in @($env:LOCALAPPDATA, $env:ProgramFiles)) {
        if (-not $parent) { continue }
        $relative = if ($parent -eq $env:LOCALAPPDATA) { 'Programs\Python\Python313\python.exe' } else { 'Python313\python.exe' }
        $candidate = Join-Path $parent $relative
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $candidates += [pscustomobject]@{ Command = $candidate; Prefix = @() } }
    }
    foreach ($command in @(Get-Command python -All -ErrorAction SilentlyContinue)) {
        if ($command.Source -and $command.Source -notmatch '\\WindowsApps\\') {
            $candidates += [pscustomobject]@{ Command = $command.Source; Prefix = @() }
        }
    }
    foreach ($candidate in $candidates) {
        if (Test-WorkbenchPython -Command $candidate.Command -PrefixArguments $candidate.Prefix -Recommended) { return $candidate }
    }
    throw 'Python 3.13 (64-bit) was not found. Install it from python.org, then rerun bootstrap.ps1. A Python 3.11 environment cannot install this project''s pinned packages.'
}

function Initialize-WorkbenchEnvironment {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot, [string]$Name = '.venv')
    if ($Name -notin @('.venv', '.venv-training')) { throw 'Unexpected environment name.' }
    $root = [System.IO.Path]::GetFullPath($ProjectRoot)
    if ($root.Length -gt [System.IO.Path]::GetPathRoot($root).Length) { $root = $root.TrimEnd('\') }
    $environmentPath = Join-Path $root $Name
    $python = Join-Path $environmentPath 'Scripts\python.exe'
    if (Test-Path -LiteralPath $environmentPath) {
        $item = Get-Item -LiteralPath $environmentPath -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'The virtual environment is a link. Inspect it before replacing it.' }
    }
    if ((Test-Path -LiteralPath $python -PathType Leaf) -and (Test-WorkbenchPython -Command $python)) { return $python }
    $candidate = Find-WorkbenchPython
    if (Test-Path -LiteralPath $environmentPath) {
        $resolved = (Resolve-Path -LiteralPath $environmentPath).Path
        if ([System.IO.Path]::GetDirectoryName($resolved) -ine $root) { throw 'Refusing to move an environment outside this project.' }
        $backupName = $Name + '-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
        $backupPath = Join-Path $root $backupName
        if ([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($backupPath)) -ine $root) { throw 'Invalid environment backup path.' }
        Rename-Item -LiteralPath $resolved -NewName $backupName
        Write-Host "Preserved the old environment: $backupPath"
    }
    $code = Invoke-WorkbenchNative -Command $candidate.Command -Arguments (@($candidate.Prefix) + @('-m', 'venv', $environmentPath))
    if ($code -ne 0 -or -not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Could not create the independent Python environment. Rerun bootstrap.ps1 to retry.' }
    if (-not (Test-WorkbenchPython -Command $python)) { throw 'The new Python environment failed its version/architecture check.' }
    return $python
}

function Get-WorkbenchRuntimeReport {
    param([string]$ProjectRoot)
    $python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        return [pscustomobject]@{ ok = $false; issues = @('The .venv Python runtime is missing. Run bootstrap.ps1.'); warnings = @(); python = $null; packages = @() }
    }
    try {
        $raw = & $python -B (Join-Path $ProjectRoot 'scripts\check_runtime.py') --json
        $code = $LASTEXITCODE
        $report = ($raw -join "`n") | ConvertFrom-Json
        if ($null -eq $report.ok -or ($code -ne 0 -and $report.ok)) { throw 'Invalid runtime-check result.' }
        return $report
    }
    catch {
        return [pscustomobject]@{ ok = $false; issues = @('The Python runtime check could not finish: ' + $_.Exception.Message); warnings = @(); python = $null; packages = @() }
    }
}
