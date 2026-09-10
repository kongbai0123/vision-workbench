# Explicit-click prerequisite installer for Local Annotation Hub. Probe is read-only.
# Sources: Microsoft WSL install/basic-commands; Docker Windows install/release notes.
[CmdletBinding()]
param([ValidateSet('Probe', 'Setup', 'Elevated')][string]$Mode = 'Probe')
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$ProgressPreference = 'SilentlyContinue'
$taskRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$taskFolder = Join-Path $taskRoot 'data\cvat'
$taskState = Join-Path $taskFolder 'installer-state.json'
$taskLog = Join-Path $taskFolder 'prerequisites.log'
$dockerVersion = '4.90.0'
$dockerUrl = 'https://desktop.docker.com/win/main/amd64/238679/Docker%20Desktop%20Installer.exe'
$dockerSha256 = '2ecc54255702ffbf2e2779cb35ebecde535a0b31e4223171be2195dc318ebd3d'

function Get-Probe {
    $osInfo = Get-CimInstance Win32_OperatingSystem
    $systemInfo = Get-CimInstance Win32_ComputerSystem
    $cpuInfo = Get-CimInstance Win32_Processor | Select-Object -First 1
    $arch = [Environment]::GetEnvironmentVariable('PROCESSOR_ARCHITEW6432')
    if (-not $arch) { $arch = [Environment]::GetEnvironmentVariable('PROCESSOR_ARCHITECTURE') }
    $build = [int]$osInfo.BuildNumber
    $supported = $arch -eq 'AMD64' -and [int]$osInfo.ProductType -eq 1 -and ($build -eq 19045 -or $build -ge 22631)
    $reason = ''
    if (-not $supported) { $reason = '需要 Windows 10 22H2（19045）或 Windows 11 23H2（22631）以上的 x64 桌面系統。' }
    if (-not $systemInfo.HypervisorPresent -and $cpuInfo.VirtualizationFirmwareEnabled -eq $false) {
        $supported = $false
        $reason = '硬體虛擬化尚未啟用，請先於 BIOS／UEFI 啟用 Intel VT-x 或 AMD-V，再重新檢查。'
    }
    if ([double]$systemInfo.TotalPhysicalMemory -lt 7.5GB) {
        $supported = $false
        $reason = 'Docker Desktop 至少需要 8 GB 記憶體；目前記憶體不足。'
    }
    $wslVersion = ''
    if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
        $oldErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $wslOutput = (& wsl.exe --version 2>&1 | Out-String).Replace([string][char]0, '')
        $ErrorActionPreference = $oldErrorPreference
        if ($LASTEXITCODE -eq 0 -and $wslOutput -match '(\d+\.\d+\.\d+(?:\.\d+)?)') { $wslVersion = $Matches[1] }
    }
    $vmFeature = Get-CimInstance Win32_OptionalFeature -Filter "Name='VirtualMachinePlatform'" -ErrorAction SilentlyContinue
    $featuresReady = $null -ne $vmFeature -and [int]$vmFeature.InstallState -eq 1
    $dockerInstalled = (Test-Path -LiteralPath (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe')) -or
                       (Test-Path -LiteralPath (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'))
    return @{ supported=$supported; reason=$reason; build=$build; architecture=$arch;
              ram_gb=[math]::Round($systemInfo.TotalPhysicalMemory / 1GB, 1);
              virtualization=([bool]$systemInfo.HypervisorPresent -or [bool]$cpuInfo.VirtualizationFirmwareEnabled);
              wsl_version=$wslVersion; wsl_ready=($wslVersion -and [version]$wslVersion -ge [version]'2.1.5');
              features_ready=$featuresReady; docker_installed=$dockerInstalled; docker_target_version=$dockerVersion;
              boot_id=$osInfo.LastBootUpTime.ToUniversalTime().ToString('o') }
}

function Set-Stage([string]$Phase, [string]$Text, [string]$Step = 'wsl') {
    if (-not (Test-Path -LiteralPath $taskFolder)) { New-Item -ItemType Directory -Path $taskFolder -Force | Out-Null }
    $boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')
    $stateValue = @{ phase=$Phase; text=$Text; step=$Step; boot_id=$boot; pid=$PID;
                     updated_at=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0 }
    $stateTemp = $taskState + '.' + $PID + '.tmp'
    [IO.File]::WriteAllText($stateTemp, ($stateValue | ConvertTo-Json -Compress), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $stateTemp -Destination $taskState -Force
    # Only predefined progress messages are recorded; never arguments or credentials.
    [IO.File]::AppendAllText($taskLog, ([DateTime]::UtcNow.ToString('o') + ' ' + $Phase + ' ' + $Text + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
}

function Test-RebootPending {
    return (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending') -or
           (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired')
}

try {
    if ($Mode -eq 'Probe') { Get-Probe | ConvertTo-Json -Compress; exit 0 }
    $probe = Get-Probe
    if (-not $probe.supported) { Set-Stage 'blocked' $probe.reason 'system'; exit 0 }
    if (Test-RebootPending) {
        Set-Stage 'reboot_required' 'Windows 有待完成的重新啟動；請儲存工作並自行重新啟動，再回到中心按繼續準備。'
        exit 0
    }
    if ($Mode -eq 'Elevated') {
        $adminIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $adminPrincipal = [Security.Principal.WindowsPrincipal]::new($adminIdentity)
        if (-not $adminPrincipal.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) { throw 'Elevation required' }
        Set-Stage 'wsl' '正在啟用 Windows 虛擬化與 WSL 2；不會自動重新啟動。'
        $rebootNeeded = $false
        foreach ($featureName in @('VirtualMachinePlatform', 'Microsoft-Windows-Subsystem-Linux')) {
            $feature = Get-WindowsOptionalFeature -Online -FeatureName $featureName
            if ($feature.State -eq 'EnablePending' -or $feature.State -eq 'DisablePending') { $rebootNeeded = $true; continue }
            if ($feature.State -ne 'Enabled') {
                $result = Enable-WindowsOptionalFeature -Online -FeatureName $featureName -All -NoRestart
                if ($result.RestartNeeded) { $rebootNeeded = $true }
            }
        }
        if ($rebootNeeded) {
            Set-Stage 'reboot_required' 'Windows 功能已啟用，需要重新啟動；請儲存工作並自行重新啟動，再按繼續準備。'
            exit 0
        }
        if (-not $probe.wsl_ready) {
            $oldErrorPreference = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            if ($probe.wsl_version) { & wsl.exe --update --web-download 2>&1 | Out-Null }
            else { & wsl.exe --install --no-distribution --web-download 2>&1 | Out-Null }
            $wslCode = $LASTEXITCODE
            $ErrorActionPreference = $oldErrorPreference
            if ($wslCode -eq 3010 -or (Test-RebootPending)) {
                Set-Stage 'reboot_required' 'WSL 2 已準備，需要重新啟動；重啟後回到中心按繼續準備。'; exit 0
            }
            if ($wslCode -ne 0) { throw 'WSL installation failed' }
        }
        $afterProbe = Get-Probe
        if (-not $afterProbe.wsl_ready) { throw 'WSL verification failed' }
        Set-Stage 'wsl_ready' 'WSL 2 已備妥。'
        exit 0
    }

    # Setup runs under the original desktop user. Only enabling/updating WSL is elevated.
    Set-Stage 'checking' '正在檢查 Windows 與 WSL 執行環境。' 'system'
    if (-not $probe.wsl_ready -or -not $probe.features_ready) {
        Set-Stage 'uac' '請在 Windows 管理員確認視窗允許環境準備。'
        $scriptArg = '"' + $PSCommandPath + '"'
        try {
            $elevated = Start-Process -FilePath 'powershell.exe' -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $scriptArg, '-Mode', 'Elevated')
        } catch {
            Set-Stage 'waiting_action' '管理員確認未完成；可再次按繼續準備。'; exit 0
        }
        $elevatedState = Get-Content -LiteralPath $taskState -Raw | ConvertFrom-Json
        if ($elevatedState.phase -ne 'wsl_ready') { exit 0 }
    }

    if (-not $probe.docker_installed) {
        $installer = Join-Path $taskFolder ('DockerDesktop-' + $dockerVersion + '-Installer.exe')
        $existingValid = (Test-Path -LiteralPath $installer) -and ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant() -eq $dockerSha256)
        if (-not $existingValid) {
            Set-Stage 'downloading_docker' '正在下載官方 Docker Desktop（首次下載可能需要數分鐘）。' 'docker'
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -UseBasicParsing -Uri $dockerUrl -OutFile $installer
        }
        if ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant() -ne $dockerSha256) { throw 'Checksum mismatch' }
        $signature = Get-AuthenticodeSignature -LiteralPath $installer
        $publisher = if ($signature.SignerCertificate) { $signature.SignerCertificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false) } else { '' }
        if ($signature.Status -ne 'Valid' -or $publisher -notin @('Docker Inc', 'Docker Inc.', 'Docker, Inc.', 'Docker, Inc')) {
            throw 'Docker publisher signature verification failed'
        }
        Set-Stage 'installing_docker' '下載與發行者簽章已驗證，正在安裝 Docker Desktop。' 'docker'
        # Do not silently accept Docker's subscription agreement. Docker displays it on first launch.
        $installProcess = Start-Process -FilePath $installer -WindowStyle Hidden -Wait -PassThru -ArgumentList @('install', '--user', '--quiet', '--backend=wsl-2', '--no-windows-containers')
        if ($installProcess.ExitCode -eq 3010 -or (Test-RebootPending)) {
            Set-Stage 'reboot_required' 'Docker 安裝完成，需要重新啟動；重啟後回到中心按繼續準備。' 'docker'; exit 0
        }
        if ($installProcess.ExitCode -ne 0) { throw 'Docker installation failed' }
    }
    $finalProbe = Get-Probe
    if (-not $finalProbe.docker_installed -or -not $finalProbe.wsl_ready) { throw 'Prerequisite verification failed' }
    Set-Stage 'prerequisites_ready' 'WSL 2 與 Docker 已備妥，正在準備 CVAT。' 'docker'
} catch {
    if ($Mode -eq 'Probe') {
        @{ supported=$true; reason='尚無法完整偵測環境，按一鍵準備後重新檢查。'; boot_id=''; probe_error=$_.Exception.GetType().Name } | ConvertTo-Json -Compress
        exit 0
    }
    # Native installer details remain in its own logs. Avoid capturing arbitrary error data.
    $knownErrors = @{
        'Checksum mismatch'='Docker 下載檔案校驗不符，未執行安裝；請重新下載。';
        'Docker publisher signature verification failed'='Docker 發行者簽章驗證失敗，未執行安裝；請檢查系統時間與憑證，再重新下載。';
        'WSL installation failed'='WSL 安裝未成功；請檢查 Windows 更新、網路與組織系統原則後重試。';
        'WSL verification failed'='WSL 安裝後尚未達到 2.1.5 以上版本；請完成 Windows 更新後重試。';
        'Docker installation failed'='Docker Desktop 安裝未完成；請查看 Docker 安裝程式的錯誤訊息後重試。';
        'Prerequisite verification failed'='環境安裝後仍未找到 Docker 或 WSL 2；請重新啟動中心並重試。';
        'Elevation required'='Windows 管理員權限尚未取得，請再次按繼續準備。'
    }
    $failureText = $knownErrors[$_.Exception.Message]
    if (-not $failureText) { $failureText='環境準備未完成。請檢查網路、Windows 更新與管理員權限後重試；下載檔案會重新驗證。' }
    Set-Stage 'error' $failureText
    exit 0
}
