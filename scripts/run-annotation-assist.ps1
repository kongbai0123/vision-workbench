[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ManifestPath,
    [Parameter(Mandatory=$true)][string]$Destination,
    [Parameter(Mandatory=$true)][string]$ModelHome,
    [Parameter(Mandatory=$true)][string]$CancelFile
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$profile = 'VisionWorkbenchAnnotationAssistV1'
$workerRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'workbench'
$source = Join-Path $PSScriptRoot 'AnnotationAssistProcess.cs'

function Get-RegularEntry([string]$Path, [bool]$Directory) {
    $entry = Get-Item -LiteralPath $Path -Force
    if ($entry.PSIsContainer -ne $Directory -or
        ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Expected a regular local input: $Path"
    }
    return $entry.FullName
}

function Get-LocalSha256([string]$Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

function Test-Inside([string]$Path, [string]$Root) {
    return $Path.StartsWith(($Root.TrimEnd([char[]]@('/','\')) + '\'), [StringComparison]::OrdinalIgnoreCase)
}

$manifestSource = Get-RegularEntry $ManifestPath $false
$inputRoot = Split-Path -Parent $manifestSource
$destinationPath = Get-RegularEntry $Destination $true
$models = Get-RegularEntry $ModelHome $true
$runtime = Get-RegularEntry (Join-Path $models 'runtime') $true
$python = Get-RegularEntry (Join-Path $runtime 'python.exe') $false
$dino = Get-RegularEntry (Join-Path $models 'MM-Grounding-DINO-Tiny') $true
$qwen = Get-RegularEntry (Join-Path $models 'Qwen3-VL-4B-Instruct') $true
if ($inputRoot -eq $destinationPath -or (Test-Inside $inputRoot $destinationPath) -or
    (Test-Inside $destinationPath $inputRoot)) {
    throw 'Keep the job input and result directories separate.'
}

$work = $null
$cancelled = $false
$listeners = [System.Collections.Generic.List[Net.Sockets.TcpListener]]::new()
$stageName = 'manifest.json'
try {
    Add-Type -Path $source
    $sid = [AnnotationAssistProcess]::ProfileSid($profile)
    $exemptions = & CheckNetIsolation.exe LoopbackExempt -s
    if ($LASTEXITCODE -ne 0 -or ($exemptions -join "`n").Contains($sid)) {
        throw 'The AppContainer loopback exemption check failed.'
    }

    $tempRoot = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) "Packages\$profile\AC\Temp"
    $work = Join-Path $tempRoot "run-$([Guid]::NewGuid().ToString('N'))"
    $stagedInput = Join-Path $work 'input'
    New-Item -ItemType Directory -Path $stagedInput -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $destinationPath 'live.json'),
        (@{run_folder=(Split-Path -Leaf $work)} | ConvertTo-Json -Compress))
    $stagedManifest = Join-Path $stagedInput $stageName
    Copy-Item -LiteralPath $manifestSource -Destination $stagedManifest
    $manifest = Get-Content -Raw -Encoding utf8 -LiteralPath $stagedManifest | ConvertFrom-Json
    $stagedImages = Join-Path $stagedInput 'images'
    New-Item -ItemType Directory -Path $stagedImages | Out-Null
    $imageRoot = Get-RegularEntry (Join-Path $inputRoot 'images') $true
    foreach ($frame in $manifest.frames) {
        $relative = ([string]$frame.image).Replace('\','/')
        if ($relative -notmatch '^images/[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+$') {
            throw 'The worker input manifest may reference only flat, local staged images.'
        }
        $image = Get-RegularEntry (Join-Path $inputRoot $relative) $false
        if (-not (Test-Inside $image $imageRoot)) { throw 'Invalid inference image path.' }
        Copy-Item -LiteralPath $image -Destination (Join-Path $stagedInput $relative)
    }
    foreach ($name in @('annotation_assist_probe.py','annotation_assist_worker.py',
                       'annotation_assist_confidence.py','annotation_assist_grounding.py')) {
        Copy-Item -LiteralPath (Join-Path $workerRoot $name) -Destination $work
    }

    foreach ($directory in @($runtime,$dino,$qwen)) {
        & icacls.exe $directory /grant:r "*${sid}:(OI)(CI)RX" /Q | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Unable to grant AppContainer read access: $directory" }
    }

    $env:VW_ANNOTATION_ASSIST_PROFILE = $profile
    $env:TEMP = $work
    $env:TMP = $work
    $env:HF_HOME = Join-Path $work 'hf'
    $env:TORCH_HOME = Join-Path $work 'torch'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:HF_HUB_DISABLE_TELEMETRY = '1'
    $env:USE_HUB_KERNELS = 'NO'

    $addresses = [ordered]@{loopback_ipv4=[Net.IPAddress]::Loopback}
    if ([Net.Sockets.Socket]::OSSupportsIPv6) { $addresses['loopback_ipv6'] = [Net.IPAddress]::IPv6Loopback }
    $adapter = [Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
        Where-Object { $_.OperationalStatus -eq 'Up' } |
        ForEach-Object { $_.GetIPProperties().UnicastAddresses } |
        Where-Object { $_.Address.AddressFamily -eq 'InterNetwork' -and -not [Net.IPAddress]::IsLoopback($_.Address) } |
        Select-Object -First 1
    if ($adapter) { $addresses['host_adapter'] = $adapter.Address }
    $probe = @{private_files=@($manifestSource); listeners=@()
        launcher_sha256=(Get-LocalSha256 $PSCommandPath)
        token_launcher_sha256=(Get-LocalSha256 $source)
        manifest_sha256=(Get-LocalSha256 $stagedManifest)}
    foreach ($name in $addresses.Keys) {
        $address = $addresses[$name]
        $listener = [Net.Sockets.TcpListener]::new($address, 0)
        $listener.Start()
        $listeners.Add($listener)
        $port = $listener.LocalEndpoint.Port
        $control = [Net.Sockets.TcpClient]::new($address.AddressFamily)
        try {
            $control.Connect($address, $port)
            $listener.AcceptTcpClient().Dispose()
        } finally {
            $control.Dispose()
        }
        $probe.listeners += @{name=$name; ip=$address.ToString(); port=$port}
    }
    $env:VW_ANNOTATION_ASSIST_PROBE = $probe | ConvertTo-Json -Compress -Depth 5
    $output = Join-Path $work 'inference.json'
    $tokens = @($python,'-I','-B','-X','utf8',(Join-Path $work 'annotation_assist_probe.py'),
        (Join-Path $work 'annotation_assist_worker.py'),'--manifest',$stagedManifest,
        '--dino-dir',$dino,'--qwen-dir',$qwen,'--output',$output)
    $command = ($tokens | ForEach-Object { '"' + $_ + '"' }) -join ' '
    if (Test-Path -LiteralPath $CancelFile) { throw [OperationCanceledException]::new('Stopped before inference.') }
    $exitCode = [AnnotationAssistProcess]::RunCancelable($profile, $python, $command, $work, 7200, $CancelFile)
    if ($exitCode -ne 0) { throw "The isolated inference worker exited with code $exitCode." }
    $proof = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $work 'isolation.json') | ConvertFrom-Json
    if ($proof.schema -ne 'vision_workbench.annotation_assist_isolation.v1' -or
        $proof.profile -ne $profile -or $proof.appcontainer_token -ne $true -or
        $null -eq $proof.os_network_capabilities -or $proof.os_network_capabilities -ne 0 -or
        $proof.network_denied -ne $true -or $proof.host_files_denied -ne $true -or
        $proof.cuda -ne $true -or $proof.manifest_sha256 -ne (Get-LocalSha256 $manifestSource)) {
        throw 'Isolation proof does not validate. No model result can be published.'
    }
    Copy-Item -LiteralPath $output -Destination $destinationPath
} catch {
    $cause = $_.Exception
    while ($cause) {
        if ($cause -is [OperationCanceledException]) { $cancelled = $true }
        $cause = $cause.InnerException
    }
    if (-not $cancelled) { throw }
} finally {
    foreach ($listener in $listeners) { $listener.Stop() }
    foreach ($name in @('isolation.json','worker.log')) {
        if ($work -and (Test-Path -LiteralPath (Join-Path $work $name) -PathType Leaf)) {
            Copy-Item -LiteralPath (Join-Path $work $name) -Destination $destinationPath
        }
    }
    if ($work -and (Test-Path -LiteralPath $work -PathType Container)) {
        $resolved = (Resolve-Path -LiteralPath $work).Path
        $expectedParent = (Resolve-Path -LiteralPath $tempRoot).Path
        if ((Split-Path -Parent $resolved) -eq $expectedParent -and
            (Split-Path -Leaf $resolved) -match '^run-[a-f0-9]{32}$' -and
            -not (Get-Item -LiteralPath $resolved).Attributes.HasFlag([IO.FileAttributes]::ReparsePoint)) {
            Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction Continue
        }
    }
}
if ($cancelled) { exit 130 }
