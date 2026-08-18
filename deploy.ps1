<#
.SYNOPSIS
    Nestling deploy automation (Windows). See deploy.sh for the Linux/macOS
    equivalent, and `make help` for the short-form wrapper targets.

.DESCRIPTION
    Fully automates: prerequisite checks (Docker, GPU/NVIDIA Container
    Toolkit), .env bootstrap, LLM model download, `docker compose build/up`,
    health polling, and a post-deploy summary. vLLM and CUDA are never
    installed on the host directly -- they live inside the `llm` service's
    container image (docker/llm/Dockerfile, FROM vllm/vllm-openai) -- this
    script only needs the host GPU driver + NVIDIA Container Toolkit so that
    image can see the GPU.

.PARAMETER Mode
    'app' (backend + UI only) or 'full' (adds the GPU vLLM sidecar). Default: full.

.EXAMPLE
    .\deploy.ps1
    .\deploy.ps1 -Mode app
    .\deploy.ps1 -Down
    .\deploy.ps1 -Logs -LogsService llm
#>
[CmdletBinding()]
param(
    [ValidateSet('app', 'full')]
    [string]$Mode = 'full',
    [switch]$SkipModelDownload,
    [switch]$Down,
    [switch]$Clean,
    [switch]$Logs,
    [string]$LogsService = '',
    [switch]$Status,
    [switch]$ModelOnly,
    [switch]$Yes,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

$ModelId = 'Qwen/Qwen3.5-4B'
$HubSubdir = 'models--Qwen--Qwen3.5-4B'
$LbPort = if ($env:NESTLING_LB_HOST_PORT) { $env:NESTLING_LB_HOST_PORT } else { 8080 }
$LlmPort = if ($env:NESTLING_LLM_HOST_PORT) { $env:NESTLING_LLM_HOST_PORT } else { 8001 }
$script:HfCmd = $null

# ---------- console helpers ----------
function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "[ok] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Write-ErrLine($msg) { Write-Host "[error] $msg" -ForegroundColor Red }

function Show-Usage {
    @'
Usage: .\deploy.ps1 [options]

Actions (default: run the full deploy pipeline):
  -Down                    stop the stack (keeps data volumes)
  -Clean                   stop the stack AND delete data volumes (destructive)
  -Logs [-LogsService x]   tail logs (all services, or one: nginx|nestling|llm)
  -Status                  health check + `docker compose ps`, no changes
  -ModelOnly               download the LLM model only, then exit
  -Help                    show this help

Options (deploy action only):
  -Mode app|full           app-only, or full stack with GPU LLM sidecar (default: full)
  -SkipModelDownload       don't try to download the model even in full mode
  -Yes                     auto-confirm destructive prompts (used by -Clean)
'@ | Write-Host
}

function Test-CommandExists($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# A stopped/starting Docker Desktop can leave the named pipe hanging instead
# of failing fast, which would otherwise block the whole script. Runs an
# external command with a hard wall-clock cap, killing it on timeout.
function Invoke-WithTimeout {
    param(
        [Parameter(Mandatory)] [string]$FilePath,
        [string]$Arguments = '',
        [int]$TimeoutSec = 10
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = $Arguments
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
        try { $proc.Kill() } catch {}
        return [pscustomobject]@{ TimedOut = $true; ExitCode = -1; Output = '' }
    }
    $out = $proc.StandardOutput.ReadToEnd() + $proc.StandardError.ReadToEnd()
    return [pscustomobject]@{ TimedOut = $false; ExitCode = $proc.ExitCode; Output = $out }
}

# ---------- prerequisite checks ----------
function Assert-Docker {
    if (-not (Test-CommandExists docker)) {
        throw "Docker not found. Install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
    }
    $r = Invoke-WithTimeout -FilePath docker -Arguments 'version' -TimeoutSec 10
    if ($r.TimedOut -or $r.ExitCode -ne 0) {
        throw "Docker daemon not reachable. Start Docker Desktop and retry."
    }
    $r = Invoke-WithTimeout -FilePath docker -Arguments 'compose version' -TimeoutSec 10
    if ($r.TimedOut -or $r.ExitCode -ne 0) {
        throw "Docker Compose v2 not found (bundled with modern Docker Desktop)."
    }
    Write-Ok "Docker + Compose v2 available"
}

function Test-GpuAvailable {
    if (-not (Test-CommandExists nvidia-smi)) {
        Write-Warn "no nvidia-smi found -- no NVIDIA driver on this host"
        return $false
    }
    nvidia-smi --query-gpu=name --format=csv,noheader *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "nvidia-smi present but failed to query a GPU"
        return $false
    }
    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "GPU driver found but 'docker run --gpus all' failed -- install the NVIDIA Container Toolkit (WSL2 backend): https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
        return $false
    }
    Write-Ok "GPU detected and usable by Docker"
    return $true
}

# ---------- .env bootstrap ----------
function Initialize-EnvFile {
    if (Test-Path .env) {
        Write-Ok ".env already present, leaving as-is"
        return
    }
    $hfCache = if ($env:NESTLING_HF_CACHE_HOST) { $env:NESTLING_HF_CACHE_HOST } else { Join-Path $env:USERPROFILE ".cache\huggingface" }
    $hfCache = $hfCache -replace '\\', '/'
    New-Item -ItemType Directory -Force -Path $hfCache | Out-Null
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    @"
# Created by deploy.ps1 on $stamp. Safe to edit;
# deploy.ps1/deploy.sh will not overwrite an existing .env.
NESTLING_HF_CACHE_HOST=$hfCache
# NESTLING_LB_HOST_PORT=8080
# NESTLING_LLM_HOST_PORT=8001
# NESTLING_INSTALL_ML=0
"@ | Set-Content -Path .env -Encoding utf8
    Write-Ok "wrote .env (NESTLING_HF_CACHE_HOST=$hfCache)"
}

function Get-HfCacheFromEnv {
    if (Test-Path .env) {
        $line = Select-String -Path .env -Pattern '^NESTLING_HF_CACHE_HOST=' | Select-Object -First 1
        if ($line) {
            return ($line.Line -replace '^NESTLING_HF_CACHE_HOST=', '')
        }
    }
    if ($env:NESTLING_HF_CACHE_HOST) { return $env:NESTLING_HF_CACHE_HOST }
    return (Join-Path $env:USERPROFILE ".cache\huggingface") -replace '\\', '/'
}

# ---------- model download (mirrors docker/llm/entrypoint.sh) ----------
function Resolve-ModelSnapshotPath($hubDir) {
    if (-not (Test-Path $hubDir)) { return $null }
    $refFile = Join-Path $hubDir "refs\main"
    if (Test-Path $refFile) {
        $rev = (Get-Content $refFile -Raw).Trim()
        $snapshotPath = Join-Path $hubDir "snapshots\$rev"
        if (Test-Path $snapshotPath) { return $snapshotPath }
    }
    $snapshotsDir = Join-Path $hubDir "snapshots"
    if (Test-Path $snapshotsDir) {
        $first = Get-ChildItem $snapshotsDir -Directory | Select-Object -First 1
        if ($first) { return $first.FullName }
    }
    return $null
}

function Test-WeightsPresent($path) {
    if (-not $path -or -not (Test-Path $path)) { return $false }
    $safetensors = Get-ChildItem -Path $path -Filter "model*.safetensors" -ErrorAction SilentlyContinue
    if ($safetensors) { return $true }
    return Test-Path (Join-Path $path "pytorch_model.bin")
}

function Test-ModelDownloaded($hfHome) {
    $hubDir = Join-Path $hfHome "hub\$HubSubdir"
    $snapshot = Resolve-ModelSnapshotPath $hubDir
    return Test-WeightsPresent $snapshot
}

function Install-HfCli {
    if (Test-CommandExists hf) { $script:HfCmd = 'hf'; return }
    if (Test-CommandExists huggingface-cli) { $script:HfCmd = 'huggingface-cli'; return }
    Write-Step "installing huggingface_hub CLI"
    $py = if (Test-CommandExists python) { 'python' } elseif (Test-CommandExists python3) { 'python3' } else { $null }
    if (-not $py) {
        throw "no python found -- install Python 3 to download the model, or download it manually."
    }
    & $py -m pip install --user -U "huggingface_hub[cli]"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to install huggingface_hub[cli] via pip"
    }
    if (Test-CommandExists hf) { $script:HfCmd = 'hf'; return }
    if (Test-CommandExists huggingface-cli) { $script:HfCmd = 'huggingface-cli'; return }
    throw "huggingface_hub CLI still not on PATH after install. Try: $py -m pip install -U 'huggingface_hub[cli]' and re-run (may need a new shell for PATH to refresh)."
}

function Invoke-ModelDownload {
    $hfHome = Get-HfCacheFromEnv
    New-Item -ItemType Directory -Force -Path $hfHome | Out-Null
    if (Test-ModelDownloaded $hfHome) {
        Write-Ok "model already cached at $hfHome -- skipping download"
        return
    }
    Install-HfCli
    Write-Step "downloading $ModelId (several GB, this can take a while)"
    $env:HF_HOME = $hfHome
    & $script:HfCmd download $ModelId
    if ($LASTEXITCODE -ne 0) {
        throw "model download failed. Check network access to huggingface.co and retry, or run: .\deploy.ps1 -ModelOnly"
    }
    if (Test-ModelDownloaded $hfHome) {
        Write-Ok "model downloaded to $hfHome"
    } else {
        Write-Warn "download finished but weights not found at the expected path ($hfHome\hub\$HubSubdir) -- the llm service may fail to start"
    }
}

# ---------- compose orchestration ----------
function Invoke-ComposeBuild($mode) {
    if ($mode -eq 'full') {
        docker compose --profile llm build
    } else {
        docker compose build nestling nginx
    }
    if ($LASTEXITCODE -ne 0) { throw "docker compose build failed" }
}

function Invoke-ComposeUp($mode) {
    if ($mode -eq 'full') {
        docker compose --profile llm up --build -d
    } else {
        docker compose up --build -d nestling nginx
    }
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
}

function Wait-Health($url, [int]$timeoutSec = 180) {
    Write-Step "waiting for $url (up to ${timeoutSec}s)"
    $waited = 0
    while ($waited -lt $timeoutSec) {
        try {
            Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing | Out-Null
            Write-Ok "healthy: $url"
            return
        } catch {
            Start-Sleep -Seconds 3
            $waited += 3
        }
    }
    Write-ErrLine "health check timed out after ${timeoutSec}s: $url"
    docker compose logs --tail 80
    throw "health check timed out"
}

function Wait-LlmHealth {
    $url = "http://localhost:$LlmPort/v1/models"
    Write-Step "checking LLM sidecar (advisory only -- first load can take up to 15 min)"
    $waited = 0
    $timeout = 30
    while ($waited -lt $timeout) {
        try {
            Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing | Out-Null
            Write-Ok "LLM sidecar ready: $url"
            return
        } catch {
            Start-Sleep -Seconds 3
            $waited += 3
        }
    }
    Write-Warn "LLM sidecar not ready yet -- this is normal on first run (large model load)."
    Write-Warn "watch progress with: docker compose --profile llm logs -f llm"
}

function Invoke-DbMigrateSafety {
    Write-Step "running DB schema safety net inside the container"
    docker compose exec -T nestling python scripts/migrate_db.py
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "migrate_db.py step failed (non-fatal -- the app creates tables lazily on first use)"
    } else {
        Write-Ok "database schema verified"
    }
}

function Show-Summary($mode, $gpuUsed) {
    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Nestling is up."
    Write-Host ""
    Write-Host "  Web UI / API (via nginx):  http://localhost:$LbPort"
    Write-Host "  Health check:              http://localhost:$LbPort/api/health"
    if ($mode -eq 'full' -and $gpuUsed) {
        Write-Host "  LLM (OpenAI-compatible):   http://localhost:$LlmPort/v1"
    } elseif ($mode -eq 'app') {
        Write-Host ""
        Write-Host "  Note: deployed in APP-ONLY mode -- chat uses extractive RAG, not generative Qwen."
        Write-Host "  Re-run with GPU + toolkit available: .\deploy.ps1 -Mode full"
    }
    Write-Host ""
    Write-Host "  Logs:      .\deploy.ps1 -Logs"
    Write-Host "  Stop:      .\deploy.ps1 -Down"
    Write-Host "  Wipe data: .\deploy.ps1 -Clean   (DESTROYS volumes)"
    Write-Host "=================================================="
}

function Invoke-Down {
    docker compose --profile llm down
}

function Invoke-Clean {
    if (-not $Yes) {
        $reply = Read-Host "This will DELETE all Nestling data volumes (children DB, uploads, HF cache volume). Type 'yes' to continue"
        if ($reply -ne 'yes') {
            Write-ErrLine "aborted"
            exit 1
        }
    }
    docker compose --profile llm down -v
}

function Show-Logs($service) {
    if ($service) {
        docker compose --profile llm logs -f $service
    } else {
        docker compose --profile llm logs -f
    }
}

function Show-Status {
    $url = "http://localhost:$LbPort/api/health"
    try {
        $resp = Invoke-RestMethod -Uri $url -TimeoutSec 5
        $resp | ConvertTo-Json -Compress
    } catch {
        Write-Warn "not reachable: $url"
    }
    $r = Invoke-WithTimeout -FilePath docker -Arguments 'compose ps' -TimeoutSec 10
    if ($r.TimedOut) {
        Write-Warn "docker compose ps did not respond (Docker daemon likely unreachable)"
    } else {
        Write-Host $r.Output
    }
}

# ---------- main ----------
try {
    if ($Help) { Show-Usage; exit 0 }
    if ($Down) { Invoke-Down; exit 0 }
    if ($Clean) { Invoke-Clean; exit 0 }
    if ($Logs) { Show-Logs $LogsService; exit 0 }
    if ($Status) { Show-Status; exit 0 }
    if ($ModelOnly) {
        Initialize-EnvFile
        Invoke-ModelDownload
        exit 0
    }

    Write-Step "checking prerequisites"
    Assert-Docker

    Write-Step "preparing .env"
    Initialize-EnvFile

    $gpu = $false
    $effectiveMode = $Mode
    if ($effectiveMode -eq 'full') {
        Write-Step "checking for GPU"
        $gpu = Test-GpuAvailable
        if (-not $gpu) {
            Write-Warn "no usable GPU -- falling back to app-only deploy"
            $effectiveMode = 'app'
        }
    }

    if ($effectiveMode -eq 'full' -and -not $SkipModelDownload) {
        Write-Step "ensuring LLM model is downloaded"
        Invoke-ModelDownload
    } elseif ($effectiveMode -eq 'full') {
        Write-Warn "skipping model download (-SkipModelDownload) -- llm service will fail to start if weights are missing"
    }

    Write-Step "building images"
    Invoke-ComposeBuild $effectiveMode

    Write-Step "starting stack (mode=$effectiveMode)"
    Invoke-ComposeUp $effectiveMode

    Wait-Health "http://localhost:$LbPort/api/health" 180
    if ($effectiveMode -eq 'full') {
        Wait-LlmHealth
    }

    Invoke-DbMigrateSafety

    Show-Summary $effectiveMode $gpu
    exit 0
} catch {
    Write-ErrLine $_.Exception.Message
    exit 1
}
