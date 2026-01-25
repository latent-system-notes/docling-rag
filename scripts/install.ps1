#Requires -Version 5.1
<#
.SYNOPSIS
    Docling-RAG Installation Script for Windows

.DESCRIPTION
    Installs docling-rag as a command-line tool on Windows.
    Supports installation from wheel, source, or PyPI.

.PARAMETER Method
    Installation method: 'pip', 'pipx', 'wheel', or 'source'
    Default: 'pip'

.PARAMETER GPU
    Install with GPU (CUDA) support

.PARAMETER Dev
    Install development dependencies

.EXAMPLE
    .\install.ps1
    # Standard pip install

.EXAMPLE
    .\install.ps1 -Method pipx
    # Install in isolated environment with pipx

.EXAMPLE
    .\install.ps1 -GPU
    # Install with GPU support
#>

param(
    [ValidateSet('pip', 'pipx', 'wheel', 'source')]
    [string]$Method = 'pip',

    [switch]$GPU,

    [switch]$Dev
)

$ErrorActionPreference = 'Stop'

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Docling-RAG Installation Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Get script and project directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir

# Check Python installation
function Test-Python {
    try {
        $version = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Found $version" -ForegroundColor Green
            return $true
        }
    } catch {}

    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.11+ from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or install via winget:" -ForegroundColor Yellow
    Write-Host "  winget install Python.Python.3.12" -ForegroundColor Yellow
    return $false
}

# Check pipx installation
function Test-Pipx {
    try {
        $null = pipx --version 2>&1
        if ($LASTEXITCODE -eq 0) { return $true }
    } catch {}
    return $false
}

# Install pipx if needed
function Install-Pipx {
    Write-Host "Installing pipx..." -ForegroundColor Yellow
    pip install pipx
    python -m pipx ensurepath
    Write-Host "pipx installed. You may need to restart your terminal." -ForegroundColor Green
}

# Main installation
function Install-DoclingRag {
    param($Method, $GPU, $Dev)

    $extras = @()
    if ($GPU) { $extras += "gpu" }
    if ($Dev) { $extras += "dev" }

    $extrasStr = ""
    if ($extras.Count -gt 0) {
        $extrasStr = "[$($extras -join ',')]"
    }

    switch ($Method) {
        'pip' {
            Write-Host "Installing with pip..." -ForegroundColor Yellow

            # Check for local wheel first
            $wheelFile = Get-ChildItem -Path "$RootDir\dist\*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1

            if ($wheelFile) {
                Write-Host "Found wheel: $($wheelFile.Name)" -ForegroundColor Green
                pip install "$($wheelFile.FullName)$extrasStr"
            } elseif (Test-Path "$RootDir\pyproject.toml") {
                Write-Host "Installing from source..." -ForegroundColor Green
                pip install "$RootDir$extrasStr"
            } else {
                Write-Host "Installing from PyPI..." -ForegroundColor Green
                pip install "docling-rag$extrasStr"
            }
        }

        'pipx' {
            if (-not (Test-Pipx)) {
                Install-Pipx
            }

            Write-Host "Installing with pipx (isolated environment)..." -ForegroundColor Yellow

            $wheelFile = Get-ChildItem -Path "$RootDir\dist\*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1

            if ($wheelFile) {
                pipx install "$($wheelFile.FullName)"
            } elseif (Test-Path "$RootDir\pyproject.toml") {
                pipx install "$RootDir"
            } else {
                pipx install docling-rag
            }

            if ($GPU) {
                pipx inject docling-rag accelerate
            }
        }

        'wheel' {
            $wheelFile = Get-ChildItem -Path "$RootDir\dist\*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1

            if (-not $wheelFile) {
                Write-Host "No wheel found. Building..." -ForegroundColor Yellow
                pip install build
                python -m build --wheel $RootDir
                $wheelFile = Get-ChildItem -Path "$RootDir\dist\*.whl" | Select-Object -First 1
            }

            Write-Host "Installing wheel: $($wheelFile.Name)" -ForegroundColor Green
            pip install "$($wheelFile.FullName)$extrasStr"
        }

        'source' {
            Write-Host "Installing from source..." -ForegroundColor Yellow
            pip install -e "$RootDir$extrasStr"
        }
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Installation failed" -ForegroundColor Red
        exit 1
    }
}

# Verify installation
function Test-Installation {
    Write-Host ""
    Write-Host "Verifying installation..." -ForegroundColor Yellow

    try {
        $help = rag --help 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: 'rag' command is available" -ForegroundColor Green
            return $true
        }
    } catch {}

    Write-Host "WARNING: 'rag' command not found in PATH" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You may need to:" -ForegroundColor Yellow
    Write-Host "  1. Restart your terminal" -ForegroundColor Yellow
    Write-Host "  2. Add Python Scripts folder to PATH" -ForegroundColor Yellow

    # Find Python Scripts folder
    $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($pythonPath) {
        $scriptsPath = Join-Path (Split-Path (Split-Path $pythonPath)) "Scripts"
        Write-Host ""
        Write-Host "Python Scripts folder: $scriptsPath" -ForegroundColor Cyan
    }

    return $false
}

# Main execution
if (-not (Test-Python)) {
    exit 1
}

Install-DoclingRag -Method $Method -GPU $GPU -Dev $Dev

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

Test-Installation

Write-Host ""
Write-Host "Quick start:" -ForegroundColor Cyan
Write-Host "  rag init                    # First-time setup" -ForegroundColor White
Write-Host "  rag project list            # List projects" -ForegroundColor White
Write-Host "  rag ingestion start ./docs  # Ingest documents" -ForegroundColor White
Write-Host "  rag mcp serve               # Start MCP server" -ForegroundColor White
Write-Host ""
Write-Host "For help: rag --help" -ForegroundColor Cyan
Write-Host ""
