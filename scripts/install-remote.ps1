#Requires -Version 5.1
<#
.SYNOPSIS
    Remote installation script for docling-rag

.DESCRIPTION
    Downloads and installs docling-rag from GitHub releases or PyPI.
    Can be run directly:

    irm https://raw.githubusercontent.com/yourusername/docling-rag/main/scripts/install-remote.ps1 | iex

.PARAMETER Version
    Specific version to install (default: latest)

.PARAMETER GPU
    Install with GPU support
#>

param(
    [string]$Version = "latest",
    [switch]$GPU
)

$ErrorActionPreference = 'Stop'

Write-Host ""
Write-Host "Installing docling-rag..." -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pyver = python --version 2>&1
    Write-Host "Found $pyver" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Install Python 3.11+ first:" -ForegroundColor Red
    Write-Host "  winget install Python.Python.3.12" -ForegroundColor Yellow
    exit 1
}

# Install
$package = "docling-rag"
if ($Version -ne "latest") {
    $package = "docling-rag==$Version"
}
if ($GPU) {
    $package = "$package[gpu]"
}

Write-Host "Installing $package..." -ForegroundColor Yellow
pip install $package

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Installation complete! Run 'rag --help' to get started." -ForegroundColor Green
} else {
    Write-Host "Installation failed." -ForegroundColor Red
    exit 1
}
