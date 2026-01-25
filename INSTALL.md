# Installation Guide

## Quick Install (Windows Command Line)

### Option 1: pip install (Recommended)

```powershell
# Install from source
pip install .

# Or install from PyPI (when published)
pip install docling-rag

# With GPU support
pip install docling-rag[gpu]
```

### Option 2: Using the install script

```powershell
# PowerShell (recommended)
.\scripts\install.ps1

# With GPU support
.\scripts\install.ps1 -GPU

# With pipx (isolated environment)
.\scripts\install.ps1 -Method pipx

# Command Prompt
scripts\install.bat
```

### Option 3: One-liner remote install

```powershell
# Install latest from PyPI
pip install docling-rag

# Or download and run install script
irm https://raw.githubusercontent.com/yourusername/docling-rag/main/scripts/install-remote.ps1 | iex
```

## Building from Source

### Build wheel package

```powershell
# Build wheel
python scripts/build.py wheel

# The wheel will be in dist/docling_rag-0.1.0-py3-none-any.whl
pip install dist/docling_rag-0.1.0-py3-none-any.whl
```

### Build standalone executable (no Python required)

```powershell
# Build standalone EXE
python scripts/build.py exe

# The executable will be in dist/rag.exe
# Copy it to a folder in your PATH
```

## Installation Methods Comparison

| Method | Python Required | Isolated | Size | Best For |
|--------|----------------|----------|------|----------|
| pip install | Yes | No | Small | Developers |
| pipx install | Yes | Yes | Small | CLI users |
| Standalone EXE | No | Yes | ~500MB | Distribution |

## Requirements

- Python 3.11 or later
- Windows 10/11 (64-bit)
- 4GB+ RAM recommended
- 2GB+ disk space for models

## Post-Installation

After installation, verify it works:

```powershell
# Check version
rag --help

# Initialize first project
rag init

# Download models (required for offline operation)
rag config models --download
```

## Troubleshooting

### 'rag' command not found

Add Python Scripts to PATH:

```powershell
# Find Python Scripts folder
python -c "import sys; print(sys.prefix + '\\Scripts')"

# Add to PATH (run as Administrator)
$scriptsPath = python -c "import sys; print(sys.prefix + '\\Scripts')"
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$scriptsPath", "User")
```

### GPU not detected

For NVIDIA GPU support:

```powershell
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install GPU extras
pip install docling-rag[gpu]
```

### SSL/Network errors

For air-gapped environments:

```powershell
# Download models first (with network)
rag config models --download

# Then run in offline mode (default)
# All operations will use local models
```

## Uninstall

```powershell
# pip
pip uninstall docling-rag

# pipx
pipx uninstall docling-rag

# Remove data (optional)
rm -r ~/.rag
```
