#!/usr/bin/env python
"""Create a portable distribution with embedded Python and all dependencies.

This approach is more reliable than PyInstaller for ML applications.
Creates a self-contained folder that can be copied to any Windows machine.

Usage:
    python scripts/create_portable.py

Output:
    dist/docling-rag-portable/
        python/           - Embedded Python
        Lib/              - All dependencies
        models/           - Pre-downloaded models
        rag.bat          - Launcher script
        install.bat      - PATH setup script
"""
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DIST_DIR = ROOT_DIR / "dist"
PORTABLE_DIR = DIST_DIR / "docling-rag-portable"

# Python embedded distribution URL (Windows x64)
PYTHON_VERSION = "3.12.8"
PYTHON_EMBED_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"


def download_embedded_python():
    """Download embedded Python distribution."""
    print("\n" + "=" * 60)
    print("Step 1: Downloading embedded Python...")
    print("=" * 60 + "\n")

    python_dir = PORTABLE_DIR / "python"
    python_dir.mkdir(parents=True, exist_ok=True)

    zip_path = DIST_DIR / "python-embed.zip"

    if not zip_path.exists():
        print(f"  Downloading Python {PYTHON_VERSION}...")
        urllib.request.urlretrieve(PYTHON_EMBED_URL, zip_path)
        print(f"  Downloaded: {zip_path}")

    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(python_dir)

    # Enable pip by modifying python3XX._pth
    pth_files = list(python_dir.glob("python*._pth"))
    if pth_files:
        pth_file = pth_files[0]
        content = pth_file.read_text()
        # Uncomment import site
        content = content.replace("#import site", "import site")
        # Add Lib folder
        content += "\n../Lib\n../Lib/site-packages\n"
        pth_file.write_text(content)
        print(f"  Modified: {pth_file.name}")

    return python_dir


def install_pip(python_dir):
    """Install pip in embedded Python."""
    print("\n" + "=" * 60)
    print("Step 2: Installing pip...")
    print("=" * 60 + "\n")

    python_exe = python_dir / "python.exe"

    # Download get-pip.py
    getpip_path = DIST_DIR / "get-pip.py"
    if not getpip_path.exists():
        print("  Downloading get-pip.py...")
        urllib.request.urlretrieve(
            "https://bootstrap.pypa.io/get-pip.py",
            getpip_path
        )

    # Run get-pip.py
    print("  Installing pip...")
    subprocess.run([str(python_exe), str(getpip_path)], check=True)

    return python_dir / "Scripts" / "pip.exe"


def install_dependencies(pip_exe):
    """Install all dependencies."""
    print("\n" + "=" * 60)
    print("Step 3: Installing dependencies (this may take a while)...")
    print("=" * 60 + "\n")

    # Create Lib directory for site-packages
    lib_dir = PORTABLE_DIR / "Lib" / "site-packages"
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Install wheel first
    wheel_files = list((ROOT_DIR / "dist").glob("docling_rag-*.whl"))

    if wheel_files:
        wheel_file = wheel_files[0]
        print(f"  Installing from wheel: {wheel_file.name}")
        subprocess.run([
            str(pip_exe), "install",
            "--target", str(lib_dir),
            "--no-warn-script-location",
            str(wheel_file)
        ], check=True)
    else:
        print("  Installing from source...")
        subprocess.run([
            str(pip_exe), "install",
            "--target", str(lib_dir),
            "--no-warn-script-location",
            str(ROOT_DIR)
        ], check=True)


def download_models():
    """Download models for offline use."""
    print("\n" + "=" * 60)
    print("Step 4: Downloading models for offline use...")
    print("=" * 60 + "\n")

    models_dir = PORTABLE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Set environment for model download
    os.environ["HF_HOME"] = str(models_dir / ".cache")
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir / ".cache")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    # Add our Lib to path
    sys.path.insert(0, str(PORTABLE_DIR / "Lib" / "site-packages"))

    print("  Downloading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            cache_folder=str(models_dir / "sentence-transformers")
        )
        del model
        print("  Embedding model downloaded.")
    except Exception as e:
        print(f"  Warning: Could not download embedding model: {e}")

    # Create models marker file
    (models_dir / ".downloaded").write_text(f"Downloaded for portable distribution")


def create_launchers():
    """Create launcher scripts."""
    print("\n" + "=" * 60)
    print("Step 5: Creating launcher scripts...")
    print("=" * 60 + "\n")

    # Main launcher: rag.bat
    launcher = PORTABLE_DIR / "rag.bat"
    launcher.write_text('''@echo off
setlocal

REM Docling-RAG Portable Launcher
set "SCRIPT_DIR=%~dp0"
set "PYTHON_DIR=%SCRIPT_DIR%python"
set "LIB_DIR=%SCRIPT_DIR%Lib\\site-packages"
set "MODELS_DIR=%SCRIPT_DIR%models"

REM Set Python paths
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\\Scripts;%PATH%"
set "PYTHONPATH=%LIB_DIR%;%PYTHONPATH%"

REM Set offline mode and model paths
set "HF_HOME=%MODELS_DIR%\\.cache"
set "TRANSFORMERS_CACHE=%MODELS_DIR%\\.cache"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "RAG_OFFLINE_MODE=true"

REM Run the CLI
"%PYTHON_DIR%\\python.exe" -m cli.cli %*

endlocal
''')
    print(f"  Created: {launcher}")

    # Install to PATH script
    install_script = PORTABLE_DIR / "install.bat"
    install_script.write_text('''@echo off
REM Add docling-rag to user PATH

echo.
echo Adding docling-rag to PATH...
echo.

set "SCRIPT_DIR=%~dp0"

REM Add to user PATH
for /f "tokens=2*" %%a in ('reg query "HKCU\\Environment" /v Path 2^>nul') do set "CURRENT_PATH=%%b"

echo Current PATH: %CURRENT_PATH%
echo.
echo Adding: %SCRIPT_DIR%

setx PATH "%CURRENT_PATH%;%SCRIPT_DIR%"

echo.
echo Done! Please restart your terminal.
echo You can now run 'rag' from anywhere.
echo.
pause
''')
    print(f"  Created: {install_script}")

    # Uninstall from PATH script
    uninstall_script = PORTABLE_DIR / "uninstall.bat"
    uninstall_script.write_text('''@echo off
REM Remove docling-rag from PATH

echo.
echo To uninstall:
echo 1. Remove this folder from your PATH environment variable
echo 2. Delete this folder
echo.
echo Open System Properties ^> Environment Variables to edit PATH
echo.
pause
''')
    print(f"  Created: {uninstall_script}")

    # README
    readme = PORTABLE_DIR / "README.txt"
    readme.write_text('''Docling-RAG Portable Edition
============================

This is a fully self-contained distribution that includes:
- Embedded Python runtime
- All Python dependencies
- Pre-downloaded ML models

No Python installation required on the target machine.

INSTALLATION
------------
Option 1: Run from this folder
  - Double-click 'rag.bat' or run it from command line
  - Example: rag.bat --help

Option 2: Add to PATH (run commands from anywhere)
  - Run 'install.bat' as Administrator
  - Restart your terminal
  - Now you can just type 'rag --help'

QUICK START
-----------
1. rag init                    - First-time setup
2. rag ingestion start C:\\docs - Add documents
3. rag query "your question"   - Search documents
4. rag mcp serve               - Start MCP server

OFFLINE OPERATION
-----------------
This distribution is fully offline-capable:
- All ML models are included
- No internet connection required

SYSTEM REQUIREMENTS
-------------------
- Windows 10/11 (64-bit)
- 4GB RAM minimum
- 3GB disk space

For help: rag --help
''')
    print(f"  Created: {readme}")


def calculate_size():
    """Calculate and report distribution size."""
    if not PORTABLE_DIR.exists():
        return

    total_size = sum(f.stat().st_size for f in PORTABLE_DIR.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    size_gb = total_size / (1024 * 1024 * 1024)

    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print(f"\nPortable distribution: {PORTABLE_DIR}")
    print(f"Total size: {size_mb:.1f} MB ({size_gb:.2f} GB)")
    print(f"\nTo distribute:")
    print(f"  1. Copy the entire '{PORTABLE_DIR.name}' folder")
    print(f"  2. Run 'install.bat' on target machine (optional)")
    print(f"  3. Or just run 'rag.bat' directly")
    print(f"\nTo test:")
    print(f"  {PORTABLE_DIR}\\rag.bat --help")


def main():
    print("\n" + "=" * 60)
    print("  Docling-RAG Portable Distribution Builder")
    print("  (Self-contained, offline-capable)")
    print("=" * 60)

    # Clean previous build
    if PORTABLE_DIR.exists():
        print(f"\nRemoving previous build: {PORTABLE_DIR}")
        shutil.rmtree(PORTABLE_DIR)

    PORTABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Build wheel first if not exists
    wheel_files = list((ROOT_DIR / "dist").glob("docling_rag-*.whl"))
    if not wheel_files:
        print("\nBuilding wheel first...")
        subprocess.run([sys.executable, "-m", "pip", "install", "build"], check=True)
        subprocess.run([sys.executable, "-m", "build", "--wheel", str(ROOT_DIR)], check=True)

    # Step 1: Download embedded Python
    python_dir = download_embedded_python()

    # Step 2: Install pip
    pip_exe = install_pip(python_dir)

    # Step 3: Install dependencies
    install_dependencies(pip_exe)

    # Step 4: Download models
    download_models()

    # Step 5: Create launchers
    create_launchers()

    # Report
    calculate_size()


if __name__ == "__main__":
    main()
