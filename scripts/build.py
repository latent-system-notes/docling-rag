#!/usr/bin/env python
"""Build script for docling-rag distribution packages.

Usage:
    python scripts/build.py wheel      # Build pip-installable wheel
    python scripts/build.py exe        # Build standalone Windows EXE
    python scripts/build.py all        # Build both
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"


def clean():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    for dir_path in [DIST_DIR, BUILD_DIR, ROOT_DIR / "docling_rag.egg-info"]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    print("Done.")


def build_wheel():
    """Build pip-installable wheel package."""
    print("\n" + "=" * 60)
    print("Building wheel package...")
    print("=" * 60 + "\n")

    subprocess.run([sys.executable, "-m", "pip", "install", "build"], check=True)
    subprocess.run([sys.executable, "-m", "build", "--wheel", str(ROOT_DIR)], check=True)

    # Find the built wheel
    wheels = list(DIST_DIR.glob("*.whl"))
    if wheels:
        print(f"\nWheel built: {wheels[0]}")
        print(f"\nInstall with:")
        print(f"  pip install {wheels[0]}")

    return wheels[0] if wheels else None


def build_exe():
    """Build standalone Windows executable using PyInstaller."""
    print("\n" + "=" * 60)
    print("Building standalone Windows executable...")
    print("=" * 60 + "\n")

    # Install PyInstaller if needed
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"], check=True)

    # Create spec file content
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

block_cipher = None

# Collect all source files
src_path = Path('src')
cli_path = Path('cli')
config_path = Path('config')

a = Analysis(
    ['cli/cli.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('config', 'config'),
        ('models/.gitkeep', 'models'),
    ],
    hiddenimports=[
        'typer',
        'rich',
        'chromadb',
        'sentence_transformers',
        'torch',
        'docling',
        'langdetect',
        'fastmcp',
        'pydantic',
        'pydantic_settings',
        'rank_bm25',
        'pypdf',
        'src',
        'src.config',
        'src.models',
        'src.storage',
        'src.storage.chroma_client',
        'src.storage.bm25',
        'src.ingestion',
        'src.ingestion.pipeline',
        'src.ingestion.parallel_pipeline',
        'src.ingestion.checkpoint',
        'src.ingestion.status',
        'src.ingestion.lock',
        'src.ingestion.audit_log',
        'src.mcp',
        'src.mcp.server',
        'src.mcp.status',
        'src.project',
        'src.query',
        'src.utils',
        'cli',
        'cli.cli',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'notebook',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='rag',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''

    spec_file = ROOT_DIR / "rag.spec"
    spec_file.write_text(spec_content)

    # Run PyInstaller
    subprocess.run([
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_file)
    ], check=True, cwd=str(ROOT_DIR))

    exe_path = DIST_DIR / "rag.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\nExecutable built: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"\nInstall by copying rag.exe to a folder in your PATH")

    return exe_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "clean":
        clean()
    elif command == "wheel":
        clean()
        build_wheel()
    elif command == "exe":
        clean()
        build_exe()
    elif command == "all":
        clean()
        build_wheel()
        build_exe()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
