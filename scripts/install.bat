@echo off
REM Docling-RAG Windows Installation Script
REM Run this script in an elevated (Administrator) command prompt for system-wide install
REM Or run without admin for user-only install

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   Docling-RAG Installation Script
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.11+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo Found Python %pyver%

REM Check if we have a wheel file or should install from source
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."

REM Check for wheel in dist folder
set "WHEEL_FILE="
for %%f in ("%ROOT_DIR%\dist\docling_rag-*.whl") do set "WHEEL_FILE=%%f"

if defined WHEEL_FILE (
    echo Installing from wheel: %WHEEL_FILE%
    pip install "%WHEEL_FILE%"
) else (
    echo Installing from source...
    pip install "%ROOT_DIR%"
)

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed
    exit /b 1
)

echo.
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.
echo The 'rag' command is now available.
echo.
echo Quick start:
echo   rag init                    # First-time setup
echo   rag project list            # List projects
echo   rag ingestion start ./docs  # Ingest documents
echo   rag mcp serve               # Start MCP server
echo.
echo For help: rag --help
echo.

REM Verify installation
rag --help >nul 2>&1
if errorlevel 1 (
    echo WARNING: 'rag' command not found in PATH
    echo You may need to restart your terminal or add Python Scripts to PATH
    echo.
    echo Typical location: %USERPROFILE%\AppData\Local\Programs\Python\Python3XX\Scripts
)

endlocal
