@echo off
REM Ingestion Script - Activate virtual environment and run document ingestion
REM Usage: ingest.bat [optional_path]
REM Example: ingest.bat "C:\my\documents"

echo ========================================
echo Document Ingestion Script
echo ========================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment
    echo Make sure .venv exists in the current directory
    pause
    exit /b 1
)

echo Virtual environment activated
echo.

REM Check if a path was provided as argument
if "%~1"=="" (
    REM No argument provided, use default docs folder
    echo Running: rag ingest docs
    rag ingest docs
) else (
    REM Use provided path
    echo Running: rag ingest "%~1"
    rag ingest "%~1"
)

REM Check if ingestion was successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Ingestion completed successfully!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo ERROR: Ingestion failed
    echo ========================================
)

echo.
pause
