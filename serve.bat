@echo off
REM Serve Script - Activate virtual environment and start RAG server
REM Usage: serve.bat

echo ========================================
echo RAG Server Startup Script
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

REM Start the RAG server
echo Starting RAG server...
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo Server is starting...
echo ========================================
echo.

rag serve

REM If the server stops, show a message
echo.
echo ========================================
echo Server stopped
echo ========================================
echo.
pause
