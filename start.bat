@echo off
title FloodAI - Starting...
cd /d "%~dp0"

echo ============================================
echo   FloodAI - One Click Setup and Start
echo ============================================

IF NOT EXIST venv (
    echo [1/3] Creating virtual environment...
    python -m venv venv
)

echo [2/3] Installing required libraries...
call venv\Scripts\activate.bat
pip install -q -r requirements.txt

echo [3/3] Starting FloodAI...
echo.
echo Once you see "Starting FloodAI at http://127.0.0.1:5000",
echo open that link in your browser.
echo.
python app.py

pause
