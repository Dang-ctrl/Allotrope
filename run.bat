@echo off
REM Allotrope Command Center launcher (Windows)
REM Starts the FastAPI backend and the Vite frontend dev server, each in
REM its own window, then opens the Command Center in your default browser.
REM
REM First-time setup (run once):
REM   python -m venv .venv
REM   .venv\Scripts\activate
REM   pip install -e ".[api]"
REM   cd frontend
REM   npm install
REM   copy .env.example .env.local
REM   cd ..

setlocal

set REPO_DIR=%~dp0
cd /d "%REPO_DIR%"

if not exist ".venv\Scripts\activate.bat" (
    echo [!] .venv not found. Run the first-time setup in this file's header comment first.
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [!] frontend\node_modules not found. Run "npm install" in frontend\ first.
    pause
    exit /b 1
)

echo Starting backend (uvicorn) on http://127.0.0.1:8000 ...
start "Allotrope backend" cmd /k ".venv\Scripts\activate.bat && uvicorn allotrope.api.app:app --reload --host 127.0.0.1 --port 8000"

echo Starting frontend (vite) on http://127.0.0.1:5173 ...
start "Allotrope frontend" cmd /k "cd /d "%REPO_DIR%frontend" && npm run dev -- --host 127.0.0.1 --port 5173"

echo Waiting for the frontend to come up ...
timeout /t 6 /nobreak >nul

start "" "http://127.0.0.1:5173"

echo.
echo Allotrope is starting in two separate windows (backend + frontend).
echo Close those windows (or Ctrl+C in each) to stop the servers.
endlocal
