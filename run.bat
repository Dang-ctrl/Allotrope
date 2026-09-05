@echo off
REM Allotrope Command Center launcher (Windows)
REM One command, no manual setup: on first run this creates the Python
REM venv, installs the backend, and runs npm install for the frontend --
REM all three used to be separate steps a user had to remember to do
REM (and get right, in order) before this script would even start.
REM Starts the FastAPI backend and the Vite frontend dev server, each in
REM its own window, then opens the Command Center in your default browser.

setlocal

set REPO_DIR=%~dp0
cd /d "%REPO_DIR%"

REM Sanity check: run.bat only works from the repo root. A common mistake
REM is running it from a parent folder after `git clone` created a nested
REM directory of the same name (e.g. Allotrope\Allotrope) -- pyproject.toml
REM living next to this script is what actually proves we're in the right
REM place, not just this script's own presence.
if not exist "%REPO_DIR%pyproject.toml" (
    echo [!] pyproject.toml not found next to run.bat -- this doesn't look
    echo     like the repo root. If your clone created a nested folder
    echo     ^(e.g. Allotrope\Allotrope^), cd into that inner folder and run
    echo     run.bat from there instead.
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo [!] python was not found on PATH. Install Python 3.11+ from
    echo     https://www.python.org/downloads/ -- check the box to add
    echo     python.exe to PATH during install -- then re-run this script.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [!] npm was not found on PATH. Install Node.js 22+ from
    echo     https://nodejs.org/, then re-run this script.
    pause
    exit /b 1
)

if not exist "%REPO_DIR%.venv\Scripts\activate.bat" (
    echo [*] First run: creating the Python virtual environment ^(.venv^) ...
    python -m venv "%REPO_DIR%.venv"
    if errorlevel 1 (
        echo [!] Failed to create .venv. See the error above.
        pause
        exit /b 1
    )
)

echo [*] Checking backend dependencies ...
call "%REPO_DIR%.venv\Scripts\activate.bat"
python -c "import fastapi, uvicorn" >nul 2>nul
if errorlevel 1 (
    echo [*] Installing backend dependencies ^(pip install -e ".[api]"^) -- this
    echo     can take a few minutes the first time ...
    pip install -e ".[api]"
    if errorlevel 1 (
        echo [!] pip install failed. See the error above.
        pause
        exit /b 1
    )
)

if not exist "%REPO_DIR%frontend\node_modules" (
    echo [*] Installing frontend dependencies ^(npm install^) -- this can take
    echo     a few minutes the first time ...
    pushd "%REPO_DIR%frontend"
    call npm install
    if errorlevel 1 (
        popd
        echo [!] npm install failed. See the error above.
        pause
        exit /b 1
    )
    popd
)

REM The four simulation-control endpoints require an API key (see
REM docs/api.md's "Authentication"). Generate one per launch and hand the
REM same value to both the backend and the frontend, rather than leaving
REM either side to fall back to a key the other side doesn't know --
REM without this, the Command Center's start/stop/reset/step buttons would
REM all get a 401.
for /f %%k in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString('N')"') do set ALLOTROPE_API_KEY=%%k
(
    echo VITE_API_BASE_URL=http://localhost:8000
    echo VITE_API_KEY=%ALLOTROPE_API_KEY%
) > "%REPO_DIR%frontend\.env.local"

REM Relative paths below (not %REPO_DIR%-qualified): `start` inherits this
REM script's current directory for the new window, and a quoted path here
REM would nest inside the /k string's own quotes -- a classic cmd.exe
REM quoting trap. `cd /d "%REPO_DIR%"` above already put us in the right place.
echo Starting backend (uvicorn) on http://127.0.0.1:8000 ...
start "Allotrope backend" cmd /k "set ALLOTROPE_API_KEY=%ALLOTROPE_API_KEY% && .venv\Scripts\activate.bat && uvicorn allotrope.api.app:app --reload --host 127.0.0.1 --port 8000"

echo Starting frontend (vite) on http://127.0.0.1:5173 ...
start "Allotrope frontend" cmd /k "cd /d "%REPO_DIR%frontend" && npm run dev -- --host 127.0.0.1 --port 5173"

echo Waiting for the frontend to come up ...
timeout /t 6 /nobreak >nul

start "" "http://127.0.0.1:5173"

echo.
echo Allotrope is starting in two separate windows (backend + frontend).
echo Close those windows (or Ctrl+C in each) to stop the servers.
endlocal
