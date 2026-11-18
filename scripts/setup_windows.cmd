@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

echo [1/4] Checking python...
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: "python" is not on PATH. Install Python 3.11+ and enable "Add python.exe to PATH".
  exit /b 1
)
python --version
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 (
  echo ERROR: Python 3.11 or newer is required.
  exit /b 1
)

echo [2/4] Creating .venv...
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 exit /b 1

echo [3/4] Installing numpy and pytest...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 exit /b 1

echo [4/4] Checking SQLite...
".venv\Scripts\python.exe" -c "import sqlite3, numpy, pytest; print('python', __import__('sys').version); print('sqlite', sqlite3.sqlite_version); print('numpy', numpy.__version__); print('pytest', pytest.__version__)"
if errorlevel 1 exit /b 1
echo Setup complete.
