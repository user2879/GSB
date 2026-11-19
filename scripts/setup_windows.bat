@echo off
setlocal
cd /d "%~dp0.."
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" was not found. Install Python 3.11+ with Tcl/Tk support.
  exit /b 2
)
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv || exit /b 2
)
.venv\Scripts\python.exe -m pip install --upgrade pip || exit /b 2
.venv\Scripts\python.exe -m pip install -r requirements.txt || exit /b 2
echo Environment ready. Tkinter is provided by the official Python installer.
endlocal
