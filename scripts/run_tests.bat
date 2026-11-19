@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run scripts\setup_windows.ps1 first.
  exit /b 1
)
".venv\Scripts\python.exe" -m pytest -q
exit /b %ERRORLEVEL%
