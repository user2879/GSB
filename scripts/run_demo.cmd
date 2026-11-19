@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo Run scripts\setup_windows_offline.ps1 first.
  exit /b 2
)
".venv\Scripts\python.exe" -m audiofp demo --out artifacts\demo --seed 20261119
exit /b %ERRORLEVEL%
