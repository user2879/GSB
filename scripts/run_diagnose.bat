@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run scripts\setup_windows.ps1 first.
  exit /b 1
)
if not exist "artifacts\demo\run_a\snapshots\LATEST" (
  call scripts\run_demo.bat
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m acoustic_workbench diagnose --workspace artifacts\demo\run_a --port 8765 --open-browser
exit /b %ERRORLEVEL%
