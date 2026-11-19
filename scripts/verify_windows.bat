@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run scripts\setup_windows.ps1 first.
  exit /b 1
)
".venv\Scripts\python.exe" -m acoustic_workbench demo --output-dir artifacts\demo
".venv\Scripts\python.exe" -m acoustic_workbench verify --workspace artifacts\demo\run_a --full-compare --gl-probe
".venv\Scripts\python.exe" -m pytest -q
exit /b %ERRORLEVEL%
