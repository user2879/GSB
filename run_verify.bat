@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run: powershell -ExecutionPolicy Bypass -File setup_windows.ps1
  exit /b 1
)
".venv\Scripts\python.exe" -m mocap_workbench verify --workspace . --frames all

