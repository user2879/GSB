@echo off
setlocal
if "%~1"=="" (
  echo usage: launch_ui.bat ^<workspace-directory^>
  exit /b 1
)
python -m workbench.cli ui --workspace %1
exit /b %errorlevel%
