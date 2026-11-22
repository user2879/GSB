@echo off
setlocal
cd /d "%~dp0"
if not exist "workbench\slow-sources.json" (
  py -3 -m slowdiag demo --workspace workbench
  if errorlevel 1 exit /b %errorlevel%
)
py -3 -m slowdiag gui --workspace workbench
exit /b %errorlevel%
