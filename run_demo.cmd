@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  echo Missing .venv. Create it and install requirements.txt first.
  exit /b 20
)
.venv\Scripts\python.exe -m mocap_workbench demo --out workdir %*
exit /b %ERRORLEVEL%
