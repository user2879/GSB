@echo off
setlocal
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo Missing .venv. See README.md for environment setup. 1>&2
  exit /b 1
)
.venv\Scripts\python.exe -m tactical_path demo --config config\demo.yaml --workspace workspace\demo
if errorlevel 1 exit /b %errorlevel%
endlocal

