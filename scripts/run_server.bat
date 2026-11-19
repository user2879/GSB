@echo off
setlocal
cd /d "%~dp0\.."
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)
%PY% -m chunkdeps.cli check-env --json
if errorlevel 1 exit /b 1
%PY% -m chunkdeps.cli server --root workbench --listen-host 127.0.0.1 --listen-port 8765 --global-concurrency 4 --session-concurrency 2
exit /b %errorlevel%

