@echo off
setlocal
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe call scripts\setup_windows.bat
.venv\Scripts\python.exe -m dns_local.upstream --zone data\upstream\v1.json --host 127.0.0.1 --port 15354
endlocal
