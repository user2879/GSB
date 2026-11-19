@echo off
setlocal
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe call scripts\setup_windows.bat
.venv\Scripts\python.exe -m dns_local.cli --zone data\zones\lab.v1.json --udp-port 15353 --tcp-port 15353 --upstream-port 15354 --auto-start --snapshot-dir snapshots
endlocal
