@echo off
setlocal
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe call scripts\setup_windows.bat
start "Loopback upstream simulator" .venv\Scripts\python.exe -m dns_local.upstream --zone data\upstream\v1.json --port 15354
.venv\Scripts\python.exe -m dns_local.gui --zone data\zones\lab.v1.json --udp-port 15353 --tcp-port 15353 --upstream-port 15354
taskkill /FI "WINDOWTITLE eq Loopback upstream simulator*" /T /F >nul 2>nul
endlocal
