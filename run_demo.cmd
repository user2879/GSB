@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m pip install -r requirements.txt
if errorlevel 1 goto :error
py -3.12 scripts\probe_ports.py --config samples\workload.json
if errorlevel 1 goto :error
py -3.12 -m speaker_diag demo --root .
if errorlevel 1 goto :error
echo.
echo Verification report: %CD%\artifacts\verification.json
exit /b 0

:error
echo Demo failed with exit code %errorlevel%.
exit /b %errorlevel%

