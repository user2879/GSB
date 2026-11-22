@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
py -3.11 -m qplan.cli gui --spec examples\sample_spec.json --workspace workspace
endlocal
