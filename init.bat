@echo off
setlocal
cd /d "%~dp0"
python -m batch_audit.cli init --workspace workspace --seed 20261120 --samples 12 --input-dim 6 --hidden-dim 8 --slots 0,1,2,3,4 %*
exit /b %ERRORLEVEL%
