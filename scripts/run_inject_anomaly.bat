@echo off
REM Inject one of: rollback_cycle | missing_renewal | snapshot_digest
setlocal
set MODE=%1
if "%MODE%"=="" set MODE=rollback_cycle
set WD=%~dp0..\workspace\inject_%MODE%
python -m lease_diag --workdir "%WD%" --inject %MODE% demo
endlocal
