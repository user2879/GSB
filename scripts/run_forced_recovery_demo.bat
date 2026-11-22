@echo off
REM 1) force-terminate mid-batch, 2) resume from durable cursor, 3) verify replay
setlocal
set WD=%~dp0..\workspace\kill_demo
python -m lease_diag --workdir "%WD%" generate
python -m lease_diag --workdir "%WD%" --hard-exit-batch 2 run
echo Process exited with code 87 (simulated SIGKILL). Resuming...
python -m lease_diag --workdir "%WD%" --resume run
python -m lease_diag --workdir "%WD%" recover
endlocal
