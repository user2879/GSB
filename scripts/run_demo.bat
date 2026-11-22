@echo off
REM Repeatable local demo on Windows; only Python 3.11+ standard library.
setlocal
set WD=%~dp0..\workspace\demo
python -m lease_diag --workdir "%WD%" demo %*
endlocal
