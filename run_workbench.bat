@echo off
setlocal
if "%~1"=="" (
    python -m h2burst gui samples\minimal.json --db snapshots.sqlite3
) else (
    python -m h2burst gui %*
)
endlocal
