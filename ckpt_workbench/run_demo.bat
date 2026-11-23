@echo off
REM Repeatable local demo for Windows (no network, no containers).
setlocal
cd /d %~dp0
python --version
python -c "import numpy, sqlite3, tkinter; print('deps ok')"
python -m workbench.cli demo --root out\demo --decl samples\declaration.json --changes samples\change_train.json samples\change_partition.json samples\change_cursor.json
endlocal

