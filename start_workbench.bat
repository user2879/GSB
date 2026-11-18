@echo off
setlocal
cd /d "%~dp0"
python -m slowquery_workbench gui
if errorlevel 1 (
  echo.
  echo Workbench exited with an error. Verify: python -c "import tkinter, sqlite3"
  pause
)
endlocal
