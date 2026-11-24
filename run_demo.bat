@echo off
setlocal
cd /d "%~dp0"
python -c "import numpy, tkinter" >nul 2>nul
if errorlevel 1 (
  echo Installing runtime dependency NumPy...
  python -m pip install -r requirements.txt
)
python -c "import tkinter" >nul 2>nul
if errorlevel 1 (
  echo ERROR: this Python distribution does not include Tkinter.
  echo Reinstall Python 3.11+ with the "tcl/tk and IDLE" option.
  exit /b 2
)
python -m leasecache demo --root reports\acceptance
if errorlevel 1 exit /b %errorlevel%
echo.
echo Acceptance report: reports\acceptance\latest.json
echo Opening the Tkinter console...
python -m leasecache gui --spec samples\demo_spec.json --db reports\gui.sqlite --workspace reports\gui-workspace
