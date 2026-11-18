@echo off
REM Launch the Tkinter operator console (python.org Windows installer ships Tk).
cd /d "%~dp0\.."
python -m pool_workbench.gui
pause
