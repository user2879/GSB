@echo off
REM Launch the Tkinter arbitration workbench on Windows.
REM Requires the standard python.org CPython (Tkinter bundled) and NumPy.
python -m pip install numpy
cd /d "%~dp0"
python -m arbworkbench --gui
