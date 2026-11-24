@echo off
setlocal
cd /d "%~dp0"
python -c "import numpy, tkinter" >nul 2>nul
if errorlevel 1 python -m pip install -r requirements.txt
python -m leasecache gui --spec samples\demo_spec.json --db reports\gui.sqlite --workspace reports\gui-workspace
