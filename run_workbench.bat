@echo off
REM Local Windows entry point. Requires Python 3.10+ with NumPy installed:
REM   py -m pip install numpy
REM Tkinter ships with the standard python.org Windows installer.
setlocal
if not exist samples\cluster.json py -m workbench.cli init-samples
py -m workbench.cli --db data\workbench.db --log logs\workbench.jsonl run samples\cluster.json
py -m workbench.cli --db data\workbench.db --log logs\workbench.jsonl replay samples\cluster.json
py -m workbench.cli --db data\workbench.db --log logs\workbench.jsonl gui
endlocal
