@echo off
setlocal
cd /d "%~dp0"
python -m pip install -r requirements.txt
python -c "import numpy, sqlite3, tkinter; print('numpy', numpy.__version__); print('sqlite3', sqlite3.sqlite_version); print('tkinter', tkinter.TkVersion)"
exit /b %ERRORLEVEL%
