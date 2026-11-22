@echo off
REM Create a local virtualenv and install NumPy. Requires Python 3.11 with Tcl/Tk.
setlocal
cd /d "%~dp0.."
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import tkinter; print('Tkinter OK')"
python -m pytest -q
endlocal
