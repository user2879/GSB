@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
py -3.11 -m venv .venv
if errorlevel 1 goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -e .[test]
if errorlevel 1 goto :error
python -m haptic_diag.cli generate
if errorlevel 1 goto :error
python -m pytest -q
if errorlevel 1 goto :error
python -m haptic_diag.cli verify --sample cross_slice_boundary --protocol 2 --workspace workbench --output reports\verification-windows.json --batch-size 20
if errorlevel 1 goto :error
exit /b 0
:error
exit /b 1
