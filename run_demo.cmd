@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul
  if errorlevel 1 (
    python -m venv .venv
  ) else (
    py -3 -m venv .venv
  )
  if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :error

".venv\Scripts\python.exe" -m spatial_diag run --workspace artifacts
if errorlevel 1 goto :error

echo.
echo Diagnostics committed under artifacts\.
echo Verify with: .venv\Scripts\python.exe -m spatial_diag verify --workspace artifacts
exit /b 0

:error
echo Demo failed with exit code %errorlevel%.
exit /b %errorlevel%
