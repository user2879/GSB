@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Windows Python launcher "py" was not found.
  exit /b 2
)

py -3 -c "import sys, numpy; assert sys.version_info >= (3, 11); print(sys.version.split()[0], 'numpy', numpy.__version__)"
if errorlevel 1 (
  echo ERROR: Python 3.11+ with NumPy is required.
  exit /b 2
)

py -3 -m unittest discover -s tests -v > validation_tests.txt 2>&1
if errorlevel 1 (
  type validation_tests.txt
  exit /b 1
)

py -3 -m aqeval demo --workspace validation_workspace > validation_output.txt 2>&1
if errorlevel 1 (
  type validation_output.txt
  exit /b 1
)

py -3 -m aqeval generate --workspace sample_workspace >> validation_output.txt 2>&1
if errorlevel 1 (
  type validation_output.txt
  exit /b 1
)

py -3 -m aqeval run --workspace sample_workspace >> validation_output.txt 2>&1
if errorlevel 1 (
  type validation_output.txt
  exit /b 1
)

type validation_output.txt
exit /b 0
