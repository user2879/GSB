# Windows one-shot environment setup (run from the repository root).
# Requires the official python.org Python 3.11+ installer, which bundles Tk/Tkinter.
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) {
    py -3.11 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import sys, numpy, tkinter; print(sys.version); print('numpy', numpy.__version__); print('Tk', tkinter.TkVersion)"
Write-Host "Environment ready. Run: .\.venv\Scripts\python.exe scripts\demo.py"
