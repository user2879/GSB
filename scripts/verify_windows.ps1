param(
  [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python --version
if (-not $SkipInstall) {
  python -m pip install -r requirements.txt
}
python run_demo.py
python -m unittest discover -s tests -v
python scripts/check_tkinter.py
