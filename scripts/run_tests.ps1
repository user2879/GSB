$ErrorActionPreference = "Stop"
$env:QT_QPA_PLATFORM = "offscreen"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
python -m unittest discover -v
