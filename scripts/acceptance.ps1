# Windows acceptance run (PowerShell).  Usage:  powershell -ExecutionPolicy Bypass -File scripts/acceptance.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path .venv)) {
    py -3.10 -m venv .venv
    .\.venv\Scripts\python.exe -m ensurepip --upgrade
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt

$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONPATH = "src"

& $py scripts/check_gl_context.py
if ($LASTEXITCODE -ne 0) { throw "E_GL_CONTEXT: offscreen GL/FBO unavailable" }

& $py -m pcrecon generate
& $py -m pcrecon run --undo
& $py -m pcrecon crash-demo
& $py -m pcrecon hash-verify
& $py -m unittest discover -s tests
