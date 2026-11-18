$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$Python = ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run scripts\setup.ps1 first"
}
& $Python -m unittest discover -v tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\validate.py
exit $LASTEXITCODE

