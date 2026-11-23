param(
    [string]$Workspace = "."
)

$ErrorActionPreference = "Stop"
Write-Host "Python:"
python --version
Write-Host "NumPy:"
python -c "import numpy; print(numpy.__version__)"
Write-Host "Project import and sample validation:"
python (Join-Path $PSScriptRoot "selfcheck.py") --workspace $Workspace
if ($LASTEXITCODE -ne 0) {
    throw "Self-check failed with exit code $LASTEXITCODE"
}

