param(
    [string]$Python = "py",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

& $Python "-$PythonVersion" -c "import sys, numpy, pytest; assert sys.version_info[:2] == (3, 11); print(sys.version); print('numpy', numpy.__version__, 'pytest', pytest.__version__)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "-$PythonVersion" -m vc_workbench generate --dir samples --force
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "-$PythonVersion" -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "-$PythonVersion" tools/run_verification.py
exit $LASTEXITCODE
