param(
    [string]$Python = "py -3.11"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$WheelDir = Join-Path $Root "wheels"

if (-not (Test-Path $WheelDir)) {
    throw "Missing local wheel directory: $WheelDir"
}

$Wheels = Get-ChildItem -Path $WheelDir -Filter "numpy-*.whl"
if ($Wheels.Count -eq 0) {
    throw "No numpy wheel found in $WheelDir. Copy a Windows cp311/cp312 wheel there first."
}

Invoke-Expression "$Python -m pip install --no-index --find-links `"$WheelDir`" --only-binary=:all: --no-deps numpy"
Invoke-Expression "$Python -c `"import numpy; print('numpy', numpy.__version__)`""
