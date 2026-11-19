[CmdletBinding()]
param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv-win"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$Wheelhouse = Join-Path $Root "wheelhouse"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Windows virtual environment at $Venv"
    if ($PythonLauncher -eq "py") {
        & py "-$PythonVersion" -m venv $Venv
    } else {
        & $PythonLauncher -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create the Python $PythonVersion virtual environment. Install CPython $PythonVersion for Windows first."
    }
}

$DetectedVersion = & $VenvPython -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ($DetectedVersion.Trim() -ne $PythonVersion) {
    throw "Expected Python $PythonVersion, found $DetectedVersion"
}

$Wheel = Get-ChildItem -Path $Wheelhouse -Filter "numpy-1.26.4-cp311-cp311-win_amd64.whl" | Select-Object -First 1
if (-not $Wheel) {
    throw "Windows NumPy wheel is missing from $Wheelhouse"
}

try {
    & $VenvPython -m pip install --disable-pip-version-check --no-index --only-binary=:all: --find-links $Wheelhouse "numpy==1.26.4"
    if ($LASTEXITCODE -ne 0) {
        throw "pip offline installation failed"
    }
} catch {
    Write-Warning "pip could not install NumPy; extracting the bundled wheel directly."
    $SitePackages = & $VenvPython -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    & $VenvPython -m zipfile -e $Wheel.FullName $SitePackages
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to extract the bundled NumPy wheel"
    }
}

& $VenvPython -c "import numpy as np; print('NumPy', np.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "NumPy import verification failed"
}

Push-Location $Root
try {
    & $VenvPython "scripts\generate_samples.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Deterministic sample generation failed"
    }
} finally {
    Pop-Location
}

Write-Host "Offline environment ready: $VenvPython"

