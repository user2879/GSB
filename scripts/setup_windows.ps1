$ErrorActionPreference = "Stop"

$python = "py"
$pythonArgs = @("-3.10")
try {
    & py -3.10 -c "import sys; assert sys.version_info >= (3, 10); print(sys.version)"
} catch {
    try {
        & python -c "import sys; assert sys.version_info >= (3, 10); print(sys.version)"
        $python = "python"
        $pythonArgs = @()
    } catch {
        throw "Python 3.10+ is required. Install it from python.org or the Microsoft Store."
    }
}

if (-not (Test-Path ".venv")) {
    & $python @pythonArgs -m venv .venv
}

$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

& $venvPython -m rope_extrap generate --output samples
& $venvPython -m unittest discover -s tests -v
& $venvPython -m rope_extrap verify --output-dir artifacts

Write-Host "Environment ready. Activate with: .\.venv\Scripts\Activate.ps1"
