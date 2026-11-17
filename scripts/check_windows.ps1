$ErrorActionPreference = "Stop"

if (Get-Command py -ErrorAction SilentlyContinue) {
    $python = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = "python"
} else {
    throw "Python 3.10+ was not found. Install Python and ensure 'py' or 'python' is on PATH."
}

& $python --version
& $python -m pip install -U pip
& $python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
& $python -m pytest -q
& $python -m softmax_layernorm.cli demo --output output/windows-demo

Write-Host "Windows check completed. See output/windows-demo/verification.json"
