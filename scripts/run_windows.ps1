param(
    [string]$Config = "config/default.json",
    [string]$Commands = "scenarios/incremental_sequence.jsonl"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"

py -3.11 -c "import numpy, PIL; print('numpy', numpy.__version__, 'Pillow', PIL.__version__)"
py -3.11 -m image_edit_workbench run --root . --config $Config --commands $Commands --reset-state

