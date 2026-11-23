$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m timbre_workbench demo --out examples/demo

if (Test-Path "examples/demo/.workbench") {
    Remove-Item -Recurse -Force "examples/demo/.workbench"
}

python -m timbre_workbench run `
    --input examples/demo/audio `
    --config examples/demo/config.json `
    --labels examples/demo/labels.json `
    --commands examples/demo/commands.json `
    --work examples/demo/.workbench `
    --report examples/demo/report.json `
    --html examples/demo/report.html `
    --responses examples/demo/responses.json

python -m timbre_workbench compare `
    --input examples/demo/audio `
    --config examples/demo/config.json `
    --labels examples/demo/labels.json `
    --commands examples/demo/commands.json `
    --output examples/demo/compare.json

python -m unittest discover -s tests -v

