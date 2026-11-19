$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (!(Test-Path .\sample-data\scene.json)) {
  & .\scripts\generate-samples.ps1
}
dotnet run --project .\src\LevelWorkbench.Desktop -- `
  --scene .\sample-data\scene.json `
  --records .\sample-data\records\mixed.json `
  --script .\sample-data\pick-script.json `
  --state .\workbench-state `
  --out .\workbench-output\picked-records.json
