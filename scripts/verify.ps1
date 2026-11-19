$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
dotnet build .\LevelWorkbench.sln
dotnet run --project .\src\LevelWorkbench.SampleGen -- --output .\sample-data
dotnet run --project .\tests\LevelWorkbench.SelfTest

$runDir = Join-Path ([System.IO.Path]::GetTempPath()) ("level-workbench-verify-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $runDir | Out-Null
try {
  dotnet run --project .\src\LevelWorkbench.Cli -- validate `
    --scene .\sample-data\scene.json --records .\sample-data\records\mixed.json
  dotnet run --project .\src\LevelWorkbench.Cli -- replay `
    --scene .\sample-data\scene.json --records .\sample-data\records\mixed.json `
    --state (Join-Path $runDir "state") --report (Join-Path $runDir "replay-report.json")
  dotnet run --project .\src\LevelWorkbench.Cli -- pick `
    --scene .\sample-data\scene.json --script .\sample-data\pick-script.json `
    --state (Join-Path $runDir "pick-state") --out (Join-Path $runDir "picked-records.json")
  dotnet run --project .\src\LevelWorkbench.Cli -- replay `
    --scene .\sample-data\scene.json --records (Join-Path $runDir "picked-records.json") `
    --state (Join-Path $runDir "picked-replay-state")
}
finally {
  Remove-Item -Recurse -Force $runDir -ErrorAction SilentlyContinue
}
