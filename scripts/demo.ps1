[CmdletBinding()]
param(
    [string]$Sample = "samples/demo",
    [string]$State = ".runtime/demo"
)

$ErrorActionPreference = "Stop"
python --version
python -m pip install numpy==1.26.4

python -m lease_arbitrator init-sample --dir $Sample --force
if (Test-Path $State) {
    Remove-Item -Recurse -Force $State
}

python -m lease_arbitrator run --sample $Sample --state-dir $State
python -m lease_arbitrator verify --sample $Sample --state-dir $State
python -m lease_arbitrator ui --state-dir $State --text

Write-Host "Tkinter window: python -m lease_arbitrator ui --state-dir $State"
