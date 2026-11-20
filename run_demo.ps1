$ErrorActionPreference = "Stop"
python run_demo.py
$quantcalExit = $LASTEXITCODE
Write-Host "Exit code: $quantcalExit"
exit $quantcalExit
