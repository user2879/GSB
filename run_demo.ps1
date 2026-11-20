$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python --version
python -m pip install -r requirements.txt
python demo.py @args
exit $LASTEXITCODE
