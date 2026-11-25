$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m qos_diag sample --workspace .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m qos_diag drill --workspace . --output artifacts/reports/acceptance-drill.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m qos_diag verify --spec-dir samples/edited_param --ledger artifacts/ledger/qos.db
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
