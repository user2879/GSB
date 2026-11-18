$ErrorActionPreference = "Stop"

python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m kg_workbench --config config.json --fresh
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
