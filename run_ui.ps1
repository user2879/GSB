param(
    [int]$Revision = 2
)
$ErrorActionPreference = "Stop"
python -m meshwork.cli ui `
    --spec samples\sample_spec.json `
    --requests samples\requests.json `
    --ledger artifacts\ui.db `
    --snapshot-dir artifacts\ui-snapshots `
    --revision $Revision
