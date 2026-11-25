# Ledger artifacts

`demo_report.json` is the machine-readable acceptance evidence from the latest
local run. SQLite files (`*.sqlite`, `*.sqlite-wal`, `*.sqlite-shm`) are runtime
artifacts and are ignored by Git. Remove them with `scripts/clean.sh` before a
new reproducibility run.

The ledger uses one SQLite database per demo layout. Each completed probe batch
is written with a single `BEGIN IMMEDIATE` transaction. A separate batch intent
row is written before probing so a forced child termination can be recovered as
an `unknown` batch that must be rerun.

