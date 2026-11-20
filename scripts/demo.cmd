@echo off
REM Windows demo: full run, forced termination, recovery, verify, faults
set PYTHONPATH=.
set PY=.venv\Scripts\python
%PY% -m chaos.cli run --spec samples/experiment.json --ledger demo\ledger.sqlite --snapshot-dir demo\snapshots --log-dir demo\logs
%PY% -m chaos.cli verify --spec samples/experiment.json --ledger demo\ledger.sqlite --snapshot-dir demo\snapshots --log-dir demo\logs
%PY% -m chaos.cli run --spec samples/experiment.json --ledger demo\ledger_kill.sqlite --snapshot-dir demo\snapshots_kill --log-dir demo\logs_kill --kill-after 2
%PY% -m chaos.cli resume --spec samples/experiment.json --ledger demo\ledger_kill.sqlite --snapshot-dir demo\snapshots_kill --log-dir demo\logs_kill
%PY% -m chaos.cli ui --ledger demo\ledger_kill.sqlite
