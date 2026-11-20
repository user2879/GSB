@echo off
REM Requires the official Python installer with the tcl/tk (Tkinter) feature.
python -m elastic_scaler.cli --db data\workbench.sqlite3 --log data\workbench-events.jsonl init --spec samples\cluster_v1.json
python -m elastic_scaler.cli --db data\workbench.sqlite3 --log data\workbench-events.jsonl advance
python -m elastic_scaler.cli --db data\workbench.sqlite3 --log data\workbench-events.jsonl gui
