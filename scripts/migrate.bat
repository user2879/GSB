@echo off
REM Migrate the sample chain into a local SQLite store and write a snapshot.
setlocal
cd /d "%~dp0.."
if not exist work mkdir work
python -m snapshot_migration.cli migrate --chain samples\snapshot_chain.json --db work\store.sqlite3 --snapshot work\snapshot.json
endlocal
