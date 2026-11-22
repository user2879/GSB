@echo off
REM Launch the Tkinter diagnostic console (requires a Windows desktop session).
setlocal
cd /d "%~dp0.."
if not exist work mkdir work
python -m snapshot_migration.cli ui --chain samples\snapshot_chain.json --db work\store.sqlite3
endlocal
