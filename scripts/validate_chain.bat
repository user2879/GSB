@echo off
setlocal
cd /d "%~dp0.."
python -m snapshot_migration.cli validate --chain samples\snapshot_chain.json
endlocal
