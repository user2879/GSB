@echo off
setlocal
if not exist samples\sample.json python main.py generate --input samples\sample.json
python main.py gui --input samples\sample.json --state-dir state\workbench
endlocal
