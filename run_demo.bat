@echo off
setlocal
python main.py generate --input samples\sample.json
python main.py solve --input samples\sample.json --state-dir state\demo --output out\demo
python main.py verify --input samples\sample.json --state-dir state\demo
endlocal
