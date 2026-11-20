@echo off
setlocal
cd /d "%~dp0"

python -m meshgov doctor --workspace . --report artifacts\reports\doctor.json
if errorlevel 1 exit /b 1

python scripts\check_ports.py --config samples\upstreams.json
if errorlevel 1 exit /b 1

python -m meshgov run ^
  --rules samples\rules.json ^
  --requests samples\requests.json ^
  --db artifacts\demo.db ^
  --log artifacts\logs\demo.jsonl ^
  --snapshots artifacts\snapshots ^
  --report artifacts\reports\demo.json ^
  --round-id demo-%RANDOM% ^
  --upstream-config samples\upstreams.json ^
  --start-upstreams
if errorlevel 1 exit /b 1

python -m meshgov gui --snapshot artifacts\snapshots\latest.json

