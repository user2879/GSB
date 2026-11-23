@echo off
setlocal
cd /d "%~dp0"
python scripts\generate_demo.py >nul
python -c "from pathlib import Path; [p.unlink() for p in Path('artifacts').glob('ui_demo.sqlite*') if p.is_file()]"
python -m leasebench run --declaration samples\declarations_v1.json --events samples\events.ndjson --db artifacts\ui_demo.sqlite --limit-events 6 >nul
if errorlevel 1 exit /b %errorlevel%
python -m leasebench run --declaration samples\declarations_v2.json --events samples\events.ndjson --db artifacts\ui_demo.sqlite >nul
if errorlevel 1 exit /b %errorlevel%
python -m leasebench ui --db artifacts\ui_demo.sqlite
