@echo off
REM Start deterministic mock upstreams (127.0.0.1 high ports only).
setlocal
cd /d "%~dp0.."
python scripts\run_upstream.py --rules samples\rules_v1.json
endlocal
