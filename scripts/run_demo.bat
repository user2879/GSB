@echo off
REM End-to-end deterministic demo: dispatch, breaker, rule evolution, replay.
setlocal
cd /d "%~dp0.."
python -m meshbench.cli demo --rules samples\rules_v1.json --rules-v2 samples\rules_v2.json
endlocal
