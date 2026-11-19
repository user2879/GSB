@echo off
setlocal
cd /d "%~dp0\.."
python scripts\inject_faults.py samples\demo_scene.json --workspace "%CD%\workspaces\inject-run"
endlocal
