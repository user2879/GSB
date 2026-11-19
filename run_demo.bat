@echo off
REM Deterministic terrain occlusion-culling demo (Windows entry point).
REM Pinned dependency versions live in requirements.txt.
python -m pip install --user -r requirements.txt
if errorlevel 1 exit /b 1
if not exist artifacts\demo mkdir artifacts\demo
python -m terrain_culling.demo --out artifacts/demo --frames 16
if errorlevel 1 exit /b 1
python -m pytest tests -q
