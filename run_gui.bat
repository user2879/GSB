@echo off
REM Launch the interactive timeline GUI (Windows)
setlocal
cd /d "%~dp0"
python -m rts_replay --workspace workspace generate
python -m rts_replay --workspace workspace gui
endlocal
