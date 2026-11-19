@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
py -3.11 -m story_workbench gui --spec artifacts\demo\samples\normal.yaml --sequence normal_path
