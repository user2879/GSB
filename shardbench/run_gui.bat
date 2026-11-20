@echo off
REM Windows 启动 Tkinter 诊断界面
cd /d %~dp0\..
python -m shardbench --workspace "%CD%" gui
pause
