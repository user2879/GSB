@echo off
REM Windows 本地演示：先确认 Python 3.11+，再安装 NumPy 并运行演示
python --version
python -m pip install -r requirements.txt
python -m coldstart_diag.demo
pause
