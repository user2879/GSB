@echo off
REM Windows 一键演示：在仓库根目录双击或命令行执行
cd /d %~dp0\..
python -m shardbench.demo
pause
