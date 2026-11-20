@echo off
REM 运行全部单元/端到端测试
cd /d %~dp0
python -m unittest discover -s tests -v
