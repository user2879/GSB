@echo off
REM 全量单元与端到端测试
python -m unittest discover -s tests -v
