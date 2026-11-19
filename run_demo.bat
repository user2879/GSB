@echo off
REM Windows 一键复现实验：生成确定性样本 -> 端到端实验 -> 单元测试
setlocal
cd /d %~dp0
python --version
python data\generate_sample.py
python run_demo.py
python -m unittest tests.test_kvstore -v
endlocal
