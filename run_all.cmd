@echo off
REM Windows 一键验收：建立虚拟环境 -> 安装依赖 -> 测试 -> 三类报告
setlocal
if not exist .venv (
    py -3.11 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
set PYTHONPATH=.
python -m unittest discover -s tests -v
if errorlevel 1 exit /b 1
python scripts\run_demo.py --workspace artifacts\persist --reports artifacts\reports
if errorlevel 1 exit /b 1
python scripts\crash_recovery_demo.py --workspace artifacts\persist-crash --reports artifacts\reports
if errorlevel 1 exit /b 1
python scripts\error_chain_demo.py
if errorlevel 1 exit /b 1
python scripts\reproducibility_check.py
endlocal
