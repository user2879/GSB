@echo off
REM Windows 一键：生成样本 -> 运行诊断 -> 启动视图
python -m pip install -r requirements.txt || goto :err
python -m mwb.cli generate --samples-dir samples || goto :err
python -m mwb.cli run --state-dir state --output-dir output || goto :err
python -m mwb.cli gui --output-dir output
exit /b %ERRORLEVEL%
:err
echo Workbench failed with business error, see JSON above.
exit /b 1
