@echo off
setlocal
py -3 -m deploy_workbench %*
exit /b %ERRORLEVEL%

