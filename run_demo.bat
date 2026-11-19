@echo off
setlocal
py -3.11 app.py demo --out runtime\demo --report reports\demo_report.json --force
exit /b %errorlevel%
