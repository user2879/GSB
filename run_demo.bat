@echo off
setlocal

py -3.11 -m imputation_workbench generate --out samples\demo --seed 20261121 --rows 24
if errorlevel 1 exit /b %errorlevel%

py -3.11 -m imputation_workbench run --workspace samples\demo --state samples\demo\work --fresh
if errorlevel 1 exit /b %errorlevel%

py -3.11 -m imputation_workbench run --workspace samples\demo --state samples\demo\work
if errorlevel 1 exit /b %errorlevel%

py -3.11 -m imputation_workbench verify --workspace samples\demo --state samples\demo\work
if errorlevel 1 exit /b %errorlevel%

py -3.11 -m pytest -q

