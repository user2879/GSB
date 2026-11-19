@echo off
setlocal
cd /d "%~dp0"
python -m shardrep init-sample --root .
python -m shardrep gui --spec samples\spec.json

