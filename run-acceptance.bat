@echo off
REM Windows entry point for the acceptance suite (no Docker, no third-party packages).
set PYTHONPATH=%~dp0src
python --version
python -m unittest discover -s tests -v
python scripts\acceptance.py
