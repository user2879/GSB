@echo off
REM Windows entry point for the repeatable acceptance demo.
python -m pip install -r requirements.txt
python scripts\generate_samples.py --out samples
python -m diagconsole demo --crash
