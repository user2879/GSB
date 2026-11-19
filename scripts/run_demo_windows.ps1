# Repeatable demo on Windows: three ordered scenarios + optional fault injection.
$ErrorActionPreference = "Stop"
.\.venv\Scripts\python.exe scripts\generate_samples.py
.\.venv\Scripts\python.exe scripts\demo.py
# Fault injections (one at a time):
#   .\.venv\Scripts\python.exe scripts\demo.py --inject corrupt-scene
#   .\.venv\Scripts\python.exe scripts\demo.py --inject occupied-port
#   .\.venv\Scripts\python.exe scripts\demo.py --inject bad-snapshot
