# Windows PowerShell bootstrap for the drift monitor component.
Write-Host "Creating virtual environment (.venv)..."
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r drift_monitor\requirements.txt
Write-Host "Running regression tests..."
Push-Location drift_monitor
..\.venv\Scripts\python -m pytest tests -q
Pop-Location
Write-Host "Running end-to-end demo..."
Push-Location drift_monitor
..\.venv\Scripts\python -m drift_monitor.interactive --config configs/default_monitor.yaml --tier small --output output/demo --command run
Pop-Location
