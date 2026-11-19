# Launch the interactive workbench window on a Windows desktop.
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) {
    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}
.\.venv\Scripts\python.exe -m subtitle_workbench.app
