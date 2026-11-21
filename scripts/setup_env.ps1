# Windows PowerShell: create a local venv and install numpy (Tkinter is bundled).
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import tkinter, sqlite3, numpy; print('environment ready')"
