$ErrorActionPreference = "Stop"
python -m unittest discover -s tier_mover/tests -p "test_*.py" -v
