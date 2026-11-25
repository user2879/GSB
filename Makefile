PYTHON ?= .venv/bin/python
TK_PYTHON ?= ./scripts/tk_python.sh

.PHONY: generate build report verify test gui gui-smoke

generate:
	$(PYTHON) scripts/generate_samples.py --root .

build:
	$(PYTHON) -m edge_ledger.cli --root . build

report:
	$(PYTHON) -m edge_ledger.cli --root . report

verify:
	$(PYTHON) -m edge_ledger.cli --root . verify --output artifacts/verification.json

test:
	$(TK_PYTHON) -m unittest discover -v

gui:
	$(TK_PYTHON) -m edge_ledger.cli --root . gui

gui-smoke:
	./scripts/run_headless_gui_check.sh
