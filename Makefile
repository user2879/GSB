PYTHON ?= python

.PHONY: venv xvfb test sample run replay crash clean-test

venv:
	@test -d .venv || (command -v virtualenv >/dev/null && virtualenv -p /usr/bin/python3.11 .venv || python3 -m venv .venv)
	.venv/bin/pip install numpy pytest

xvfb:
	bash scripts/start-xvfb.sh

test:
	bash -c 'source scripts/env.sh && export DISPLAY=$$SLO_DISPLAY && python -m pytest -q'

sample:
	bash -c 'source scripts/env.sh && python -m slo_burn.cli sample samples/demo'

run:
	bash -c 'source scripts/env.sh && python -m slo_burn.cli run samples/demo --ledger artifacts/run/ledger.db'

replay:
	bash -c 'source scripts/env.sh && python -m slo_burn.cli replay samples/demo --ledger artifacts/run/ledger.db'

crash:
	bash -c 'source scripts/env.sh && python scripts/verify_crash_recovery.py crash samples/demo artifacts/crash || true'
	bash -c 'source scripts/env.sh && python scripts/verify_crash_recovery.py recover samples/demo artifacts/crash'
