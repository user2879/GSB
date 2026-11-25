PYTHON ?= python3
STATE ?= artifacts/demo

.PHONY: samples test verify demo diagnose env

samples:
	$(PYTHON) tools/generate_samples.py

test:
	$(PYTHON) -m pytest

verify:
	$(PYTHON) -m fixedtex verify --output-dir artifacts/verification

demo: samples
	$(PYTHON) -m fixedtex run \
		--config samples/config.json \
		--material samples/material.json \
		--commands samples/commands.jsonl \
		--state-dir $(STATE) \
		--fresh

diagnose:
	$(PYTHON) -m fixedtex diagnose \
		--config samples/config.json \
		--material samples/material.json \
		--state-dir $(STATE) \
		--output-dir artifacts/diagnostics \
		--quant-steps 4,8

env:
	$(PYTHON) -m fixedtex env
