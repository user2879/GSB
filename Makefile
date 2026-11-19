PYTHON_BIN ?= .venv-local/bin/python

.PHONY: env samples test demo verify

env:
	bash scripts/bootstrap_env.sh

samples:
	$(PYTHON_BIN) scripts/generate_samples.py

test:
	$(PYTHON_BIN) -m unittest discover -s tests -v

demo:
	$(PYTHON_BIN) scripts/run_demo.py

verify:
	bash scripts/verify.sh
