.PHONY: install samples validate test verify run commands gui clean

PYTHON ?= python3
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

install:
	bash scripts/install_deps.sh

samples:
	$(PYTHON) -m condrender.cli generate-samples --root .

validate: samples
	$(PYTHON) -m condrender.cli validate --root .

test:
	$(PYTHON) -m unittest discover -s tests -v

verify:
	bash scripts/verify.sh

commands:
	$(PYTHON) -m condrender.cli run-commands --root .

run:
	$(PYTHON) -m condrender.cli render --root .

gui:
	bash -c 'source scripts/tk_vendor_env.sh && $(PYTHON) -m condrender.cli gui --root .'

clean:
	rm -rf artifacts/command-acceptance artifacts/render artifacts/snapshots artifacts/verification artifacts/validation
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
