PYTHON := .venv/bin/python

.PHONY: generate test verify demo

generate:
	$(PYTHON) tools/generate_samples.py --out samples --seed 20261120

test:
	$(PYTHON) -m unittest discover -v

verify: generate
	$(PYTHON) -m cm_fusion verify --samples samples --out reports/demo

demo: verify

