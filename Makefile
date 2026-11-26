PYTHON ?= python3
export PYTHONPATH := src

.PHONY: test acceptance demo clean-generated

test:
	$(PYTHON) -m pytest -q

acceptance:
	$(PYTHON) -m reco_shard_diag.cli acceptance --workdir work/acceptance --outdir artifacts/acceptance

demo:
	$(PYTHON) -m reco_shard_diag.cli init --workdir work/demo --force
	$(PYTHON) -m reco_shard_diag.cli run --workdir work/demo --fresh --outdir work/demo/artifacts

clean-generated:
	$(PYTHON) -c "import shutil; shutil.rmtree('work', ignore_errors=True); shutil.rmtree('.pytest_cache', ignore_errors=True)"
