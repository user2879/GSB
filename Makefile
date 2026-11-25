.PHONY: setup test demo verify clean

setup:
	./scripts/bootstrap.sh

test:
	./scripts/bootstrap.sh
	.venv/bin/python -m pytest -q

demo:
	./scripts/bootstrap.sh
	.venv/bin/python -m shard_diag demo --root .

verify:
	./scripts/verify.sh

clean:
	./scripts/clean.sh

