.PHONY: generate test all

generate:
	python3 -m driftbench generate --out samples

test:
	python3 -m unittest discover -s tests -v

all:
	python3 -m driftbench all --sample-dir samples --output-dir artifacts --snapshot-dir artifacts/snapshots
