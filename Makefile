.PHONY: generate test validate run recover clean-runtime

generate:
	python3 scripts/generate_samples.py

test:
	python3 -m unittest discover -s tests -v

validate:
	python3 -m intentlab validate

run:
	python3 -m intentlab run

recover:
	python3 -m intentlab recover

clean-runtime:
	find snapshots -mindepth 1 -delete
	find reports -mindepth 1 -delete

