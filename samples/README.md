# Deterministic offline samples

`data.sql` is the deterministic local data file. It uses arithmetic formulas and
recursive CTEs (no randomness and no network) to insert 2,000 customers and
20,000 orders. `schema.sql` defines the two domain tables. `queries.json` is the
query catalog. `dataset.json` contains the same seed/count manifest plus index
and threshold settings used by the loader.

`corrupt_data.sql` is used only by the explicit `E_DATA` injection path. It
violates the `orders` NOT NULL constraints and therefore fails in real SQLite
execution rather than through a hard-coded exception.
