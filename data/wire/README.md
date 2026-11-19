# Deterministic wire samples

`python tools/make_fixtures.py` regenerates every file from local constants only. Each
sample's byte length and SHA-256 digest is recorded in `manifest.json`; the test suite
checks that malformed samples fail with `E_WIRE` at the documented offset.

The only TCP sample uses the DNS-over-TCP two-byte length prefix. No sample contains a
real domain or causes network access.

