# Environment Record

The implementation is pure Python plus NumPy. It does not need a GPU, network
access, Pillow, OpenCV, Docker, or an external dataset.

## Verified local environment

- OS: Linux (WSL2 Ubuntu 22.04 workspace)
- Python: `3.11.0rc1`
- NumPy: `1.26.4`
- pytest: `8.3.3`
- Minimum supported Python: `3.10`
- Minimum supported NumPy: `1.24`

`requirements.txt` pins the NumPy version used for the recorded verification.

```bash
python3 -m pip install -r requirements.txt
```

## Determinism assumptions

- PNG generation uses fixed seeds and stable global-coordinate formulas.
- All JSON hashing uses sorted keys and compact UTF-8 serialization.
- Every stochastic operation is seeded by an integer from sample or patch
  configuration.
- Classification and fusion use NumPy on the CPU with the same accumulation
  order regardless of worker count.
- Tile scheduling, result commit order, budget deduction, region ordering, and
  failure ordering are explicitly sorted; thread completion order is ignored.
