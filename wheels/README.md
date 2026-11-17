# Offline NumPy wheel

The forwarding engine, tests, and demos do not import NumPy at runtime.  To
follow the target Windows preparation step without network access, place one
official NumPy wheel matching the installed Python ABI in this directory, for
example:

```text
wheels/numpy-2.1.3-cp311-cp311-win_amd64.whl
```

Then run from PowerShell:

```powershell
.\scripts\install_numpy_offline.ps1 -Python "py -3.11"
```

The install command uses `--no-index`, `--only-binary=:all:`, and `--no-deps`,
so it cannot fetch NumPy or any other package from the network.
