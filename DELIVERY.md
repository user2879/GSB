# Delivery Guide

完整设计与验收说明见 `docs/TECHNICAL.md` 和 `docs/VERIFICATION.md`。

```bash
python -m pip install numpy==1.26.4
python -m lease_arbitrator init-sample --dir samples/demo --force
python -m lease_arbitrator run --sample samples/demo --state-dir .runtime/demo
python -m lease_arbitrator verify --sample samples/demo --state-dir .runtime/demo
python -m lease_arbitrator ui --state-dir .runtime/demo --text
python -m unittest discover -s tests
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo.ps1
powershell -ExecutionPolicy Bypass -File scripts/recovery-check.ps1
```

稳定错误码：`E_SPEC`、`E_PORT`、`E_HANDSHAKE`、`E_FIELD`、`E_LEASE`、`E_VERIFY`、`E_UNKNOWN`、`E_UI`。
