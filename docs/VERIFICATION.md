# 验证记录与独立校验命令

## 环境
- Python 3.11，依赖 `numpy`、`pytest`；标准库 `sqlite3` 必选，`tkinter` 可选（缺失自动降级无头）。
- 全程仅使用 `127.0.0.1` 高位端口，无网络、账号、GPU、外部数据集。

## 一键演示
```bash
bash scripts/demo.sh            # Linux/macOS
powershell -ExecutionPolicy Bypass -File scripts/demo.ps1   # Windows
```

## 独立校验命令（不依赖主流程内存）
```bash
# 1) 强校验
python -m faultisolation.cli --spec samples/small/spec-small.json \
  --session samples/small/session.json validate

# 2) 路径选择理由/规模参数/工作量
python -m faultisolation.cli --spec samples/small/spec-small.json \
  --session samples/small/session.json plan

# 3) 三路径逐字节一致 + 槽位唯一 + 账本重放比对
python -m faultisolation.cli --spec samples/small/spec-small.json \
  --session samples/small/session.json verify --ledger artifacts/small-live.sqlite

# 4) 整数定点诊断与误差界
python -m faultisolation.cli --spec samples/small/spec-small.json \
  --session samples/small/session.json diagnose

# 5) 强制终止恢复
python -c "from faultisolation.recovery_demo import run_recovery_demo; \
print(run_recovery_demo('samples/small/spec-small.json','samples/small/session.json','artifacts/recovery-ledger.sqlite'))"

# 6) 测试
python -m pytest tests -q
```

## 验收对照
- 两档样本（小 48 事件 / 中 160 事件）三路径 canonical JSON 的 SHA-256 完全相同。
- 实时回环遥测全部 `matched=true`，且 RTT 为真实 TCP 往返观测值。
- 槽位唯一性校验 `violations=[]`。
- 强制终止后 `unknown_from_seq` 正确指向下一未提交事件，重跑后账本摘要校验通过。
