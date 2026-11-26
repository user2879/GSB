# VDU 定点语音对话情感流水线

确定性、纯整数定点的本地合成对话音频情感证据流水线，仅用 Python 标准库。

## 快速开始
```
python3 -m vdu generate            # 生成 samples/ 与 configs/
python3 -m vdu run --tier coarse   # 运行（自动 rebuild/incremental/resume）
python3 -m vdu run --tier fine
python3 -m unittest                # 测试前清空 snapshots/
bash run_demo.sh                   # 端到端示例（coarse+fine）
```

## 强制终止与恢复（仅 kill + 整数信号）
```
bash scripts/control_vdu.sh start          # 后台启动并记录 pid
bash scripts/control_vdu.sh term           # SIGTERM，可处理退出并写 exit_report
bash scripts/control_vdu.sh kill           # SIGKILL，强制终止
bash scripts/control_vdu.sh resume         # 仅依据持久化快照续跑
bash scripts/control_vdu.sh wait
```

## 增量与失效
- `python3 -m vdu relabel --version 4 --changed seg02`：整体替换标注版本，
  声学窗全复用，仅受共现影响的字证据/校准/区间结论重算，账本逐条记录命中与失效。
- 缓存键由上游内容指纹、配置指纹、量化档位、标签版本指纹派生，见
  `vdu/fingerprint.py` 与 `vdu/ledger.py`。

详见 `docs/DESIGN.md`、`docs/REQUIREMENTS.md`、`docs/VERIFICATION.md`。
