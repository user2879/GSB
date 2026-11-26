# 实际验证结果

总体验收：通过

## 环境
- Python 3.11.0rc1 (/home/zsl/feedback/uidfc5909d3-default-357/.venv/bin/python)
- NumPy 2.4.6
- Tkinter: {'ok': True, 'tk_version': '8.6.12'}
- 回环 TCP: {'ok': True, 'host': '127.0.0.1', 'port': 44918}
- 多进程: {'ok': True, 'start_method': 'fork'}

## 主样本基线
- {'samples': 6, 'keypoints': 159, 'conflicts': 0, 'failures': 0, 'windows': 2070}

## 一致性
- 命令重放逐字段一致：True
- 并发度 1/2/4 核心字段一致：True
- 租约真实回收/代次拒绝：True
- 预算/配额/待办优先：True

## 注入
- INJ-PNG-TRUNC-001 truncate_png_data_chunk: hit=True
- INJ-KP-NONINT-002 noninteger_keypoint_x: hit=True
- INJ-ORDER-SKIP-003 reading_order_gap: hit=True
- INJ-QUANT-UNKNOWN-004 unknown_quant_gear: hit=True
- 正常样本隔离：True

## GUI
- {'ok': True}

## pytest
- returncode=0 ok=True
