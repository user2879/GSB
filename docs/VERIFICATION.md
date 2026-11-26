# 验证记录

环境：WSL2 Ubuntu 22.04，Python 3.11，venv 内 numpy 2.4.6、pytest 9.2.0。
全部运行在本地文件系统，无容器、无 GPU、无外部网络依赖。

## 复现步骤

```bash
python3 -m virtualenv .venv          # 系统 ensurepip 缺失，使用 virtualenv 引导
source .venv/bin/activate
pip install numpy pytest
python scripts/run_demo.py           # 端到端演示，产物写入 artifacts/ 与 ledger/
python -m pytest                      # 40 个确定性测试
```

## 自动化测试（40 passed）

| 测试文件 | 用例数 | 覆盖要求 |
| --- | --- | --- |
| tests/test_kernel.py | 6 | 温度/位宽/档位改变整数、量化和为尺度、残差记录、高一位宽参考 |
| tests/test_truncation_sampling.py | 7 | 确定性排序、TopK/核采样前缀、半步边界、边界归因、确定性 RNG |
| tests/test_validation.py | 7 | 唯一标识、步空洞/回退、非有限/正对数概率、参数范围、多位置聚合 |
| tests/test_exchange.py | 5 | v1/v2 往返等价、定宽不足/非法字符业务码、旁路残差与参考还原 |
| tests/test_cache_incremental.py | 4 | 命中复用、替换只失效自身、追加只新增、配置变更改键 |
| tests/test_scheduler.py | 5 | 稳定排序键、预算待办与排空、任意分割点一致、配额中止、重算上限 |
| tests/test_end_to_end.py | 6 | 全量确定性、增量=全量、并发=串行、注入隔离与编号、四环节归因、原子账本 |

## 演示脚本实测结论（artifacts/）

- `full_run.json`：基线全量 18 条逐请求结论，含边界令牌与缓存事件。
- `incremental_run.json` / `rebuild_run.json`：应用同一编辑命令流后，
  增量与从空白全量重建均为 19 条结论，`incremental_vs_full.json` 记录
  `conclusions_identical=true` 且计数相等。
- `exchange_v1.txt` / `exchange_v2.json` / `v1_sidecar.json`：
  双版本回执计数一致（v1=v2），`roundtrip_ok.txt` 证明往返等价。
- `budget_run.json`：批预算 12（最大单任务 8 点）下跨批调度，
  最终 `todos=0`，结论与一次性串行运行逐字段一致。
- `injection_report.json`：注入命中带编号
  `INJ-001`(非有限概率)、`INJ-002`(越界阈限)、`INJ-003`(步序号回退)，
  业务码计数包含 `NON_FINITE_LOGPROB/TOPP_OUT_OF_RANGE/STEP_REGRESSION/...`，
  故障请求 R002/R003 被隔离，R001 正常产出；归因行覆盖四环节并标注首次超容差环节。

## 账本

`ledger/` 下每个运行场景以 `snapshot.json`（原子整体提交）+ `events.jsonl`
持久化；临时文件 `.tmp` 在提交前不可见，恢复时校验版本号与配置哈希。
