# 实际验证记录

环境：Python 3.11.0rc1（工作区 `.venv`，经 get-pip 引导 pip 26.3）、
NumPy 2.4.6、标准库 sqlite3（3.37.2）；无 Docker；全部通信仅限
127.0.0.1 高位端口。Tkinter 在该宿主不可用（无 `libtk8.6.so`，venv 无
tkinter 模块），GUI 走 CLI 降级，已在测试与演示中覆盖。

复现命令：

```bash
python3.11 -m venv .venv            # 若 ensurepip 缺失：先 get-pip 引导
.venv/bin/pip install -r requirements.txt
bash scripts/demo.sh                # 端到端演示
.venv/bin/python -m unittest discover -s tests -v   # 31 个测试
```

## 验收项与实测结果

| 验收点 | 命令/测试 | 实测结果 |
| --- | --- | --- |
| 强校验 E_SPEC（字段+行号） | `validate`；`test_espec_*` | 负到达序号报 `E_SPEC` 字段 `3.arrival` 行 34；端口冲突、非回环 host、维度不符均拒绝 |
| r1/r2/r3 归一 | `normalize`；`test_three_formats_normalize_to_same_core` | 各 12 条归一成功；r1 记录 6 个补齐字段及依据，r2 补 3 个，r3 无补齐 |
| E_FIELD / E_CHAIN | `test_efield_*`、`test_echain_*`、`test_rank_*` | 负耗时 `E_FIELD`（行 8）；幽灵 duplicate_ref `E_CHAIN`（行 23）；名次越界/批内重复均 `E_FIELD` |
| 确定性切批/批标识 | `test_plan_byte_identical_under_dispatch_order` | 逆序请求+逆序工作器得到完全相同的批标识与归属；批标识含显式 canonical arrivals |
| 顺序无关输出摘要 | `test_output_digest_order_independent` | 批内逆序处理 output/param 摘要不变；参数版本变化摘要必变 |
| 失效边界 | `incremental`；`test_propagation_boundary_*` | 仅 w-bravo 的 2 批、4 请求失效；w-alpha/w-gamma 摘要逐字节保留 |
| 增量=全量 | `incremental_equals_full` | `true`，规范字节长度 3007，两次 SHA-256 一致 |
| 并发配额 | `scheduler`；`test_real_loopback_scheduling_respects_quotas` | 6 批全部成功；全局峰值 2/上限 2；每工作器峰值 w-alpha 2、w-bravo 1、w-gamma 1，均未超配额；等待时钟字段齐全 |
| 真实超时 E_TIMEOUT | `scheduler --timeout-demo` | stall 工作器上 2 批报 `E_TIMEOUT`，800ms 真实 socket 超时 |
| 快照整体替换 | `snapshot --tamper` | 动作 `replaced`，差异来源 `schema_meta.content_digest`；版本号篡改同样替换 |
| 强制终止恢复 | `crash --after 2` + `recover` | 退出码 137；2 批已提交可恢复，4 批标记 unknown/rerun_required；重放通过，最终 6 批齐全 |
| E_REPLAY 定位 | `test_replay_check_locates_first_divergence` | 定位到首个分歧 `B:w-alpha:0:...` / `req-000` / `per_request_digest`，附 stored/recomputed |
| GUI 降级 | `test_gui_command_degrades_without_tk` | 无 Tk 环境输出结构化降级信息，不抛异常 |

测试统计：`Ran 31 tests ... OK`（含真实 TCP 子进程、真实套接字超时、
SIGKILL 子进程恢复、SQLite 篡改与重放）。静态检查：`pyflakes batchdiag tests`
无告警。

## 关键确定性证据（本次运行）

* 全量重建规范字节 SHA-256 前缀（乱序对照两次一致）：
  见 `run/demo_output.log` 的 `verify` 输出 `digest_a == digest_b`。
* 增量/全量合并字节长度：3007，`incremental_equals_full=true`。
* 批标识示例：`B:w-alpha:0:200231664249`、
  `B:w-bravo:0:39cb4bb16fad`，重跑保持不变。

## 已知环境限制

* 该容器 venv 无法安装/导入 tkinter（缺系统 Tk 库），GUI 仅提供可选实现；
  在带 Tk 的标准桌面 Python 上 `python -m batchdiag.cli gui ...` 可直接弹窗。
* `ensurepip` 在基础 Python 缺失，venv 通过 `get-pip.py` 引导 pip；
  `scripts/setup_env.sh` 记录标准流程。
