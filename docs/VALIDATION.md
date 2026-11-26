# 验证记录（本地确定性样本）

环境：WSL Ubuntu-22.04，Python 3.11.0rc1，venv（virtualenv 创建），
numpy 2.4.6，pytest 9.2.0。Tkinter 系统库 `libtk8.6.so` 在该环境缺失，
本组件为无头 CLI/套接字服务，不依赖任何 GUI，故未使用 Tkinter。

## 1. 自动化测试

`python -m pytest` 结果：**40 passed**。

| 测试文件 | 覆盖点 |
| --- | --- |
| `tests/test_loader_validation.py` | 标识/行键唯一、非负整数（拒绝 bool）、结局枚举、commit_seq 约束、链引用与环、快照越界、首错定位、摘要稳定 |
| `tests/test_visibility.py` | 提交过晚、回滚/未决排除、被取代、自身未提交可见、理由链稳定序、摘要稳定、epsilon 决胜、规则冲突、epsilon=0 不触发 |
| `tests/test_executor.py` | JSON 行回执、E_PORT、真实 E_TIMEOUT（hold/reset）、E_FIELD |
| `tests/test_scheduler.py` | 双重限额、部分放行不越界、单调时钟等待窗口、轨迹确定性 |
| `tests/test_invalidation.py` | 事务/链差异、稳定失效集合、传播边、仅失效重算、未失效逐字节保持 |
| `tests/test_broken_chain.py` | 断链成因与 `version_chain_broken` 占比 |
| `tests/test_ledger_recovery.py` | 原子提交、整体替换+只读备份、SIGKILL 恢复重放、E_REPLAY 定位、与一次性判定一致 |

## 2. CLI 异常矩阵（实测）

| 注入参数 | 结果 |
| --- | --- |
| `--inject port_busy` | `E_PORT`（真实占用回环端口后绑定失败） |
| `--inject silent_hold` | `E_TIMEOUT`（接受连接但不响应，真实 socket 超时约 2s） |
| `--inject silent` | `E_TIMEOUT`（不 accept，连接重置归并为超时码） |
| `--inject missing_field` | `E_FIELD`（回执缺 `rule_version`） |
| `--inject rule_conflict` | 成因 `rule_version_conflict`，占比 1.0 |

所有超时均由真实 socket timeout 触发，未使用人为延迟。

## 3. 关键端到端结论

* 主样本 8 条查询，全局上限 2、每读点配额 1：实测峰值
  `global_peak=2`、各读点 `rp_peak=1`，无越界放行；
* 强制终止返回码 `-9`（POSIX SIGKILL，shell 视角 137），账本仅留
  `open` 轮次意图、零结论；重启后 8 条全部识别为未知并重跑，
  `replay_check.status=consistent`；
* 变更声明重开同一账本触发 `replaced`，生成 `0o444` 只读备份与
  `replacement_log` 差异列表；
* 失效传播实测失效集合 `{Q1,Q2,Q3,Q4,Q7,Q8}`，`Q5/Q6`（行 R-B）
  摘要逐字节保持。

## 4. 复现命令

```bash
python3 -m virtualenv .venv && . .venv/bin/activate
pip install numpy pytest
python -m pytest
python -m mvcc_lab --config samples/sample_spec.json \
  --ledger build/demo.sqlite --replay --report build/demo.json
```
