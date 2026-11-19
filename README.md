# 文件系统权限与路径检查审计引擎

离线、确定性的文件系统路径规范化与权限越权审计引擎。输入本地 JSON 清单，
输出含规范比较键、有效权限、业务错误码、误差界与临界不确定标记的原子
JSON 报告。设计细节见 `DESIGN.md`，实测结果见 `VERIFICATION.md`。

## 目录布局

```
audit_engine/   引擎源码（解析、路径、权限、ACL、预算、检查点、并发、CLI）
tests/          pytest 测试集（63 个用例）
samples/        确定性样本清单输出目录
checkpoints/    检查点与锁文件目录
reports/        原子 JSON 审计报告目录
scripts/        样本生成入口
```

## 快速开始

```bash
# 1. 准备环境（Python 3.10+；引擎零第三方运行时依赖）
python -m venv .venv
.venv/bin/pip install pytest            # 离线环境可复制本地 pytest 包

# 2. 生成确定性样本（无网络、无系统时钟真值）
.venv/bin/python scripts/generate_samples.py

# 3. 运行测试
.venv/bin/pytest -q

# 4. 运行审计（串行 / 线程 / 独立子进程三种后端）
.venv/bin/python -m audit_engine.cli samples/generated/*.json \
  --checkpoint-dir checkpoints/run --report reports/run.json \
  --concurrency 4 --backend thread --timeout 10 --max-records 1000
```

## 业务错误码

| 码 | 触发条件 |
| --- | --- |
| E_PARSE | 记录字段非法或标识重复，拒绝但不中断整批 |
| E_TRAVERSAL | 父目录回退/绝对路径越出受检根，附越界深度 |
| E_LINK_ESCAPE | 符号/硬链接跳转落点越出受检根 |
| E_DENY_SHADOW | 拒绝项被同级或更近的允许项覆盖 |
| E_INHERIT | 继承标记使子路径获得父路径未授权权限 |
| E_POLICY_CONFLICT | 同一记录在不同规范化策略下结论冲突 |
| E_TIMEOUT | 单文件真实墙钟超时，进程后端可真实 terminate |
| E_BUDGET | 单文件记录数超过上限 |
| E_LOCK | 检查点并发写锁冲突 |
| E_TRUTH_MISMATCH | 同步标记真值对照不一致 |

## 退出码

`0` 无违规；`1` 存在业务违规；`2` 超时/超预算；`3` 锁冲突；`4` 致命输入错误。
