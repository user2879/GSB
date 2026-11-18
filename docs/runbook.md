# 运行手册与验证记录

## 环境准备（一次性）

系统 Python 为 3.11.0rc1，但发行版裁剪了 `ensurepip`，且无 sudo 权限。采用
`get-pip.py` 引导后创建不带 pip 的虚拟环境，再在环境内安装 pip 与 NumPy：

```bash
python3 -m venv --without-pip .venv311
.venv311/bin/python /tmp/get-pip.py
.venv311/bin/python -m pip install numpy   # numpy 2.4.6
.venv311/bin/python -c "import sys, numpy; print(sys.version, numpy.__version__)"
```

后续所有命令均使用 `.venv311/bin/python`，不依赖系统站点包。

## 1. 生成确定性样本

```bash
.venv311/bin/python -c "from ab_workbench.generator import generate; generate('data/sample')"
```

产出 `data/sample/` 下四份输入：`exposures.jsonl`（默认 2880 条曝光）、
`experiment_config.json`、`user_attributes.json`、`injections.json`。固定种子
`SEED=20261118`，重复生成逐字节一致。

## 2. 并发评测 / 串行对照 / 全量重建

```bash
# 并发（线程池，并发度取配置 concurrency）
.venv311/bin/python -m ab_workbench.cli run \
  --log data/sample/exposures.jsonl \
  --config data/sample/experiment_config.json \
  --attributes data/sample/user_attributes.json \
  --injections data/sample/injections.json \
  --snapshot-dir snapshots/demo --cache-dir tmp/demo-cache \
  --tmp-dir tmp/demo-tmp --report tmp/report-concurrent.json

# 强制单线程串行
.venv311/bin/python -m ab_workbench.cli serial  ...（相同参数，报告换成 report-serial.json）

# 串行对照摘要（逐字段比对并输出顺序）
.venv311/bin/python scripts/serial_compare.py \
  --log data/sample/exposures.jsonl --config data/sample/experiment_config.json \
  --attributes data/sample/user_attributes.json --injections data/sample/injections.json \
  --workdir tmp/sercmp --summary tmp/serial_summary.json

# 全量重建（清空缓存与快照后一次性计算）
.venv311/bin/python scripts/full_rebuild.py ... --report tmp/fb/report.json
```

## 3. 增量 vs 全量等价性矩阵

```bash
.venv311/bin/python scripts/equivalence_check.py
```

编辑序列覆盖：新增曝光批次 → 单条曝光修正 → 实验配置变更（最小样本量）→
定点位宽调整；矩阵覆盖两种定点位宽与两种桶数组合 `(width,buckets)=(6,18)` 与
`(4,24)`。每步增量结果与“从空白工作区按最终输入一次性全量”的结果，对
`group_stats / estimation / attribution / failures` 逐字段比较，并校验失败清单顺序
与缓存复用/重算行为。

## 4. 异常注入复现

注入清单 `injections.json` 中：

| 注入 ID | 故障 | 阶段 | 期望业务码 |
| --- | --- | --- | --- |
| INJ-001 | 非有限模型分数（NaN） | load | E_LOG_INVALID |
| INJ-002 | 分组标识越界 | load | E_LOG_INVALID |
| INJ-003 | 点击标记非 0/1 | load | E_LOG_INVALID |
| INJ-004 | 桶数变化后旧快照复用 | snapshot | E_CONFIG_MISMATCH |

紧容差场景（测试中以 `attribution_tolerance=1e-9` 构造）触发
`E_ATTRIBUTION_DRIFT`，并在报告中给出逐层修正明细且最终偏差归零。

## 5. 信号与快照

```bash
# 子进程发送 SIGTERM，验证完成当前批次后退出码 0、快照完整
```

`tests/test_snapshot_resilience.py` 与手动子进程实验均确认：SIGTERM 后
`exit_code=0`、`signal_handled=SIGTERM`、`manifest.json` 与各阶段 `latest.json`
完整；强制终止后仅依据已 `os.replace` 的完整快照恢复。

## 实际验证结果

- 单元与行为测试：`python -m unittest discover -s tests` → 25 个用例全部 OK，
  无 ResourceWarning。
- 等价性矩阵：两种位宽×两种桶数，增量/全量/串行报告逐字段一致。
- 串行对照摘要：`identical=true`，分组表顺序、分层表顺序、失败码顺序稳定。
- 注入命中：INJ-001/002/003 → E_LOG_INVALID（各 1 条），INJ-004 → E_CONFIG_MISMATCH。
- 精确复用：相同输入重跑 7 个产物全部复用、0 重算；仅 control 追加批次时，两个未受
  影响分组 stats 直接复用，受影响分组与下游 estimation/attribution 重算。
- 量化档位：`quantization_step=0.01` 时指标值对齐到 0.01 网格；桶数 18→24 后分桶结果
  与配置指纹缓存键相应变化。
