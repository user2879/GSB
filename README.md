# Hot-Block Sparse Attention Operator Diagnostics

面向长序列注意力热块稀疏化的确定性算子诊断应用：小端张量包解析、块级定点打分、
热块选择反馈迭代、缓存依赖失效传播、预算受限并行调度、阶段快照恢复与故障注入诊断。

## 环境准备（Windows / Linux 通用）

需要 Python 3.11 与 NumPy，无 GPU、无外部网络依赖。

```
py -3.11 -m pip install -r requirements.txt
```

## 生成确定性样本

```
python scripts/generate_samples.py --sample-dir samples/demo
```

产物：`tensor_pack.bin`、`prefilter.json`、`fixedpoint.json`、`commands.json`、
`injections.json` 以及分离的块级/截断/快照篡改注入文件。

## 运行诊断

```
python -m hotblock_diag.cli --sample-dir samples/demo --workspace runs/demo --fresh
python -m hotblock_diag.cli --sample-dir samples/demo --workspace runs/demo
```

常用参数：`--injections`、`--commands`、`--fixedpoint`、`--max-batches`、`--fresh`。
报告写入 `<workspace>/diagnostic_report.json`，缓存在 `cache/`，快照在 `snapshots/`。

退出码：`0` 成功/信号安全停止，`2` 致命输入错误，`3` 预算暂停可恢复，`4` 张量包截断。

## 测试与文档

```
python -m unittest discover -s tests
bash scripts/run_demo.sh
```

- 设计说明：`docs/DESIGN.md`
- 验证结果：`docs/VERIFICATION.md`
- 原始任务书：`docs/TASK_BRIEF.md`

## 代码结构

```
hotblock_diag/
  tensorpack.py  小端二进制张量包与 CRC 校验
  config.py      预筛/定点/命令流校验
  fixedpoint.py  量化、饱和累加、确定性舍入、块对指纹
  blocks.py      切块与零填充
  cache.py       缓存键、依赖图与失效传播
  hottable.py    热块账本、逐出、压制、重建
  scheduler.py   线程池与预算配额调度
  snapshot.py    原子快照、清单、信号
  injection.py   故障注入
  engine.py      反馈环编排与诊断报告
  cli.py         命令行入口
tests/           38 个确定性单元测试
scripts/         样本生成与演示脚本
```
