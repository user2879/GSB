# 列式段压缩与扫描诊断应用

面向 Windows 本地环境的确定性列式段诊断工具。应用从 JSON 规范构建二进制列式段，提供原始/字典/游程块编码、列裁剪与谓词下推、三重预算线程池调度、增量失效传播、SQLite 块级账本、权威重放、版本整体替换和会话快照恢复。

## 快速开始（Windows）

要求 CPython 3.11+，且安装时勾选 `tcl/tk`。

```bat
python --version
python -m pip install -r requirements.txt
python -m tkinter
setup.bat
run.bat
```

`setup.bat` 会检查 Python/Tk、安装 NumPy，并运行完整验收。纯诊断与验收也可直接使用命令行：

```bat
python run.py demo --output artifacts
python run.py init --workspace work --overwrite
python run.py scan --workspace work --batch-size 5 --delay-ms 2
python run.py verify --workspace work
```

Linux/WSL 无 `libtk8.6` 时可运行除 `gui` 外的全部命令；本次实际验收即采用该方式。Windows 官方 Python 自带 Tk，`run.bat` 会直接打开六页桌面诊断台。

## 已实现能力

- `E_SPEC` 强校验：列名唯一、行组从 0 开始首尾相接、编码枚举、非负字节数、块/列/行组完整覆盖、查询列存在；错误带首个 `block_id` 与 `field`。
- 三种确定性块编码：`raw`、`dictionary`、`rle`。字典项显式按数值升序；块以长度前缀写入二进制段，支持可选 zlib 压缩。
- 规范化谓词 AST：等值、范围比较、`IN`、`AND` 与括号；空白和冗余括号不影响计划摘要。
- 列裁剪、行组/块候选推导与推导依据输出；跨列谓词在块级掩码上安全合并。
- 三重预算：全局并行任务数、每列解码内存上限、每段并发配额；实际投射到 `ThreadPoolExecutor`，记录队列开始/结束、运行起止与峰值指标。
- 稳定槽位块序号：候选块因数据变化增删时，存活任务不重排，便于账本与快照定位。
- 增量变更集：直接失效块重编码；沿查询×行组×块依赖精确传播；仅失效任务重扫，其余块记录、块摘要、任务结果与查询结论逐项校验保持。
- SQLite 批次账本：任务、块序号、完成行数、解码内存、单调时钟、块摘要、结果摘要、计划和查询行组结论；批次事务原子提交。
- 权威重算路径：只从 `segment.seg` 解码并重算，不读取账本摘要作为输入；普通账本一致性差异以 `E_DIVERGE` 定位，快照恢复重放差异以 `E_REPLAY` 定位，均带首个块序号、字段与双方值。
- 单调版本：预算参数版本与编码选择版本递增；旧生成统一标陈旧并从当前视图/汇总隔离；整体替换并稳定输出 `added/deleted/changed/unchanged`。
- JSON 会话快照：段集合摘要、格式/编码/预算版本、任务轨迹、块摘要、结果摘要、统计、内容摘要与游标；内容校验失败报 `E_SNAPSHOT`。
- 恢复为整视图替换；按快照游标继续同一扫描序列；逐块结局、块摘要、结果摘要、查询行集合与统计逐项重放校验。
- 强制终止安全：每个账本批次单事务；中途异常不会留下部分批次，恢复只基于完整提交批次。

## 常用命令

```bat
python run.py sample --output samples
python run.py init --workspace work --spec samples/sample_spec.json --overwrite
python run.py scan --workspace work --parallelism 4 --memory-budget 128 --segment-quota 3 --batch-size 5
python run.py update --workspace work --changeset samples/sample_changeset.json
python run.py snapshot --workspace work --output work/session.json
python run.py restore --workspace work --snapshot work/session.json
python run.py replay --workspace work
python run.py budget --workspace work --parallelism 3 --memory-budget 100 --segment-quota 2
python run.py encoding-policy --workspace work --column region --encoding raw --compressed no
python run.py gui
```

`scan` 从账本游标续跑；修改预算参数时自动创建新版本并整体替换当前视图。`update` 接受数组或含 `changes` 数组的 JSON 变更集。

## 项目结构

- `src/columnar_diag/`：规范、编解码、段文件、规划、调度、扫描、账本、快照、重放、CLI 与 Tk GUI。
- `tests/`：20 项 `unittest`，覆盖校验、编码稳定性、段完整性、计划归一化与语义、预算调度、增量与重开一致性、账本发散、快照、版本与批次原子性。
- `samples/`：由交付物生成的内置确定性规范、变更集与预算参数。
- `artifacts/acceptance_report.json`：机器可读实测报告。
- `artifacts/ACCEPTANCE.md`：实测摘要。
- `artifacts/TEST_RESULTS.txt`、`artifacts/CLI_SMOKE.txt`：单元测试与命令行冒烟原始记录。
- `docs/ARCHITECTURE.md`：格式、调度、账本、快照与恢复不变量。

## 本次实际验证

执行环境：Python `3.11.0rc1`、SQLite `3.37.2`、NumPy `1.26.4`；当前 WSL 环境缺少 `libtk8.6`，故 GUI 未在本容器启动，CLI/调度/账本/快照全链路均已无界面验证。

```text
python -m unittest discover -s tests -v
Ran 20 tests ... OK

python run.py demo --output artifacts
all_checks_passed: true
```

关键实测：

- 30 个块级任务，全局峰值 3/4，段并发峰值 3/3，每列解码内存峰值均 ≤128 字节，最大排队等待约 24 ms。
- 修改 `amount@rg1` 后仅该块重编码；11 个未失效块记录、27 个未失效任务结果与 11 个未失效查询行组结论逐字节/逐项一致。
- 预算版本 2 与编码选择版本 3 均单调推进，旧生成全部 `stale`；当前结论差异稳定分类。
- 快照在游标 10 保存，继续推进到 30 后恢复，游标回到 10，再续跑到 30；权威逐块重放通过。
- 篡改快照统计触发 `E_SNAPSHOT`；篡改账本 `result_digest` 时权威校验触发 `E_DIVERGE`、快照重放触发 `E_REPLAY`，均定位 `block_ordinal=3`, `field=result_digest`。
- 注入批次中途异常后，当前完整批次数不变，账本无半批残留，重放仍通过。

详见 `artifacts/ACCEPTANCE.md` 与 `artifacts/acceptance_report.json`。
