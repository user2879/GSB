# affinity-scheduler

面向 Kubernetes 亲和性调度的确定性 Python 包与本地可重复示例。它在单机上完成节点/Pod 声明校验、Filter 与 Score 两阶段调度、三代调度记录归一迁移、参数版本失效传播、SQLite 多版本账本、真实回环 TCP 子进程打分和独立核对。

本仓库不依赖容器、真实集群或外部样本。所有节点、Pod、历史记录、SQLite 账本和日志都在本地生成。

## 结论

同一批 Pod 的调度结论可以被唯一复核：记录归一时保留格式来源与补齐依据，绑定结论使用排序固定的规范 JSON 字节和 SHA-256；迁移前后逐条比较该字节，参数变更后只复用未失效回执，并由只读账本 + 从零全量重算独立验证。最终验收中 3 条 v1/v2/v3 记录迁移后的结论哈希全部一致，增量与全量逐 Pod 一致，未受影响的 `pod-0002` 结论与回执逐字节不变。

## 交付内容

- `affinity_scheduler/`：可集成 Python 包与 `affinity-scheduler` CLI。
- `config/`、`data/`、`logs/`：任务运行生成的样本、账本、状态与结构化日志。
- `tests/`：31 项标准库 `unittest`，覆盖业务错误码、迁移中断、确定性、增量重算与账本恢复。
- `examples/run_acceptance.py`：从空工作目录自动完成全量验收。
- `docs/DESIGN.md`：字段口径、整数公式、迁移/失效/账本状态机。
- `docs/ACCEPTANCE.md`：本机实际运行结果。

## 环境与安装

本机为 WSL2 Ubuntu，确认解释器为 Python 3.11（`3.11.0rc1`）。隔离环境位于 `.venv/`，NumPy 固定为 `1.26.4`。

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4
python -m pip install wheel==0.41.3
python -m pip install -e . --no-build-isolation
```

当前镜像的系统 Python 缺少 `ensurepip`，因此首次建 venv 时使用了 `--without-pip`，再把同版本解释器已有的 pip 注入 `.venv`；之后仍用 `.venv` 内 pip 安装 NumPy。`wheel` 仅用于旧版 pip 的 editable 构建，不是运行时依赖。

验证安装：

```bash
source .venv/bin/activate
affinity-scheduler --help
python -c 'import affinity_scheduler, numpy; print(affinity_scheduler.__version__, numpy.__version__)'
```

## 快速运行

从空目标目录执行一键验收：

```bash
source .venv/bin/activate
python examples/run_acceptance.py \
  --workdir . \
  --summary-out artifacts/acceptance-result.json
```

目标目录中的 `config/`、`logs/`、`data/` 必须不存在或为空；脚本不会覆盖已有运行产物。它会选择连续可用的高位回环端口，不访问网络。

分步命令：

```bash
affinity-scheduler generate-sample --root .
affinity-scheduler validate-spec --spec config/spec.json
affinity-scheduler normalize-records --records data/records.jsonl --spec config/spec.json
affinity-scheduler migrate --records data/records.jsonl --spec config/spec.json \
  --db data/ledger.sqlite3 --batch-size 1

# 先以参数版本 1 全量建立基线
affinity-scheduler schedule --spec config/spec.json --records data/records.jsonl \
  --state data/state.json --ledger data/ledger.sqlite3

# 在覆盖 state 前先比较“增量子集”与“从零全量”
affinity-scheduler verify-incremental --spec config/spec-v2.json --state data/state.json

# 参数版本 2 只重算失效子集
affinity-scheduler schedule --spec config/spec-v2.json --records data/records.jsonl \
  --state data/state.json --ledger data/ledger.sqlite3

# 只读账本 + 从零独立重算
affinity-scheduler verify --ledger data/ledger.sqlite3 \
  --spec config/spec-v2.json --records data/records.jsonl
```

## 样本语义

样本含 4 个节点和 4 个 Pod：

- `node-a`：无污点，`zone=east`，资源充足。
- `node-b`：`dedicated=ml:NoSchedule`，由 `pod-0002` 容忍。
- `node-c`：`maintenance=ops:PreferNoSchedule`，参与软性打分。
- `node-d`：小容量西区节点。
- `pod-0001`：硬/软节点亲和到 `zone=east`。
- `pod-0002`：带污点容忍，不依赖节点/Pod 亲和插件。
- `pod-0003`：软 Pod 反亲和到 `app=web`，拓扑键 `rack`。
- `pod-0004`：软节点亲和到 `zone=west`。

参数版本 2 把 `node_affinity` 权重从 `3500` 调到 `2500`、`pod_anti_affinity` 从 `1500` 调到 `2500`，权重和仍为 `10000`。因此只有回执中相应插件分项非零的 `pod-0001/0003/0004` 失效；`pod-0002` 原样复用。

## 记录版本

磁盘记录为混合 JSONL/定宽行文件。格式版本按每行检测：

- v1：`v1` + 定宽字段 `pod_id[18] node_name[18] total_score[6] veto_code[4]`，总计 48 字符。
- v2：JSON，含浮点插件分项、浮点总分、否决码和命名决胜依据。
- v3：JSON，增加参数版本、`scale_exponent/scale_factor`、逐插件定点贡献和整数误差界。

所有版本归一为 v3 内部表示，并携带 `source_format_version`、`filled_fields` 与逐字段 `basis`。v1 缺省的插件分项使用显式零值，v2 浮点分项按 half-up 取整；补齐来源会进入记录与迁移输出。

## 错误码

| 代码 | 触发点 |
|---|---|
| `E_SPEC` | 声明、唯一性、正整数、污点、权重和、缩放因子或版本单调性非法 |
| `E_FORMAT` | 未知版本、必需字段缺失、JSON/行格式损坏、总分越界、否决码非法 |
| `E_WIDTH` | v1 定宽行实际宽度与声明宽度不一致 |
| `E_PORT` | 回环端口被占用或子进程无法绑定 |
| `E_TIMEOUT` | 握手、连接或回执读取发生真实套接字超时 |
| `E_REPLY` | 回执字段缺失、身份/参数版本不一致或插件分项不完整 |
| `E_MIGRATE` | 迁移前后结论不一致，或检查点/备份不可续做 |
| `E_SNAPSHOT` | 账本版本集合/内容摘要不符，或摘要异常注入 |
| `E_VERIFY` | 独立重算与账本差异，或其他稳定性对照失败 |

CLI 业务错误以 JSON 输出到 stdout、退出码为 `2`，并写入含 `monotonic_ms` 的 JSONL 日志。

## 测试

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

本机实际结果：`Ran 31 tests ... OK`。完整验收记录见 `docs/ACCEPTANCE.md`。

## 关键口径

- Filter 严格按 `taint_toleration → node_affinity → pod_anti_affinity → resource_fit` 执行，并在每个节点保存逐步轨迹。
- 四个插件原始分为 `0..100` 整数；贡献为 `floor(score × weight_myriad × 10^e / 10000)`，总分向下截断，操作数受 `int64` 约束并在溢出时饱和。
- 同一 Pod 的候选节点始终按节点名排序后打分；决胜顺序为总分、四插件原始分、节点名。
- 稳定结论字节不包含回执毫秒耗时，因此节点枚举顺序和网络调度时序不会改变绑定结论。
- SQLite 迁移按批原子提交并更新检查点；原记录备份权限为 `0444`。
- 强制终止后只信已提交账本行；未知 Pod 必须重跑，恢复输出明确 `exit_report_available: false`。

## 边界说明

- 资源余量按当前 Pod 请求与节点可分配量计算，不把队列中其他 Pod 的累计请求从节点容量中扣减；Pod 反亲和会使用本次调度中已确定的绑定。
- v1 没有插件分项，归一保留原总分并将缺失分项标为零，同时保留完整补齐依据；不臆造历史插件分项。
- 参数版本号必须单调不减。仅版本号变化而有效权重、缩放因子、节点/Pod 声明均未变时，不强制重打分，只记录 metadata 传播步骤。
