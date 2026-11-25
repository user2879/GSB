# Speaker Diag — 说话人日志批量归属诊断组件

本项目面向 16 kHz 单声道说话人日志场景，回答一个可复核问题：

> 同一段长音频按声学窗重叠加窗、再切成批量推理窗后，各窗的说话人归属能否在多轮参数版本更替、回执格式演进、租约排队与进程中断之后被唯一复核？

实现是纯本地、确定性、整数定点优先的 Python 组件。音频、窗、真值、特征、模型参数、回执、租约账本和 SQLite 内容摘要均由声明与固定整数种子生成，不需要外部服务、账号、容器或非回环网络。

## 快速开始

Windows 10/11，Python 3.11 或 3.12：

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m speaker_diag prepare --samples samples
py -3.12 -m speaker_diag probe-ports --config samples/workload.json
py -3.12 -m speaker_diag demo --root .
```

也可以运行 `run_demo.cmd`。演示会清理上一轮已知账本与报告，再生成：

- `artifacts/verification.json`：完整验收摘要
- `artifacts/diagnostic.sqlite3`：已提交的批量归属账本
- `artifacts/content_manifest.json`：账本批序、内容链摘要和数据库摘要
- `artifacts/normalization.json`、`invalidation.json`、`lease_report.json`、`metrics.json`、`recovery.json`、`anomalies.json`
- `artifacts/audio.wav`、`features.json`、`truth.json`：可直接审计的合成对象

Linux/WSL 下把 `py -3.12` 换成 `python` 即可。Tkinter 只用于可选的报告查看器：

```text
python -m speaker_diag gui --report artifacts/verification.json
```

当前验证容器的 Python 缺少 `libtk8.6.so`，因此 GUI 未在本容器启动；CLI、测试和全部诊断路径不依赖 Tk。

## 常用命令

```text
python -m speaker_diag prepare --samples samples
python -m speaker_diag validate --config samples/workload.json
python -m speaker_diag probe-ports --config samples/workload.json
python -m speaker_diag normalize --config samples/workload.json --receipt samples/receipts/r2_B001.json --batch-id B001
python -m speaker_diag demo --root .
python scripts/probe_ports.py --config samples/workload.json
python -m unittest discover -s tests -v
```

## 目录

```text
speaker_diag/
  spec.py             JSON 强校验、行号定位、确定性窗起点
  synthetic.py        16-bit PCM、说话人/停顿/重叠、Q12 整数特征
  model.py            整数定点参数生成与前向
  engine.py           批载荷、本地/worker 共享推理路径
  receipts.py         r1/r2/r3 解码、补齐依据、稳定归一
  protocol.py         回环逐行 JSON 与真实 socket 超时
  worker.py           子进程工作进程
  orchestration.py    父进程启动、端口探测、握手与批请求
  leases.py           三重配额、单调时钟过期、乐观续约、事件重放
  invalidation.py     依赖图失效传播、仅失效子集重算、逐字节核对
  metrics.py          定点误差界、稳定决胜、归属命中率与误差成因
  ledger.py           SQLite 事务、摘要链、快照替换、恢复重放
  pipeline.py         本地产出、真实回环执行与账本执行流水线
  samples.py          内置规则样本生成器
  demo.py             可重复验收演示
  gui.py              可选 Tkinter 报告查看器
samples/              固定种子、端口、窗参数、工作进程与请求
tests/                44 项自动化测试
scripts/probe_ports.py
```

## 声明与校验

`WorkloadSpec` 从单个本地 JSON 读取。关键字段：

- 顶层：`spec_version`、`session_version`、整数 `seed`
- `audio`：`duration_ms`、`sample_rate=16000`、`channels=1`、`sample_width_bits=16`、`speaker_count=2..3`、`pause_ms`、`overlap_bps`
- `window`：`window_ms`、`hop_ms`、整数万分比 `overlap_bps`
- `batch.batch_size`
- `quantization`：`scheme`、正整数 `fractional_bits`
- `workers[]`：唯一 `worker_id`、仅允许 `127.0.0.1`、全局唯一高位端口、`model_dim`、`hidden_dim`、`seed`、`parameter_version`、并发配额、租约上限、租约毫秒
- `requests[]`：唯一 `batch_id`、已声明 worker、批内唯一且位于确定性窗范围内的 `window_ids`
- `limits`：全局租约、定点缩放、比较误差守卫、握手/I/O 超时

非法声明统一为 `E_SPEC`，并报告首个出错字段与源码行号。JSON 由内置位置感知解析器读取，因此重复键如 `workers[1].port` 也能定位到对应行。样本声明位于 `samples/workload*.json`；`seed.json`、`audio.json`、`window.json`、`workers.json`、`requests.json` 是基础版本的独立审阅镜像。

## 确定性构造

1. 每个说话人的基频、基频轮廓、三个共振峰、带宽和相位都由 `SHA-256(seed, speaker_index, ...)` 派生，不使用环境随机数。
2. 说话人按 2–3 人轮转，包括可控短停顿与按万分比触发的重叠区间。每窗输出主导真值、活跃集合、覆盖率、重叠比例、真值来源和文字说明。
3. PCM 为 `np.int16`；重叠加窗后提取 Q12 频谱带幅度比整数向量，截断方向显式声明为 `floor`。
4. 窗摘要、真值摘要和特征摘要只取决于音频声明、窗参数和最大 `model_dim`，不取决于 worker 或请求的加载顺序。样本的乱序加载/执行测试逐位一致。

样本长 7400 ms，400 ms 窗、200 ms 窗移，得到 36 个重叠窗、10 个批、3 名说话人。基础 PCM/窗/真值/特征摘要完整值见 `artifacts/verification.json`、`artifacts/audio_manifest.json` 与 `samples/manifest.json`。

## r1/r2/r3 归一

- r1：批次、窗、归属摘要。
- r2：增加 `param_version`、非负整数 `elapsed_ms`、`worker_id`。
- r3：再增加批内唯一且不越界的 `rank`、`duplicate_of`、`parameter_digest`、`parameter_snapshot` 和参与窗集合。

归一结果包含 `source_format`、`source_format_version`、`filled_fields`、`fill_basis`、参与与窗集合及稳定排序后的窗口项。r1/r2 的缺失字段由请求/worker 上下文补齐，并在 `fill_basis` 中写明依据；r3 不自带缺失核心字段。

解析检查：

- 负耗时、重复/越界名次、缺失字段、上下文不匹配：`E_FIELD`
- `duplicate_of` 指向不存在的窗、自引用、参与集外窗：`E_REF`

同一回执重复解析的规范 JSON 字节与 SHA-256 完全一致。`samples/receipts` 中的三版 B001 回执归一后归属摘要均为：

```text
afb26f55fa5404e06a5f3eec96331d7523bb7d15d53c32d043e7b5e93512b2c2
```

## 回环 worker

父进程先逐端口执行真实 `bind` 探测，再以 `sys.executable -m speaker_diag.worker` 启动独立子进程。所有监听和连接都限定在 `127.0.0.1`。协议为 UTF-8 规范 JSON 加换行：

1. 父进程连接、发送 `hello`，等待 `ready`。
2. `ready` 必须绑定 worker、参数版本和参数摘要。
3. 父进程发送批标识、参与窗集合、窗口特征与摘要。
4. worker 执行 NumPy 整数前向，回传 r3 回执；父进程再次核对批、参与集、版本和摘要。

错误映射：

- `E_PORT`：端口被占用或子进程未进入监听
- `E_HANDSHAKE`：连接/握手/回执等待的真实 socket 超时，或 worker 提前关闭
- `E_FIELD`：协议或缺字段、批绑定不一致

启动探测在 Linux/macOS 使用 `SO_REUSEADDR` 以允许重建仅有 TIME_WAIT 的高位端口；Windows 使用 `SO_EXCLUSIVEADDRUSE`。活跃监听仍会触发 `E_PORT`。

客户端连接会显式绑定与目标不同的本地端口，并复核连接后的 4 元组，避免 Linux 回环偶发选择相同源/目的端口而产生 TCP self-connect。该问题由重复端到端测试实际捕获并已回归覆盖。

## 失效边界

依赖链为 `worker 参数身份 -> 批 -> 窗实例`：

- `parameter_version`、worker seed/dim/量化口径变化：该 worker 的批为直接失效。
- `batch_size`、请求所属 worker 或参与窗变化：批布局变化；全局批大小变化使全部批直接失效。
- 音频/窗参数变化：全部窗和批直接失效。
- 直接失效批的全部窗失效；重复投递/共享窗再保守传播到其他消费批，并继续传播其窗。这样得到直接批、间接批、直接窗、间接窗及每步依据。

输出按 `worker_id, batch_order` 稳定排序。仅重算失效批，未失效批的规范归一回执被原样复用，并逐字段、逐字节核对。基础版本到 v2 的实测结果：

- 直接批：`B001,B003,B005,B007,B009`
- 间接批：`B004,B010`
- 失效窗：24
- 未失效批：`B002,B006,B008`，字节完全一致
- 选择性重算结果与 v2 全量推理的归属摘要逐批一致

v3 同时改变两名 worker、全局量化位宽与批大小，10 个批、36 个窗全部失效。

## 租约与定点诊断

一次授予同时占用：

- 全局批量租约上限
- 每 worker 租约上限
- 每 worker 并发配额

租约记录授予/到期单调毫秒、版本号和排队开始/结束毫秒。批次在配额不足时阻塞排队；到期由 `time.monotonic_ns()` 判定并释放配额；续约使用版本号乐观校验，不符为 `E_LEASE`。所有 `QUEUED/GRANTED/RENEWED/RELEASED/EXPIRED` 事件可离线重放。

定点口径以 `metric_scale` 为基础，采用非负 `floor`：

- 单值量化误差：`< 1 / scale`
- 两点比较误差：`< 2 / scale`
- 最小可区分差：3 个量化单位
- 守卫带：`max(2, 配置值)` 个量化单位

当候选批得分差或 worker 租约水位差落在守卫带内，按更小 `batch_id`/`worker_id` 稳定决胜，并标记 `affected_by_error_bound=true`。窗命中率、批耗时和租约水位均为整数定点。错误归因分为：

- `overlap_attribution`：重叠窗主导归属歧义
- `quantization_truncation`：分数差落入量化/比较误差界或残差
- `batch_reorder`：名次相对规范请求顺序变化

占比使用最大余数法，整数万分比之和严格为 10000。

## SQLite 账本与恢复

- 账本带整数 schema/格式版本、会话版本、声明摘要、逐批内容摘要链。
- 每批回执归一与窗归属在同一个 `BEGIN IMMEDIATE` 事务中写入，外键和唯一约束防止半批。
- 会话版本变化时整库替换；相同会话但声明摘要不同则 `E_REPLAY`。
- 强制终止后，`recover(expected_order)` 只信任已提交批，把缺失批写入 `unknown_batches`，返回重跑顺序。
- 重跑缺失批后，按账本参数版本和批序重放；字段级首个差异报告 `E_REPLAY`，包含批标识、窗标识、字段、期望值和实际值。重放比较归一后的归属语义（批、worker、参数版本/摘要/快照、参与窗、名次、重复引用、归属），明确忽略来源格式、补齐依据、`elapsed_ms` 与可选诊断分数，因而 r1/r2/r3 可互操作。
- 会话快照包含完整批/窗行与内容摘要；恢复前验证摘要，并按快照整体替换。

演示中，崩溃子进程提交前 4 批后以退出码 37 强制退出；父进程识别 `B005..B010` 为未知，重跑后 10 批全部提交并通过重放逐项校验。

## 业务错误码

| 代码 | 场景 |
|---|---|
| `E_SPEC` | 声明字段非法、端口重复、非回环、JSON 语法/行号 |
| `E_FIELD` | 回执/协议字段缺失、类型错误、负耗时、名次非法、上下文不一致 |
| `E_REF` | 重复投递引用不存在/自引用、回执窗不在参与集 |
| `E_PORT` | 声明的 127.0.0.1 端口不可用 |
| `E_HANDSHAKE` | 真实 socket 握手/回执等待超时 |
| `E_LEASE` | 租约过期、配额等待超时、乐观版本不符 |
| `E_REPLAY` | 账本会话/schema 冲突、重复提交、快照摘要/重放字段不一致 |

## 实际验证结果

当前工作区已执行：

```text
python -W error::ResourceWarning -m unittest discover -s tests -v
# Ran 44 tests ... OK

python -m speaker_diag demo --root .
# status=ok, 16/16 checks true
```

验收摘要见 `artifacts/verification.json`：

- 36 窗、10 批、2 个真实子进程 worker、3 名说话人
- 三版回执归一一致且二次解析逐字节一致
- 10 批真实回环执行全部写入账本，10 批重放校验通过
- 租约事件可重放，全局/worker 配额均未越界，排队毫秒为真实单调时钟
- 快照恢复后 10 批重放通过；会话版本变化完成整体替换
- 强制终止恢复重跑 6 批并通过校验
- 异常注入得到 `E_SPEC,E_FIELD,E_REF,E_PORT,E_HANDSHAKE,E_LEASE,E_REPLAY`

当前容器为 Linux/WSL2、Python 3.11.0rc1、NumPy 1.26.4。代码仅使用 Python 标准库与 NumPy，不依赖 Docker；Windows 所需路径通过 `sys.executable`、`subprocess.Popen`、`127.0.0.1`、`SO_EXCLUSIVEADDRUSE` 和 SQLite 标准接口实现。
