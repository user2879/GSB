# quantcalib

面向视觉表征模型部署前标定的纯 Python 包：逐层逐组定点重量化、轻量蒸馏校正、
三版本历史量化回执归一与权威账本、内容指纹增量失效、三重预算并发调度和
可恢复的阶段化快照。仅依赖 `numpy` 与 `PyYAML`，不使用容器或外部网络资源。

## 1. 环境准备（Windows）

1. 安装 Python 3.11 官方安装包，勾选 **Add python.exe to PATH**。
2. 打开 PowerShell 或 CMD 验证：

   ```bat
   python --version
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. 在工作区生成样本目录与可重复示例：

   ```bat
   python examples\run_demo.py
   python examples\interrupt_recovery_demo.py
   python examples\concurrent_subprocess_demo.py
   python -m unittest discover -s tests -v
   ```

CLI 也可直接使用：

```bat
python -m quantcalib init --workdir workspace
python -m quantcalib run --workdir workspace --out workspace\out\result.json --report workspace\out\report.json
python -m quantcalib stream --workdir workspace --commands workspace\commands.json --out workspace\out\stream.json
```

## 2. 目录结构

- `quantcalib/data/`：固定整数种子样本、权重包 JSONL、校准集、YAML 配置加载校验。
- `quantcalib/parsers/`：q1 定宽文本、q2 JSON、q3 分区二进制解析、统一表示、旁路还原。
- `quantcalib/calib/`：固定分位规则、对称/非对称缩放因子与零点推导。
- `quantcalib/quant/`：定点整数、饱和截断、反量化误差和四来源归因。
- `quantcalib/distill/`：温度软标签校正、逐样本偏差和偏差恶化归因。
- `quantcalib/persistence/`：SHA-256 内容指纹、缓存键、原子快照和恢复判定。
- `quantcalib/schedule/`：全局/单层/单批预算、稳定规划顺序、有界线程池和命令流。
- `quantcalib/ledger/`：多版本回执权威条目裁决。
- `examples/`：完整可重复命令流、并发子进程、中断恢复演示。
- `tests/`：25 个 `unittest` 用例，覆盖业务码、数学、账本、预算、恢复和一致性。

## 3. 确定性规则

### 分位与并列值

- 组内先升序排序；分位位置 `p = permille / 1000 * (n - 1)`。
- 位置落在两个下标之间时使用线性插值；两端值相等（并列）时固定采用较低下标。
- 该规则与插值类型、并列决断一起写入每组产物元数据。

### 缩放因子和零点

- 对称：`raw_step = max(|q_lo|, |q_hi|) / qmax`，零点固定为 0。
- 非对称：`raw_step = (q_hi - q_lo) / (qmax - qmin)`，
  零点由 `qmin - q_lo / raw_step` 四舍五入并夹到整数表示区间。
- 步长按 `fractional_bits` 吸附到定点网格，因此切换小数位宽或量化档位会
  真实改变内部整数；`step_residual` 保留吸附前后差值。

### 定点内核和归因

- 舍入模式支持 `half_even`、`half_up`、`floor`。
- 饱和策略 `saturate` 夹断，`wrap` 按二补数风格回绕；逐元素保存截断方向和
  截断前定点值。
- 超差元素按固定优先序裁决：
  `saturation > quantile_discretization > scale_quantization > dequant_rounding`，
  份额并列时优先序靠前的来源胜出，判据数值写入 `source_evidence`。

## 4. 三版本回执与权威账本

- q1：16 字符定宽层名 + 6 位千分比覆盖率。
- q2：JSON 数组，新增 int32 参数版本号与 int64 毫秒耗时。
- q3：大端分区二进制，魔数 `QCQ3`，新增分组缩放摘要、重复标定引用和权重快照。
- 统一表示 `UnifiedReceipt` 保留来源版本、缺失字段和每个字段的原始精度标签。
- q3 降 q1 时，q3/q2 专属字段写入 JSON sidecar；升回 q3 只从 sidecar 还原，
  缺失 sidecar 会报错，禁止补默认值。
- 裁决顺序：参数版本号 → 被覆盖重复引用数量 → 格式丰富度 → 覆盖率，
  最终按层名字典序输出，稳定且不依赖线程或插入顺序。

## 5. 指纹、失效边界与预算顺序

缓存键由以下 SHA-256 指纹组成：

- 层内容指纹；分组内容指纹；
- 量化参数字典（方案、分位、小数位、舍入、饱和、容差）；
- 量化档位（位宽与 levels）；
- 校准集指纹；
- 分组键额外包含分组粒度，防止旧分区产物被新粒度误用。

失效边界：

- 替换单层权重：仅该层受影响分组的标定/量化/误差，以及全局蒸馏/账本下游失效；
  其他层直接命中。
- 调整位宽：仅目标层量化与误差失效，标定保留命中。
- 调整粒度：目标层重新分区，新键自然未命中，其他层不受影响。
- 蒸馏参数变更：仅蒸馏阶段重算，分组标定和量化全部命中。
- 回执版本切换：只改变账本权威候选，不使量化产物失效。

预算规划严格按 `(layer 字典序, group_index, stage)` 排序：

1. 单层配额不足：该组记 `E_QUOTA_EXCEEDED`，不消耗预算。
2. 全局预算不足：进入待办队列并记 `E_POINT_BUDGET_PENDING`。
3. 单批预算耗尽：开启下一批，不改变整体规划顺序。
4. 线程池只执行已规划单元，结果按规划顺序收集；最大并发不超过配置值。

## 6. 快照和恢复

- 每次运行先把现有快照复制到 `snapshot.tmp`，所有阶段 JSON 写入临时树。
- 提交时写 manifest，再用同文件系统原子 `os.replace` 整体替换。
- manifest 记录全局版本、配置哈希、权重包指纹、校准集指纹、回执版本指纹、
  已完成层、已消耗预算和每层预算。
- 陈旧 `.tmp` 在下次 `begin_commit` 自动回收；崩溃不会暴露部分提交。
- 快照是内容寻址的持久存储，配置变化由缓存键精确失效；只有结构缺失或
  全局版本回退才整体作废。

## 7. 验收产物

运行 `python examples/run_demo.py` 后查看：

- `workspace/out/incremental_result.json`：9 步命令流后的增量结果。
- `workspace/out/full_rebuild_result.json`：空白隔离快照的一次性全量结果。
- `workspace/out/diagnostic_report.json`：分组误差归因、蒸馏偏差、权威裁决、
  失效边界、复用统计和预算诊断。
- `workspace/out/demo_summary.json`：逐字段相等结论与运行摘要。

当前本地验证结果：全部 25 个自动化测试通过；示例命令流最终
`field_level_equal = true`，串行与并发、增量与全量、中断恢复后的结果逐字段一致。
