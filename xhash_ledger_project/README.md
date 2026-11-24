# 跨模态哈希签名与近重复簇账本组件

面向本地确定性图文样本（固定分辨率 PNG + 受控词表文本断言），把每个样本压缩为一段
跨模态二值哈希签名，并以签名分桶维护近重复簇账本。支持分量级精确失效、增量运行与
从空白全量重建逐字段一致、点预算/配额调度、线程池并发与阶段快照恢复。

仅依赖 Python 3.11 标准库与 NumPy，不访问网络，不使用容器。

## 目录结构

```
xhash_ledger_project/
├── xhash_ledger/
│   ├── errors.py        # 错误码与失败清单（E_SAMPLE_INVALID 等）
│   ├── fixedpoint.py    # Qm 定点整数量化、整数点积、量化步长/残差
│   ├── pngio.py         # 纯标准库 PNG 编解码（支持全部 5 种行过滤器）
│   ├── hashing.py       # SHA-256 内容指纹、缓存键、确定性派生
│   ├── config.py        # 配置/命令校验、配置哈希、投影指纹派生口径
│   ├── validation.py    # 样本三件套加载与字段校验
│   ├── features.py      # 分块图像特征 + 断言计数 + 模态权重联合特征
│   ├── signature.py     # 定点投影、符号决胜、签名分桶、汉明距离
│   ├── clustering.py    # 同桶候选、汉明升序合并、簇 ID 重指定
│   ├── cache.py         # 四级缓存键与分量级失效引用关系
│   ├── scheduler.py     # 点预算/单样本配额、待办队列、线程池
│   ├── state.py         # 临时目录 + 整体替换提交、manifest 恢复判定
│   ├── generator.py     # 固定种子样本/近重复/非法样本/命令流生成
│   ├── engine.py        # 增量持久化引擎与全量重建（同一套纯函数）
│   └── cli.py           # generate / run / rebuild / resume 命令行
├── tests/               # 39 个 unittest 用例，覆盖业务要求 1-8
├── demo.py              # 可重复演示：增量 vs 全量逐字段比对
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 生成确定性工作区（10 基础样本 + 2 近重复 + 3 非法样本 + 配置与命令流）
python -m xhash_ledger.cli generate demo_ws --samples 10

# 按命令流增量运行（产出 reports_incremental.json 与 state/ 快照）
python -m xhash_ledger.cli run demo_ws

# 从空白对最终样本集合与最终配置一次性全量重建
python -m xhash_ledger.cli rebuild demo_ws

# 快照恢复判定（以全局版本号 + 配置哈希为准）
python -m xhash_ledger.cli resume demo_ws

# 一键演示并自动断言逐字段一致（退出码 0 即通过）
python demo.py --workspace demo_ws --samples 10

# 测试
python -m unittest discover -s tests -v
```

Windows 10+ 与 Linux/macOS 均可运行；路径全部使用 `os.path`，文件提交使用
`os.replace` 原子替换，无任何符号链接或 Shell 依赖。

## 关键设计与业务要求对应

### 1. 加载校验层（`validation.py` / `config.py`）
- 样本三件套：`<id>.png`、`<id>.assertion.json`、`<id>.regions.json`。
- 校验 PNG 容器签名与 IHDR、固定分辨率一致性、区域坐标为整数且落在图内、
  断言实体/属性字段完整且在受控词表内、样本标识唯一。
- 配置校验：签名位宽为 8 的正整数倍、分桶位数小于签名位宽、相似度千分比 ∈ [0,1000]、
  种子版本在已声明集合、桶容量为正整数。
- 非法项分别产出 `E_SAMPLE_INVALID` / `E_CONFIG_INVALID` 失败记录并跳过该项，不中断其余样本。

### 2. 确定性联合特征与定点量化（`features.py` / `fixedpoint.py`）
- 图像按 `tile_size` 切块，每块 5 个统计量：RGB 颜色均值、平均梯度强度、边缘密度。
- 文本侧对受控实体类型与属性名计数；两侧分别乘模态权重后拼接为联合特征。
- 量化为 Qm 定点整数（`qstep = 2^-frac_bits`），逐样本记录量化步长与逐分量残差；
  投影矩阵同为定点整数，投影通过整数乘加完成，无浮点求和顺序问题，两次运行逐元素一致。

### 3. 二值哈希签名与分桶（`signature.py`）
- 投影矩阵种子 = `种子版本 | 签名侧配置指纹 | 特征维度`，与分桶/阈值参数解耦。
- 固定决胜规则：投影恰为零 → 位取 1（记录 `zero`）；`|投影| ≤ 量化半步` → 位取 0
  （记录 `half_step`）；其余正取 1、负取 0。逐位决胜位置全部入报告。
- 按 `bucket_bits` 把签名切成子串并映射为定长十六进制桶键，记录每个桶引用的签名位集合。
- 位宽或种子版本变化会真实改变位模式（测试 `test_seed_version_changes_bit_pattern` 等）。

### 4. 近重复簇账本（`clustering.py`）
- 仅在共享至少一个桶的样本间产生候选对；按 `(汉明距离, 样本标识)` 升序稳定合并。
- 距离阈值由相似度千分比换算；簇标识取成员最小样本标识；成员记录簇内位次与合并相似度。
- 合并日志保留被合并簇标识、样本对、距离、相似度与桶键。
- 删除样本后按“剩余成员最小标识”固定规则重指定簇 ID 并记录来源，无悬空成员、无重复簇 ID。

### 5. 分量级精确失效（`cache.py`）
- 四级缓存键：feature（内容指纹 + 特征配置）、signature（feature 键 + 种子版本 + 位宽）、
  bucket（signature 键 + 分桶位数）、cluster（全量桶集合 + 阈值 + 位宽）。
- 记录每个签名位引用的特征分量集合、每个桶引用的签名位集合。
- 替换单样本仅该样本的三级产物重算，其余样本全部 `reused`；调整分桶位数时签名全部复用、
  仅桶与簇重建（见 `test_bucket_bits_change_reuses_signatures`）。

### 6. 增量与全量逐字段一致（`engine.py`）
- 增量路径与全量重建共用同一套特征/量化/签名/分桶/聚类纯函数，差别仅在缓存命中与否。
- `demo.py` 自动比对末态的逐样本签名、簇账本、缓存键集合与配置哈希，全部相等才返回 0。
- 复用清单在两种口径下都是确定性可复核产物：增量路径逐条记录 `reused/recomputed` 与命中理由、
  受影响分量编号；全量路径标记为 `recomputed`，其缓存键集合与增量路径完全相同。

### 7. 点预算、配额与待办（`scheduler.py`）
- 单样本成本 = 特征点 + 签名点（位宽 × 特征维度）+ 分桶点，按 `feature/signature/bucket`
  三阶段记录点扣减序列与稳定任务键 `sample_id|stage|bit_index`。
- 单样本超配额 → `E_SIGNATURE_QUOTA_EXCEEDED` 中止该样本；全局预算不足 →
  `E_SIGNATURE_BUDGET_PENDING` 进入待办；下一轮运行待办优先于新样本。
- 点扣减在主线程按稳定顺序进行，与线程池并发度无关（测试 `test_concurrent_and_serial_identical`）。

### 8. 并发与阶段快照（`scheduler.py` / `state.py`）
- 线程池最大并发数取 `max_workers` 且不超过就绪任务数；结果按样本标识稳定顺序归并。
- 特征量化、签名生成、分桶、簇维护四阶段产物先写入 `state/tmp/v<n>/<stage>/`，
  再整体替换到 `state/stages/v<n>/<stage>/`。
- `manifest.json` 记录全局版本号、配置哈希、种子版本、已完成样本与已消耗预算，
  并自带内容指纹防篡改；启动时以版本号 + 配置哈希判定续跑。
- 每次运行的 JSON 报告含逐样本签名、簇账本、失效复用清单、待办队列、失败清单与点扣减序列。

## 报告字段说明（每次运行输出）

| 字段 | 含义 |
| --- | --- |
| `manifest` | 版本号、配置哈希、种子版本、已完成/已安装样本、预算消耗 |
| `signatures` | 逐样本位串、签名整数、逐位决胜、桶键、桶引用位集合、量化步长/残差、缓存键 |
| `cluster_ledger` | 簇标识、成员位次与合并相似度、完整合并日志 |
| `cluster_reassignments` | 删除导致的簇 ID 重指定来源记录 |
| `reuse_log` | 四级产物的 reused/recomputed、命中理由与受影响分量编号 |
| `pending_queue` | 预算不足保留的待办样本（下一轮优先） |
| `point_deductions` | 按稳定顺序记录的逐阶段点扣减与任务键 |
| `failures` | 全部错误码记录，不影响其余样本处理 |

## 实际验证结果

- `python -m unittest discover -s tests`：39 个用例全部通过，覆盖 8 项业务要求。
- `python demo.py`：18 条编辑命令（新增/替换/删除/位宽 64→128/分桶 8→16/种子版本切换）后，
  增量末态与全量重建的签名、簇账本、缓存键集合、配置哈希逐字段一致；末轮 34 条复用记录中
  33 条命中，仅受影响样本重算。
- 并发（4 线程）与串行（1 线程）的签名、簇账本、点扣减序列完全相同。
- 预算收紧时样本按稳定顺序进入待办，下一轮待办优先消化且队列清空。
