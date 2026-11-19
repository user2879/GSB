# 文件系统权限与路径检查审计引擎 — 技术说明

## 1. 设计目标与范围

引擎对本地确定性构造的 JSON 清单做离线越权风险检测，不访问任何网络资源、
真实受检目标或外部数据集。链接声明只作为虚拟路径解析输入，不依赖真实
文件系统符号链接/硬链接对象，因此在无开发者模式或无链接权限的
Windows 工作区同样可运行。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `audit_engine/errors.py` | 业务错误码常量与带记录序号的解析异常 |
| `audit_engine/models.py` | 记录、链接、ACL、策略、裁决等不可变数据模型 |
| `audit_engine/parser.py` | 逐条 JSON 校验，非法记录拒绝但不中断整批 |
| `audit_engine/pathnorm.py` | 点段/父目录段、Unicode NFC、大小写折叠与链接解析 |
| `audit_engine/permission.py` | 两种权限口径统一解析、八进制舍入误差界与传播 |
| `audit_engine/acl.py` | 顺序无关的有效权限裁决、拒绝覆盖与继承检测 |
| `audit_engine/engine.py` | 多策略裁决、策略冲突识别、预算熔断、真值对照 |
| `audit_engine/budget.py` | 真实墙钟超时与单文件记录数预算 |
| `audit_engine/locking.py` | POSIX/Windows 文件锁与临时文件原子重命名 |
| `audit_engine/checkpoint.py` | 检查点版本校验、整体重建与恢复 |
| `audit_engine/runner.py` | 串行/线程/独立子进程三种后端与退出码 |
| `audit_engine/truth.py` | 只依据同步标记字段的真值对照 |
| `audit_engine/generator.py` | 确定性样本生成器（无时钟、无随机熵） |
| `audit_engine/workload.py` | 测试用确定性 CPU 探针（真实墙钟负载） |
| `audit_engine/cli.py` | 命令行入口 |

## 3. 路径规范化与比较误差量化

`normalize_path` 的处理顺序固定为：

1. 虚拟路径拆分，兼容 POSIX、Windows 反斜杠与盘符；
2. 点段剔除、父目录段栈式消解，量化越界深度 `traversal_depth`；
3. 按策略执行 NFC 与 `casefold`，分别累计 `encoding_error`、
   `casefold_error`（被改变的码点数）；
4. 策略允许时按链接声明重建可达路径，累计 `link_steps`，
   链接环在固定跳数内终止；
5. 落点不在受检根内返回 `E_LINK_ESCAPE`，根边界回退返回
   `E_TRAVERSAL`。

`path_error.total = encoding + casefold + link_steps`，
作为路径比较编码误差传播的单值量化输出。

## 4. 权限数值口径与误差界

- `int_mask`：整数位掩码精确解析，误差上界恒为 0；未知位或负值拒绝。
- `octal_string`：支持 `"751"` 与 `"751.24"` 形式，小数按八进制进位
  （2/8 + 4/64 ...）。
  - `truncate`：误差上界等于被丢弃的八进制小数；
  - `nearest`：四舍五入到最近整数，误差上界 1/2 单位。
- 内部统一使用 `fractions.Fraction` 精确运算，对外输出 `float`；
  本引擎涉及分母均为 2 的幂，Fraction→float 不引入额外表示误差。
- 传播链（父目录权限 + ACL 继承）按位或合并点估计，误差上界按各来源
  绝对值线性求和，给出保守最坏情形界。

### 临界不确定（按位稳定性）

对所需权限的每个位，计算翻转该位所需的最小数值偏差（等于该位位值）；
若累计误差界达到该距离，则该位在误差邻域内可能翻转，记录显式标记
`critical=true`，且不输出确定性越权结论。样本 `MIX-003` 用三条
八进制来源构造累计界 189/64 ≥ 2，稳定触发写位临界。

## 5. ACL 裁决与顺序无关性

条目先按 `(source_depth, access, mask, 原始 order)` 做全序排序再处理，
输出仅含语义键（掩码、深度）并确定性排序，因此同一 ACL 集合在任意
输入排列下有效权限、`E_DENY_SHADOW` 与 `E_INHERIT` 清单完全一致。

- `E_DENY_SHADOW`：deny 位被同级或更近（深度 ≤）的 allow 抵消且无生效位；
- `E_INHERIT`：inherited allow 授予的位在全部非继承 allow 中缺失且无
  对应 deny 约束。

## 6. 策略冲突

同一记录在策略集合内逐策略裁决；若违规码签名（codes、规范键、
可达路径、临界标记）在策略间不一致，附加 `E_POLICY_CONFLICT` 并在
文件级 `policy_conflicts` 中列出所涉策略组合。NFC/大小写口径会改变
链接匹配，进而改变链接逃逸结论，是该冲突的典型来源。

## 7. 真值对照

生成器在每条合法记录内显式写入 `truth`：固定同步标记、原始路径字节
hex、原始权限数值、预期误差界 `[lo, hi]`。验证模块只读取这些字段，
不引用生成器任何中间计算；误差界落在区间外、路径字节或原始权限值
不一致即 `E_TRUTH_MISMATCH`。报告给出一致计数与不一致明细。

## 8. 资源预算、并发与终止边界

- 预算三参数可配置：`concurrency`、`per_file_timeout_seconds`、
  `max_records_per_file`。
- 超时判定基于 `time.monotonic()` 真实墙钟，无任何人为延迟；
  超记录数返回 `E_BUDGET`，超时返回 `E_TIMEOUT`。
- 三种后端：
  - `serial`：单进程顺序，确定性基准；
  - `thread`：线程池，记录级裁决为纯函数，输出与串行按
    `(record_id, policy_name)` 签名比对完全一致；
  - `process`：每个文件独立子进程 + 信号量限流，外层 `join(timeout)`
    超时后对该进程执行真实 `terminate()`，被终止文件不影响同批其余文件。
- 进程被外部强制 kill 时不承诺生成退出报告（只保证正常完成路径的
  原子报告）。

## 9. 检查点、锁与原子写

- 检查点保存已完成文件标识与聚合状态；恢复时跳过完整检查点中的文件。
- `struct_version` 或规范化策略集合变化时整体重建（不增量续接）。
- 检查点与报告均先写同目录 `.part` 临时文件，`fsync` 后 `os.replace`
  整体替换；NTFS 与 POSIX 均提供原子重命名语义。
- 同目录并发写检查点经 `flock`（Windows 为 `msvcrt.locking`）串行化，
  非阻塞抢锁失败返回 `E_LOCK`。

## 10. 退出码

| 码 | 含义 |
| --- | --- |
| 0 | 全部文件无违规 |
| 1 | 存在业务违规（E_PARSE/E_TRAVERSAL/...） |
| 2 | 至少一个文件超时或超预算 |
| 3 | 检查点锁冲突 E_LOCK |
| 4 | 致命输入错误 |

## 11. 验证结果

见 `VERIFICATION.md`，由本地确定性样本与 pytest 全量运行产出。
