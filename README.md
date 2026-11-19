# 本地关卡约束求解引擎

一个零外部服务依赖的程序化关卡（PCG）约束求解引擎。输入 JSON 关卡规格，
在网格地图上确定性地分配房间与走廊，满足连通性、间距与区域面积约束，
并支持分步快照、中断续算与快照防篡改。

## 1. 环境与安装

- Windows 10+ 或 Linux，Python 3.10+（开发验证使用 Python 3.11 / NumPy 2.4）
- 依赖：`numpy>=1.24`

```bash
python -m pip install -r requirements.txt
```

不使用 Docker、外部账号、数据集或网络资源；样本由程序按确定性规则本地生成：

```bash
python samples/generate_samples.py
```

## 2. 目录结构

```
level_solver/
  errors.py         # E_xxx 业务错误码与异常类型
  prng.py           # SplitMix64 确定性伪随机（不依赖全局随机状态/系统时间）
  grid.py           # 网格模型、4-连通分量、L 形走廊几何
  parser.py         # JSON 规格解析与定义域校验
  constraints.py    # 五类约束：预检/候选枚举/应用/满足性检查
  precheck_pool.py  # 有界线程池并发预检（并发数+内存预算+取消汇总）
  snapshot.py       # 版本标识、SHA-256 摘要、原子写入、归档、篡改判定
  solver.py         # 顺序求解 + 迭代回溯 + 快照提交 + 中断/恢复
run_solver.py       # CLI：solve / verify 子命令
run_demo.py         # 20 项端到端验收场景
samples/            # 确定性样本生成器与样本
tests/              # 43 个 unittest 用例（标准库，无需 pytest）
```

网格状态：`0=空，1=不可用(blocked)，2=可行走(walkable，房间或走廊)`。

## 3. 规格格式

```json
{
  "width": 16,
  "height": 14,
  "seed": 20261119,
  "blocked": [[0, 0]],
  "concurrency": 4,
  "memory_budget": 8,
  "max_rollback": 12,
  "alternative_limit": 32,
  "constraints": [
    {"id": "rooms", "type": "min_rooms",
     "params": {"count": 5, "min_room_w": 2, "min_room_h": 2,
                "max_room_w": 4, "max_room_h": 4}},
    {"id": "spacing", "type": "room_spacing", "params": {"min_distance": 1}},
    {"id": "region_cap", "type": "max_region_area", "params": {"max_area": 120}},
    {"id": "all_connected", "type": "connectivity", "params": {}},
    {"id": "start_to_end", "type": "reachability",
     "params": {"start": [1, 1], "end": [14, 12]}}
  ]
}
```

允许的约束类型：`min_rooms`、`room_spacing`、`max_region_area`、
`connectivity`、`reachability`。约束严格按声明顺序应用。

## 4. 业务错误码

| 代码 | 含义 | CLI 退出码 |
| --- | --- | --- |
| `E_SCHEMA` | JSON 结构错误（非对象、缺字段、约束非对象等） | 1 |
| `E_DOMAIN` | 定义域错误（尺寸非正、坐标越界、间距为负、房间数超可用格等） | 1 |
| `E_CONSTRAINT` | 约束类型非法或预检不可行 | 1 |
| `E_CONFLICT` | 回溯深度用尽仍无法满足约束，含冲突约束对与约束标识 | 2 |
| `E_DIGEST` | 快照摘要缺失/不匹配，快照失效 | 3 |
| `E_VERSION` | 快照版本标识与当前规格不一致 | 4 |
| `E_IO` | 文件读写失败 | 1 |
| 中断 | 可处理终止信号挂起 | 130 |

校验在任何快照/输出文件创建之前完成，非法输入不会在磁盘留下产物。

## 5. 确定性设计

- 随机全部来自自研 SplitMix64 `DeterministicRNG`，种子由规格 `seed` 派生；
  不使用 `random` 全局状态、NumPy 全局 RNG 或系统时间。
- 每个约束使用 `substream(约束序号 * 常量 + 该约束重试次数)` 派生独立流；
  回溯重试因此会产生不同的候选序列，而候选枚举本身仍完全确定。
- 候选位置、组件锚点、走廊弯折方向均由该流驱动；同一规格+种子重复求解，
  最终网格逐格相同、约束应用计数相同。

## 6. 冲突检测与回溯

- 每个候选动作应用后，重新校验所有**已提交**约束；任一被破坏即记录冲突对
  （当前约束 id、被破坏约束 id、原因），尝试下一候选。
- 候选耗尽或预检失败时，回退到最近一次已提交快照，递增该约束重试计数，
  使用新的确定性子流重新枚举。
- 回退次数达到 `max_rollback` 后抛出 `E_CONFLICT`，details 中给出无法满足的
  约束标识、冲突列表与已用回退深度；不会输出部分满足的网格文件。

## 7. 快照、版本与防篡改

- 每成功提交一个约束写入 `snapshot_NNNN.json`（外加基线 `snapshot_0000.json`），
  内容含网格、房间台账、帧序号、重试计数、RNG 状态、种子、冲突列表、
  约束计数与规范化规格摘要。
- 版本标识 = 对引擎版本+网格+不可用格+种子+并发/预算参数+约束集合做规范化
  JSON 的 SHA-256 前 16 位。
- 每份快照对其 `content` 计算 SHA-256 存入 `digest`；恢复时重算比对，
  缺失或不一致即判 `E_DIGEST`。写入采用临时文件 + `os.replace` 原子替换。
- 版本变化时不覆盖旧快照：旧 `current/` 整体移动到 `archive/<version_id>/`，
  再建立新的当前目录，并在 `index.json` 登记归档。

## 8. 中断与续算

- 在约束边界检查信号（SIGINT/SIGTERM）或 `--interrupt-after N` 测试钩子；
  挂起时持久化最新快照并写 `interrupted.json`，以退出码 130 退出。
- 再次启动读取中断标记与最新快照：版本一致则从下一约束继续，结果与不间断
  求解逐格一致；版本不一致则从头重做。
- 最新快照被篡改时拒绝该快照，回退到最近一份有效快照并附带 `E_DIGEST` 警告；
  全部失效才从头开始。强制终止（如 SIGKILL）无中断标记时仅依赖已原子持久化
  的快照，不承诺退出报告。

## 9. 并发预检

- `BoundedPrecheckPool` 使用固定大小 `ThreadPoolExecutor`，工作线程数取
  `concurrency`；用 `BoundedSemaphore(memory_budget)` 限制同时驻留的候选
  状态数量（内存预算），且要求 `memory_budget >= concurrency`。
- 预检任务只读取独立网格副本，无共享可变状态；按规格顺序编号，首个失败即
  `cancel_futures=True` 取消其余任务，并汇总最早失败的业务错误。
- 最终约束应用顺序与串行预检完全一致（测试中用 concurrency=1 与并发模式
  对比最终网格与计数）。

## 10. CLI 用法

```bash
# 求解（可自动续算）
python run_solver.py solve --spec samples/sample_valid.json \
    --snapshot-dir output/run --output output/result.json

# 在第 2 个约束边界确定性挂起（退出码 130）
python run_solver.py solve --spec samples/sample_valid.json \
    --snapshot-dir output/run --interrupt-after 2

# 忽略历史快照从头求解
python run_solver.py solve --spec samples/sample_valid.json \
    --snapshot-dir output/fresh --no-resume

# 校验快照目录中所有快照摘要
python run_solver.py verify --spec samples/sample_valid.json \
    --snapshot-dir output/run
```

## 11. 测试与验收

```bash
python -m unittest discover -s tests -v   # 43 个用例
python run_demo.py                        # 20 个端到端验收场景
```

覆盖：解析与全部错误码、PRNG 确定性、五类约束满足、逐格可复现、回溯恢复、
`E_CONFLICT` 不落盘部分网格、每个约束边界的中断/续算一致性、摘要篡改拒绝、
回退到旧有效快照、版本归档、并发预检内存上界与取消、真实 SIGINT 子进程挂起续算。
