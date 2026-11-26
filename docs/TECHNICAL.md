# fxrender 技术说明

`fxrender` 是一个纯本地、确定性、定点整数域的多模态条件渲染包。它把文本词袋、
PNG 图像条件、分块运动矢量融合为潜变量，执行逐去噪步生成、逐帧解码与按分块行
PNG 输出，并以内容指纹驱动精确增量复用、预算调度、双版式帧记录与快照恢复。

## 1. 环境与运行

- Python 3.11（已在 WSL Ubuntu 验证），仅使用 CPU，无 GPU、无外部服务。
- 虚拟环境位于 `.venv/`，依赖见 `requirements.txt`：`numpy`、`Pillow`、`PyYAML`。
- Tk 支持：该环境无法 `apt install python3-tk`（sudo 需密码），因此把完整
  Tcl/Tk 运行时置于 `vendor/tk/`，激活后 `source .venv/bin/tk_env.sh` 即可
  `import tkinter` 并创建 Tk 根窗口；无显示时 UI 自动降级为文本摘要。
- 回环 `127.0.0.1` 与临时目录可写均已确认；样本由固定整数种子本地生成。

```bash
python3 -m venv --without-pip .venv
. .venv/bin/activate
# 该基础解释器缺 ensurepip，用基础环境 pip 引导到 venv
PYTHONPATH=/home/zsl/.feedback/python-base/lib/python3.11/site-packages \
  python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
source .venv/bin/tk_env.sh
pytest -q
```

## 2. 模块与需求映射

| 需求 | 实现位置 | 关键点 |
| --- | --- | --- |
| 1 加载校验 | `loader.py`, `config.py`, `codes.py` | 逐条 `CheckItem`，越界返回固定业务码并跳过，不静默修正 |
| 2 v1/v2 往返 | `formats.py` | 统一 `FrameRecord`，v2→v1 旁路元数据，再升回逐字段一致 |
| 3 定点编码/融合 | `fixedpoint.py`, `encoding.py`, `rng.py` | 显式循环、类型名/维度序、量化步长与残差、无全局随机 |
| 4 潜变量/去噪/解码 | `kernel.py` | 固定运算顺序、确定性舍入、分块行读集守卫 `E_TILE_BORROW` |
| 5 指纹失效/账本 | `fingerprint.py`, `cachestore.py`, `pipeline.py` | 阶段缓存键含上游指纹/参数/量化档位/交换版本，可复核 |
| 6 策略/预算/待办 | `policy.py`, `scheduler.py` | 跨档位表、合步、解码顺序、`E_BUDGET_PENDING`、不重复扣点 |
| 7 并发等价 | `pipeline.py` | 线程池上限取配置，稳定排序键，写一次性加锁，结果按键排序 |
| 8 交互/快照/恢复 | `interactive.py`, `snapshot.py` | 预算内最新摘要、陈旧码、临时目录原子替换、信号落盘 |

## 3. 三个约束冲突的处理

### 3.1 条件规模跨档位跃迁 vs 增量复用
`policy.select_policy` 仅依据帧数、分块行与每步代价从预置表选择
`batch_steps`、是否合步、解码顺序。档位参数进入去噪/分块缓存键的参数指纹，
因此档位跃迁会真实改变渲染路径与点数账本，而未变化的条件编码仍按内容指纹复用。

### 3.2 单批去噪点数预算
`Scheduler.run_batch` 每次获得刷新的批预算，但待办队列跨批保留。按
`(phase, frame, step, tile_row, group)` 稳定序分配，首个放不下的步次及其后
全部记入待办并写 `E_BUDGET_PENDING`；已扣点键集合去重，恢复/续算不会重复扣减。

### 3.3 分块行禁止借用相邻像素
`TileAccessGuard` 为每个分块行计算允许读取的潜变量索引区间，`decode_tile_row`
只能经守卫读取；越界立即抛 `TileBorrowError`（业务码 `E_TILE_BORROW`），
分块行摘要记录实际读取的潜变量元素，缓存键也绑定所属分块行。

## 4. 定点与确定性

- `FpSpec` 的位宽/档位是枚举：宽度 16/32/64，档位 coarse/standard/fine，
  非法组合抛 `E_FP_ENUM`；切换档位改变真实整数，而非仅改变显示。
- 量化使用远离零的半入舍入；乘法保留移位残差，加法做宽度钳制。
- 随机源 `DeterministicRNG` 基于 SplitMix64，由融合指纹、种子、配置指纹、
  帧号派生，独立于 numpy 全局 RNG。
- 融合严格按条件类型名升序、维度序号升序在 Python 整数域累加，并记录每步
  量化步长与残差，不依赖数组归约默认顺序。

## 5. 缓存键与失效

缓存键 = BLAKE2b(stage, 上游内容指纹, 参数指纹, 量化档位, 交换版本)。
`RenderEngine.invalidate_for_change` 把编辑 scope 映射到阶段依赖闭包：
文本/图像/运动 → 融合 → 去噪 → 帧 → 分块行。局部编辑只失效引用它的帧，
全局编辑（权重、运动、量化、交换版本）按依赖闭包失效；未命中失效的产物在
后续渲染中以 `reused` 命中，账本逐条记录 computed/reused/invalidated 与原因。

## 6. v1/v2 帧记录

- v1 为定宽文本行（列宽常量 `V1_LINE_WIDTH`，魔数 `FXF1`），缺失字段以 `*` 标记。
- v2 为 JSONL（魔数 `FXF2`），含浮点引导强度、命名噪声方案、精度标签。
- v1→v2：千分比精确除以 1000，噪声码经码表映射为方案名。
- v2→v1：四舍五入回千分比，截断残差、精度标签、被覆盖字段写入 sidecar；
  再升回 v2 时从 sidecar 还原，与直接解析 v2 的内部表示逐字段相等。

## 7. 快照与终止恢复

阶段产物、待办队列、帧/分块游标、已消耗预算先写入临时目录，提交前比对全局
版本号与配置哈希，不匹配返回 `E_SNAPSHOT_VERSION` 且不覆盖线上快照；匹配则
`os.replace` 原子整体替换。`TerminationHandler` 处理 SIGTERM/SIGINT 时尽力
落盘快照与诊断报告，但不承诺一定生成退出报告；恢复只信任已提交快照。

## 8. 验证

`tests/` 共 48 个用例，覆盖定点/随机、双版本往返、加载校验业务码、编码融合、
分块借用、策略档位、预算待办、编辑失效、增量-全量像素一致、并发无关、
快照恢复与信号处理。确定性样本可重复生成且字节一致。
