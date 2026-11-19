# 运行手册

## 1. 环境准备

要求 Python 3.11+。本环境系统 Python 不带 pip，使用 get-pip.py 在项目内
虚拟环境安装 NumPy（无容器、无外部设备）：

```bash
python3 -m venv .venv
# 若 venv 内缺 pip：
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/python -m pip install -r requirements.txt
```

实测版本：Python 3.11.0rc1、NumPy 2.4.6。

## 2. 生成确定性样本

```bash
.venv/bin/python scripts/generate_sample.py --seed 42 \
    --output samples/manifest.json
```

样本含 4 个不同采样率/帧时长/量化档位的设备、20 个随机事件（同种子可复现）、
2 个重叠叠加事件与 2 个同步锚点。`--strict-sync` 可把锚点允许偏差收紧到 1。

## 3. 运行合成

```bash
# 并行（受 global.concurrency / inflight_limit 约束）
.venv/bin/python -m haptic_engine.cli samples/manifest.json

# 串行参考路径（并发度 1），用于逐条比对
.venv/bin/python -m haptic_engine.cli samples/manifest.json --serial

# 需要观察中断时，给每个帧块加延迟
.venv/bin/python -m haptic_engine.cli samples/manifest.json --block-delay 0.05
```

输出：

- `out/commands.json`：原子写出的正式指令流。
- `out/commands.report.json`：同时打印到 stdout 的 JSON 报告。
- `out/state/`：版本文件、逐设备帧块、中断标记与归档目录。

退出码：`0` 成功；`2` 输入校验错误；`3` 同步越界（指令流仍写出）；
`4` 摘要失效且无法修复；`130` SIGINT 边界中断；`143` SIGTERM 边界中断。

## 4. 可处理终止信号的本地测试

Ctrl+C 等价于发送 SIGINT。终端一：

```bash
.venv/bin/python -m haptic_engine.cli samples/manifest.json --block-delay 0.05
# 按 Ctrl+C
```

或用第二个终端 / 脚本：

```bash
kill -INT <pid>   # 期望退出码 130，写 out/state/interrupt.json
kill -TERM <pid>  # 期望退出码 143
```

信号只在帧块边界生效，未完成块只可能残留隐藏临时文件，不会被当作完成块。
再次执行同一命令即按版本与块摘要恢复，仅补算缺失块。强制终止可用
`kill -9 <pid>`（退出码 137），重启不依赖中断标记，仅信任已持久化且
摘要有效的块。

## 5. 运行测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 6. 实际验证结果（本机实测）

### 单元/集成测试

`Ran 32 tests ... OK`，覆盖：

- 17 项输入校验（标识、正整数、量化档位、强度有限性与范围、曲线时序/
  覆盖、目标设备、预算下限、布尔值不得冒充整数等），且非法输入后无状态目录。
- 采样点映射与 ≤1 采样点对齐、事件区间与合成顺序无关、量化边界确定性。
- 装箱 RLE、每帧指令上限、溢出拆分后逐采样与逐点下发完全等价、
  持续时长总和等于采样点数。
- 串行/并行指令流逐条相同、归并按设备与采样点序、在途峰值不超预算。
- 块损坏与声明条数不符触发 `E_DIGEST` 补算并与干净运行一致。
- 版本变更归档旧快照并重做、二次运行复用已完成块（补算数为 0）。
- 同步越界抛 `E_SYNC` 但仍导出合法指令流。

### 端到端样本（seed=42，干净目录）

| 设备 | 指令条数 | 溢出指令 | 帧块数 |
| --- | --- | --- | --- |
| glove-a | 123 | 68 | 66 |
| pad-left | 123 | 63 | 52 |
| trigger-r | 178 | 57 | 58 |
| vest-core | 47 | 2 | 20 |

在途峰值 `peak_inflight_commands = 8`，等于预算上限 `inflight_limit = 8`，
预算等待事件 180 次，全程未超限；两个锚点偏差均为 0，退出码 0。

### 中断/恢复实测

- `kill -INT` → 退出码 130，`interrupt.json` 记录 `interrupt_count=1`、
  `pending_resume=true`，无隐藏临时文件残留。
- `kill -TERM` → 退出码 143；`kill -9` → 退出码 137（无中断标记）。
- 三种情况重启后退出码 0，指令流与未中断一次跑完的参考结果逐条相同；
  SIGTERM 续跑报告 `interrupt_count=1`，SIGKILL 续跑为 0。

### 摘要失效补算实测

手工篡改某块的指令 `level` 或虚增 `command_count` 后重启：损坏块被识别、
删除并补算，报告 `digest_failures >= 1` 且产生 `E_DIGEST` 补算事件，
最终指令流与干净运行一致。

### 版本归档实测

修改设备量化档位后重启：版本标识变化，旧快照整体移动到
`state/archive/<old-version>/`，新版本从头重做，`version.json` 已更新。
