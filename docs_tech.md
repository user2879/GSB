# 分层意图路由与槽位回填工作台 — 技术说明

面向本地确定性合成语料，实现“两级路由 → 槽位回填 → 置信度再校准”闭环，并保证
任意编辑序列下增量运行与从空白全量重建逐字段一致。仅依赖 Python 3.11 标准库与 NumPy。

## 1. 目录结构

- `workbench/` 核心库
  - `fixedpoint.py` Q16 二进制定点整数运算、十进制网格量化、非有限值拒绝
  - `errors.py` 全部诊断错误码常量
  - `io_utils.py` 规范 JSON（键排序、无空白）、SHA256 指纹、临时目录+`os.replace` 原子写
  - `models.py` 受控词表、层级配置、样本模型与配置分段指纹（实时读取可变 `raw`）
  - `loader.py` JSONL/JSON 逐行加载与四类校验，失败条目进入失败清单
  - `routing.py` 领域词袋加权命中归一化 + 触发/否定消解的领域内意图打分（NumPy 向量化）
  - `slots.py` 带窗口最长匹配槽位回填、候选决断、抑制清单与点数预算守卫
  - `recalibration.py` 定点整数再校准与多档位量化
  - `cache.py` 阶段缓存（条目携带话语 ID）与意图反向索引
  - `editing.py` 编辑计划应用（追加/迁域/槽位模式/阈值），冲突抛带码异常
  - `engine.py` 线程池执行、稳定排序键、预算账本、待办队列、快照
  - `pipeline.py` 编排、精确失效指引、全量重建与逐字段比对
  - `injections.py` 异常注入隔离运行与 JSON 诊断报告
- `scripts/gen_corpus.py` 确定性生成词表/配置/样本/编辑计划/注入清单
- `scripts/run_demo.py` 端到端演示（增量 + 全量 + 比对 + 注入）
- `tests/` 29 个 `unittest` 用例
- `data/` 生成的输入语料；`snapshots/` 阶段快照；`reports/` 诊断与一致性报告

## 2. 关键算法

### 2.1 定点与量化
所有分数以 Q16 定点整数（`1.0 = 65536`）运算，十进制配置经 `Decimal` 转换，杜绝
浮点不确定性。量化档位为十进制网格（粗 0.1 / 中 0.01 / 细 0.0001），切换档位会真实
改变再校准前/后量化值；再校准结果裁剪到 `[0,1]`。

### 2.2 两级路由
- 领域级：领域词袋中命中词项（去重）权重和 ÷ 词袋总权重，得分与领域阈值比较；
  并列时按领域名字典序决断并记录 `criterion`。
- 意图级：仅在已判领域的意图集合内匹配（禁止跨域）。触发词按“否定窗”内是否存在
  否定词消解，有效触发数 ÷ 模板触发词总数；并列或恰等于阈值时按意图名字典序决断。

### 2.3 槽位回填
按判定意图的必需槽位，在话语序列上做带窗口最长匹配。同槽多候选按
`(匹配长度降序, 起始位置升序, 取值字典序)` 决断，败者写入抑制清单；槽位名/取值
成对输出；必需槽缺失给 `E_SLOT_MISSING` 并保留已回填部分。点数按回填词项长度记账。

### 2.4 再校准
`adjusted = raw + 完整度×完整度权重 − 抑制数×抑制权重`，再施加下限并裁剪。完整度=
已回填必需槽占比。再校准前后置信度、三档量化值分别落盘。

### 2.5 精确失效
四类阶段缓存键由话语内容指纹、路由/映射/模板/槽位/再校准分段配置指纹派生。缓存条目
携带话语 ID，失效按话语身份精确删除，避免“配置切换后键已变导致旧条目删不掉”。
意图反向索引记录每个意图模板引用的触发词与话语集合；迁域或槽位模式修改仅失效引用
受影响模板的话语，未失效话语复用且不重新校准、不重复扣点。

### 2.6 预算调度
每批维护剩余回填点数，按话语 ID 稳定顺序（待办优先）消耗并写账本。超预算话语以
`E_SLOT_BUDGET_PENDING` 进入待办队列，下批优先处理；复用话语记 `reused` 且扣点为 0，
预算耗尽真实改变本批完成集合。

### 2.7 一致性与并发
- 全量重建使用与增量相同的缓存状态机，并对每个复用项额外重算做断言校验；最终逐字段
  比对领域/意图/槽位/抑制顺序/再校准/复用重算清单/失败清单/预算账本。
- 线程池并发（`max_workers` 取自配置），每个任务携带 `sha256(话语ID, 阶段名)` 稳定
  排序键；纯函数式阶段计算保证串行与并发结果一致。
- 快照先写临时目录再 `os.replace` 整体替换；清单记录全局版本号与配置哈希，恢复时
  哈希不符抛 `E_CONFIG_CHANGED`。

## 3. 运行方式

```bash
python3 -m virtualenv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/gen_corpus.py     # 生成 data/ 下确定性输入
python scripts/run_demo.py       # 增量+全量+注入，产出 snapshots/ 与 reports/
python -m unittest discover -s tests -v
```

## 4. 实际验证结果（本机）

- `python -m unittest discover -s tests`：29 用例全部通过。
- `run_demo.py`：6 个批次，15 条有效话语，失败清单 5 条（覆盖四类校验错误），
  增量与全量 `identical=true`，6 项注入全部命中唯一编号。
- 预算：阈值批次出现 `budget_pending`，后续批次待办优先出队并最终收敛，账本收支平衡。
- 量化：粗/细档位对同一置信度产出不同量化值，再校准前后值分别记录。

## 5. 错误码一览

`E_SAMPLE_INVALID`、`E_DOMAIN_UNKNOWN`、`E_INTENT_OUT_OF_DOMAIN`、
`E_SLOT_SCHEMA_INVALID`、`E_CONFIG_INVALID`、`E_THRESHOLD_NOT_FINITE`、
`E_DOMAIN_BELOW_THRESHOLD`、`E_INTENT_BELOW_THRESHOLD`、`E_SLOT_MISSING`、
`E_SLOT_BUDGET_PENDING`、`E_CONFIG_CHANGED`、`E_INJECTION_ILLEGAL_TOKEN`、
`E_INJECTION_MIGRATION_CONFLICT`。
