# 交换格式与精度规范

## 1. 偏好对原文

`preference_pairs.jsonl` 为 UTF-8 JSONL，每行一个对象：

```text
pair_id      string, 必填, 唯一, 匹配 [A-Za-z0-9][A-Za-z0-9_.:-]{0,63}
prompt       string, 必填, 非空
chosen       string, 必填, 非空
rejected     string, 必填, 非空
annotator_id string, 必填, 非空
timestamp    string, 必填, 非空
```

非法 JSON、字段缺失/类型错误、空字段、非法 ID、重复 ID 都产生 `E_FORMAT_INVALID`，失败项包含来源、行号、字段和可用时的 pair ID。

## 2. v1 定宽格式

单行总宽 48 字符：

```text
偏移 [0, 24)   pair_id             24 字符，左对齐，空格补足
偏移 [24, 36)  score_milli         12 字符，右对齐，十进制整数
偏移 [36, 48)  delta_milli         12 字符，右对齐，十进制整数
```

编码函数：`pref_reward_lab.formats.encode_v1_record`。解析器要求去除换行后宽度严格为 48；为容纳负号，整数范围限制为 `[-99_999_999_999, 99_999_999_999]`。

## 3. v2 JSON 格式

文件可以是：

- JSON 对象数组；
- 含 `records` 数组的对象；
- 每行一个对象的 JSONL。

记录规范字段：

```text
pair_id          string
reward_score     finite number
reward_delta     finite number, 缺省 0.0
reward_label     string, 必须在 config.reward_labels
precision_label  string, 必须在 config.precision_labels
```

兼容别名：`id`、`score`、`delta`、`label`、`precision`。

## 4. 统一记录

```text
pair_id
feature_buckets          {bucket_string: count}
feature_bucket_count
chosen_score
rejected_score
reward_score             model chosen_score
reward_delta             model chosen_score - rejected_score
model_reward_score
model_reward_delta
source_reward_score      selected exchange value
source_reward_delta      selected exchange value
effective_threshold
boundary
reward_label
precision_label
source_version
```

边界判定：`reward_delta <= effective_threshold`，恰好相等时命中。

## 5. 转换规则

### v1 -> v2

```text
reward_score = score_milli / 1000.0
reward_delta = delta_milli / 1000.0
precision_label = milli_exact
reward_label = v1_default_label
```

若当前存在同 pair、同 v1 整数、同 v1 内容指纹的旁路条目，则恢复旁路中的原始浮点、命名标签和精度标签，并计入 `restored_from_sidecar`。

### v2 -> v1

使用 `Decimal(str(float_value)) * 1000` 后执行 `ROUND_HALF_UP`。例如：

```text
 1.2345 ->  1235
-1.2345 -> -1235
 1.2344 ->  1234
```

旁路条目字段：

```text
pair_id
reward_score                    原始浮点（完整 Python repr）
reward_delta
milli_reward_score              还原整数
milli_reward_delta
score_residual_milli            原始毫分值 - 还原整数，十进制字符串
delta_residual_milli
reward_label                    被丢弃的 v2 命名标签
precision_label                 被丢弃的 v2 精度标签
source_version                  v2
target_version                  v1
v1_content_fingerprint          还原后 v1 内容指纹
v2_content_fingerprint          原 v2 内容指纹
```

旁路文件为确定性排序的 JSON，可用于 v1 -> v2 恢复链。
