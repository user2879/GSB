"""确定性合成表格数据生成与预处理。

流程固定且仅依赖 NumPy RandomState：
1. 按列描述生成数值/类别基础列与目标列；
2. 由基础列派生冗余列；
3. 以独立固定随机流注入缺失值；
4. 固定种子洗牌后按比例切训练/验证折；
5. 数值列用训练折中位数填充并标准化，类别列用训练折众数填充并独热编码；
6. 对原始数据与预处理参数计算数据指纹。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import DataSpec, ExperimentConfig
from .util import derive_seed, fingerprint, stable_json

STAGE_NAME = "data"


@dataclass
class PreparedData:
    spec_hash: str
    data_fingerprint: str
    n_train: int
    n_val: int
    feature_names: list[str]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    raw_table: dict[str, list[Any]]
    imputation: dict[str, Any]

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "spec_hash": self.spec_hash,
            "data_fingerprint": self.data_fingerprint,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "feature_names": self.feature_names,
            "imputation": self.imputation,
        }


def _generate_raw_table(spec: DataSpec) -> dict[str, list[Any]]:
    rng = np.random.RandomState(spec.base_seed)
    table: dict[str, list[Any]] = {}

    for col in spec.numeric:
        values = rng.normal(loc=col.mean, scale=max(col.std, 1e-9), size=spec.n_samples)
        table[col.name] = [float(v) for v in values]

    for col in spec.categorical:
        codes = rng.randint(low=0, high=col.cardinality, size=spec.n_samples)
        table[col.name] = [f"c{value}" for value in codes]

    # 目标列 = 数值列固定线性组合 + 类别列固定效应 + 噪声。
    target = np.zeros(spec.n_samples, dtype=np.float64)
    for index, col in enumerate(spec.numeric):
        column = np.asarray(table[col.name], dtype=np.float64)
        target += ((index % 5) + 1) * 0.35 * column
    for col in spec.categorical:
        codes = np.array(
            [int(token[1:]) for token in table[col.name]], dtype=np.int64
        )
        target += (codes.astype(np.float64) - (col.cardinality - 1) / 2.0) * 0.4
    target += rng.normal(0.0, max(spec.target.noise_std, 0.0), size=spec.n_samples)
    table[spec.target.name] = [float(v) for v in target]

    # 冗余列在目标列之后生成，不参与目标构造，保证泄漏可控且确定。
    for red in spec.redundant:
        source = np.asarray(table[red.source], dtype=np.float64)
        noise = rng.normal(0.0, max(red.noise, 0.0), size=spec.n_samples)
        values = red.coefficient * source + noise
        table[red.name] = [float(v) for v in values]

    return table


def _inject_missing(
    table: dict[str, list[Any]], spec: DataSpec
) -> dict[str, list[Any]]:
    if spec.missing_rate <= 0:
        return table
    rng = np.random.RandomState(derive_seed("missing", spec.base_seed))
    missing_columns = [col.name for col in spec.numeric] + [
        col.name for col in spec.categorical
    ]
    total_cells = spec.n_samples * len(missing_columns)
    mask = rng.random_sample(total_cells) < spec.missing_rate
    flat_index = 0
    for col_name in missing_columns:
        for row in range(spec.n_samples):
            if mask[flat_index]:
                table[col_name][row] = None
            flat_index += 1
    return table


def _split_indices(spec: DataSpec) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(derive_seed("split", spec.base_seed))
    order = np.arange(spec.n_samples)
    rng.shuffle(order)
    n_train = int(round(spec.n_samples * spec.split_ratio))
    n_train = max(1, min(spec.n_samples - 1, n_train))
    return order[:n_train], order[n_train:]


def _numeric_matrix(
    table: dict[str, list[Any]],
    names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    imputation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    train_cols: list[np.ndarray] = []
    val_cols: list[np.ndarray] = []
    for name in names:
        raw = table[name]
        train_values = np.array(
            [raw[i] for i in train_idx], dtype=np.float64
        )
        valid = ~np.isnan(train_values)
        if not np.any(valid):
            median = 0.0
        else:
            median = float(np.median(train_values[valid]))
        filled_train = np.where(valid, train_values, median)
        centered = filled_train - median
        scale = float(np.std(centered))
        if scale < 1e-12:
            scale = 1.0
        train_cols.append(centered / scale)

        val_values = np.array([raw[i] for i in val_idx], dtype=np.float64)
        val_valid = ~np.isnan(val_values)
        filled_val = np.where(val_valid, val_values, median)
        val_cols.append((filled_val - median) / scale)
        imputation[name] = {"strategy": "median", "fill": median, "scale": scale}

    if not train_cols:
        return (
            np.zeros((len(train_idx), 0)),
            np.zeros((len(val_idx), 0)),
        )
    return np.column_stack(train_cols), np.column_stack(val_cols)


def _categorical_matrix(
    table: dict[str, list[Any]],
    spec: DataSpec,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    imputation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_blocks: list[np.ndarray] = []
    val_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    for col in spec.categorical:
        raw = table[col.name]
        train_tokens = [raw[i] for i in train_idx]
        observed = [token for token in train_tokens if token is not None]
        if observed:
            values, counts = np.unique(np.asarray(observed), return_counts=True)
            mode = str(values[int(np.argmax(counts))])
            categories = [str(v) for v in values]
        else:
            mode = "c0"
            categories = ["c0"]
        filled_train = [token if token is not None else mode for token in train_tokens]
        filled_val = [
            raw[i] if raw[i] is not None else mode for i in val_idx
        ]
        index_map = {token: k for k, token in enumerate(categories)}
        train_codes = np.array(
            [index_map.get(token, 0) for token in filled_train], dtype=np.int64
        )
        val_codes = np.array(
            [index_map.get(token, 0) for token in filled_val], dtype=np.int64
        )
        onehot_train = np.zeros((len(train_idx), len(categories)), dtype=np.float64)
        onehot_val = np.zeros((len(val_idx), len(categories)), dtype=np.float64)
        onehot_train[np.arange(len(train_idx)), train_codes] = 1.0
        onehot_val[np.arange(len(val_idx)), val_codes] = 1.0
        train_blocks.append(onehot_train)
        val_blocks.append(onehot_val)
        feature_names.extend(f"{col.name}={token}" for token in categories)
        imputation[col.name] = {
            "strategy": "mode",
            "fill": mode,
            "categories": categories,
        }
    if not train_blocks:
        return (
            np.zeros((len(train_idx), 0)),
            np.zeros((len(val_idx), 0)),
            feature_names,
        )
    return np.column_stack(train_blocks), np.column_stack(val_blocks), feature_names


def prepare_data(config: ExperimentConfig) -> PreparedData:
    spec = config.data
    spec_hash = fingerprint(
        {"data": asdict(spec), "split_ratio": spec.split_ratio}
    )
    table = _generate_raw_table(spec)
    table = _inject_missing(table, spec)
    train_idx, val_idx = _split_indices(spec)

    numeric_names = [col.name for col in spec.numeric] + [
        col.name for col in spec.redundant
    ]
    imputation: dict[str, Any] = {}
    x_num_train, x_num_val = _numeric_matrix(
        table, numeric_names, train_idx, val_idx, imputation
    )
    x_cat_train, x_cat_val, cat_features = _categorical_matrix(
        table, spec, train_idx, val_idx, imputation
    )

    blocks_train = [block for block in (x_num_train, x_cat_train) if block.size]
    blocks_val = [block for block in (x_num_val, x_cat_val) if block.size]
    x_train = np.column_stack(blocks_train) if blocks_train else np.zeros(
        (len(train_idx), 1)
    )
    x_val = np.column_stack(blocks_val) if blocks_val else np.zeros((len(val_idx), 1))
    # 首列恒为 1 的截距项；截距不施加正则。
    x_train = np.column_stack([np.ones(len(train_idx)), x_train])
    x_val = np.column_stack([np.ones(len(val_idx)), x_val])

    y_train = np.array(
        [table[spec.target.name][i] for i in train_idx], dtype=np.float64
    )
    y_val = np.array(
        [table[spec.target.name][i] for i in val_idx], dtype=np.float64
    )
    feature_names = ["bias"] + numeric_names + cat_features

    data_fp = fingerprint(
        {
            "spec_hash": spec_hash,
            "feature_names": feature_names,
            "imputation": imputation,
            "train_index": train_idx.tolist(),
            "val_index": val_idx.tolist(),
            "y_train": np.round(y_train, 10).tolist(),
            "y_val": np.round(y_val, 10).tolist(),
            "x_train": np.round(x_train, 10).tolist(),
            "x_val": np.round(x_val, 10).tolist(),
        }
    )
    return PreparedData(
        spec_hash=spec_hash,
        data_fingerprint=data_fp,
        n_train=len(train_idx),
        n_val=len(val_idx),
        feature_names=feature_names,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        raw_table=table,
        imputation=imputation,
    )


def stable_data_summary(data: PreparedData) -> str:
    """供报告/测试比较的稳定摘要（不含原始全量矩阵）。"""

    return stable_json(
        {
            "data_fingerprint": data.data_fingerprint,
            "n_train": data.n_train,
            "n_val": data.n_val,
            "feature_names": data.feature_names,
            "imputation": data.imputation,
        }
    )
