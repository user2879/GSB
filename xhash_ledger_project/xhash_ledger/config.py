"""配置与命令流的加载校验。

非法字段统一产出 ``E_CONFIG_INVALID`` 失败记录并尽量继续校验其余字段，
不抛出中断式异常。配置哈希覆盖全部配置字段；投影指纹刻意排除
``bucket_bits`` / ``bucket_capacity`` / ``similarity_permille`` 等只影响
分桶与聚类的字段，保证调整分桶位数时签名本身可被复用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import E_CONFIG_INVALID, Failure, make_failure
from .hashing import canonical_json, fingerprint, sha256_hex, stable_int

DECLARED_SEED_VERSIONS = ("seed-v1", "seed-v2", "seed-v3")
VALID_COMMANDS = ("add", "replace", "delete", "set_config")


@dataclass
class Config:
    signature_bits: int = 64
    bucket_bits: int = 8
    similarity_permille: int = 800
    seed_version: str = "seed-v1"
    bucket_capacity: int = 4
    frac_bits: int = 12
    tile_size: int = 8
    image_weight: int = 2
    text_weight: int = 3
    max_workers: int = 4
    global_budget: int = 1_000_000
    declared_seed_versions: tuple[str, ...] = field(default_factory=lambda: DECLARED_SEED_VERSIONS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature_bits": self.signature_bits,
            "bucket_bits": self.bucket_bits,
            "similarity_permille": self.similarity_permille,
            "seed_version": self.seed_version,
            "bucket_capacity": self.bucket_capacity,
            "frac_bits": self.frac_bits,
            "tile_size": self.tile_size,
            "image_weight": self.image_weight,
            "text_weight": self.text_weight,
            "max_workers": self.max_workers,
            "global_budget": self.global_budget,
            "declared_seed_versions": list(self.declared_seed_versions),
        }

    # ---- 指纹派生口径 -------------------------------------------------
    def config_hash(self) -> str:
        """全量配置哈希，用于快照恢复判定。"""
        return fingerprint(self.to_dict())

    def signature_fingerprint_parts(self) -> list[str]:
        """影响投影矩阵与签名位模式的配置子集（不含分桶/聚类参数）。"""
        sig_scope = {
            "signature_bits": self.signature_bits,
            "frac_bits": self.frac_bits,
            "tile_size": self.tile_size,
            "image_weight": self.image_weight,
            "text_weight": self.text_weight,
            "seed_version": self.seed_version,
        }
        return [self.seed_version, fingerprint(sig_scope)]

    def projection_seed(self, feature_dim: int) -> int:
        """投影矩阵确定性随机种子：种子版本 + 签名侧配置指纹 + 特征维度。"""
        parts = self.signature_fingerprint_parts() + [str(feature_dim)]
        return stable_int("projection|" + "|".join(parts))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_config(raw: dict[str, Any]) -> tuple[Config | None, list[Failure]]:
    """解析并校验配置，返回 (Config 或 None, 失败清单)。"""
    failures: list[Failure] = []
    base = Config()
    merged = base.to_dict()
    if not isinstance(raw, dict):
        failures.append(make_failure(E_CONFIG_INVALID, "config", "config", "root", "配置必须是 JSON 对象"))
        return None, failures
    for key, value in raw.items():
        if key not in merged:
            failures.append(
                make_failure(E_CONFIG_INVALID, "config", "config", key, f"未知配置字段 {key}")
            )
        else:
            merged[key] = value

    sb = merged["signature_bits"]
    if not (_is_int(sb) and sb > 0 and sb % 8 == 0):
        failures.append(
            make_failure(
                E_CONFIG_INVALID, "config", "config", "signature_bits",
                f"签名位宽必须是 8 的正整数倍，实际 {sb!r}",
            )
        )
    bb = merged["bucket_bits"]
    if not (_is_int(bb) and bb > 0 and _is_int(sb) and bb < sb):
        failures.append(
            make_failure(
                E_CONFIG_INVALID, "config", "config", "bucket_bits",
                f"分桶位数必须为正整数且小于签名位宽，实际 {bb!r} / {sb!r}",
            )
        )
    sp = merged["similarity_permille"]
    if not (_is_int(sp) and 0 <= sp <= 1000):
        failures.append(
            make_failure(
                E_CONFIG_INVALID, "config", "config", "similarity_permille",
                f"相似度千分比必须落在 [0,1000]，实际 {sp!r}",
            )
        )
    declared = tuple(merged["declared_seed_versions"])
    sv = merged["seed_version"]
    if not (isinstance(sv, str) and sv in declared):
        failures.append(
            make_failure(
                E_CONFIG_INVALID, "config", "config", "seed_version",
                f"投影种子版本 {sv!r} 不在已声明集合 {list(declared)} 内",
            )
        )
    bc = merged["bucket_capacity"]
    if not (_is_int(bc) and bc > 0):
        failures.append(
            make_failure(
                E_CONFIG_INVALID, "config", "config", "bucket_capacity",
                f"哈希桶容量必须为正整数，实际 {bc!r}",
            )
        )
    for name in ("frac_bits", "tile_size", "image_weight", "text_weight", "max_workers"):
        val = merged[name]
        if not (_is_int(val) and val > 0):
            failures.append(
                make_failure(E_CONFIG_INVALID, "config", "config", name, f"字段必须为正整数，实际 {val!r}")
            )
    gb = merged["global_budget"]
    if not (_is_int(gb) and gb > 0):
        failures.append(
            make_failure(
                E_CONFIG_INVALID, "config", "config", "global_budget",
                f"全局预算必须为正整数，实际 {gb!r}",
            )
        )
    if failures:
        return None, failures
    return Config(**{k: (tuple(v) if k == "declared_seed_versions" else v) for k, v in merged.items()}), []


def parse_command(raw: dict[str, Any], declared_seeds: tuple[str, ...]) -> list[Failure]:
    """校验单条编辑命令；合法时返回空失败清单。"""
    failures: list[Failure] = []
    if not isinstance(raw, dict):
        return [make_failure(E_CONFIG_INVALID, "command", "?", "root", "命令必须是 JSON 对象")]
    cmd = raw.get("cmd")
    sid = str(raw.get("sample_id", "?"))
    if cmd not in VALID_COMMANDS:
        failures.append(
            make_failure(E_CONFIG_INVALID, "command", sid, "cmd", f"未知命令类型 {cmd!r}")
        )
        return failures
    if cmd in ("add", "replace", "delete"):
        if not isinstance(raw.get("sample_id"), str) or not raw["sample_id"]:
            failures.append(
                make_failure(E_CONFIG_INVALID, "command", sid, "sample_id", "sample_id 必须是非空字符串")
            )
    if cmd in ("add", "replace") and "payload" in raw and not isinstance(raw.get("payload"), dict):
        failures.append(
            make_failure(E_CONFIG_INVALID, "command", sid, "payload", "payload 若提供必须是对象")
        )
    if cmd == "set_config":
        _, cfg_failures = parse_config(raw.get("config", {}))
        failures.extend(cfg_failures)
    return failures


def hash_command_stream(commands: list[dict[str, Any]]) -> str:
    return sha256_hex(canonical_json(commands))
