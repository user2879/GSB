from __future__ import annotations

from .canonical import sha256_hex


def sample_unit_hash(rollback_id: str, volume_id: str, block_index: int) -> int:
    material = f"blockledger-sample-v1|{rollback_id}|{volume_id}|{block_index}"
    return int(sha256_hex(material)[:16], 16)


def is_selected(
    rollback_id: str,
    volume_id: str,
    block_index: int,
    sample_permille: int,
) -> bool:
    if sample_permille <= 0:
        return False
    if sample_permille >= 1000:
        return True
    return sample_unit_hash(rollback_id, volume_id, block_index) % 1000 < sample_permille
