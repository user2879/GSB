"""Deterministic local certificate-family generation and loading."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .canonical import normalize_serial, write_canonical_json
from .errors import E_CHAIN, WorkbenchError
from .rsa import RSAPrivateKey, generate_key
from .x509 import CertRecord, build_certificate, der_to_pem, parse_certificate, pem_to_der


DAY_MS = 86_400_000
DEFAULT_NOT_BEFORE_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z
GENERATOR_VERSION = "1"


@dataclass(frozen=True)
class Family:
    root: Path
    data: dict[str, Any]

    @property
    def root_fingerprints(self) -> list[str]:
        return list(self.data["inventory"]["root_fingerprints"])

    @property
    def certificate_fingerprints(self) -> list[str]:
        return list(self.data["inventory"]["certificate_fingerprints"])

    @property
    def serial_numbers(self) -> list[str]:
        return list(self.data["inventory"]["serial_numbers"])

    def certificate(self, cert_id: str) -> dict[str, Any]:
        for cert in self.data["certificates"]:
            if cert["id"] == cert_id:
                return cert
        raise KeyError(cert_id)

    def leaf(self, leaf_id: str | None = None) -> dict[str, Any]:
        leaves = self.data["leaves"]
        if leaf_id is None:
            for leaf in leaves:
                if leaf["profile"] == "normal":
                    return leaf
            return leaves[0]
        for leaf in leaves:
            if leaf["id"] == leaf_id:
                return leaf
        raise WorkbenchError(
            E_CHAIN,
            f"unknown leaf id: {leaf_id}",
            field_path="--leaf",
        )

    def leaf_chain(self, leaf_id: str | None = None) -> list[str]:
        return list(self.leaf(leaf_id)["chain_fingerprints"])

    def chain_file(self, leaf_id: str | None = None) -> Path:
        leaf = self.leaf(leaf_id)
        return self.root / leaf["chain_file"]

    def key_file(self, leaf_id: str | None = None) -> Path:
        leaf = self.leaf(leaf_id)
        return self.root / leaf["key_file"]

    def cert_path(self, cert_id: str) -> Path:
        return self.root / self.certificate(cert_id)["certificate_file"]

    def load_chain(self, leaf_id: str | None = None) -> list[CertRecord]:
        records: list[CertRecord] = []
        for fingerprint in self.leaf_chain(leaf_id):
            cert = next(
                (item for item in self.data["certificates"] if item["fingerprint"] == fingerprint),
                None,
            )
            if cert is None:
                raise WorkbenchError(
                    E_CHAIN,
                    "family metadata references a missing certificate",
                    field_path="family.certificates",
                    details={"fingerprint": fingerprint},
                )
            records.append(parse_certificate(pem_to_der((self.root / cert["certificate_file"]).read_text(encoding="ascii"))))
        return records


def _serial(seed: str, cert_id: str) -> str:
    raw = hashlib.sha256(f"serial:{seed}:{cert_id}".encode("utf-8")).digest()[:16]
    value = int.from_bytes(raw, "big")
    if value == 0:
        value = 1
    text = format(value, "X")
    if len(text) % 2:
        text = "0" + text
    return normalize_serial(text)


def _subject_key_id(public_der: bytes) -> str:
    return hashlib.sha256(public_der).hexdigest()[:40]


def validate_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int):
        raise WorkbenchError(E_CHAIN, "port must be an integer", field_path="--port")
    if port != 0 and not (1024 <= port <= 65535):
        from .errors import E_PORT

        raise WorkbenchError(
            E_PORT,
            "loopback port must be 0 or in 1024..65535",
            field_path="--port",
        )
    return port


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii", newline="\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def generate_family(
    output_dir: str | Path,
    *,
    roots: int = 2,
    intermediate_levels: int = 2,
    leaves: int = 1,
    validity_days: int = 365,
    port: int = 0,
    seed: str = "tlstrust-deterministic-v1",
    key_bits: int = 2048,
    not_before_ms: int = DEFAULT_NOT_BEFORE_MS,
) -> Family:
    if roots < 1 or roots > 16:
        raise ValueError("roots must be in 1..16")
    if intermediate_levels < 0 or intermediate_levels > 8:
        raise ValueError("intermediate_levels must be in 0..8")
    if leaves < 1 or leaves > 64:
        raise ValueError("leaves must be in 1..64")
    if validity_days < 1:
        raise ValueError("validity_days must be positive")
    if key_bits < 512 or key_bits % 16:
        raise ValueError("key_bits must be a multiple of 16 and at least 512")
    validate_port(port)

    root_dir = Path(output_dir)
    cert_dir = root_dir / "certs"
    key_dir = root_dir / "keys"
    chain_dir = root_dir / "chains"
    for directory in (cert_dir, key_dir, chain_dir):
        directory.mkdir(parents=True, exist_ok=True)

    key_pool: dict[str, RSAPrivateKey] = {}

    def key_for(key_id: str) -> RSAPrivateKey:
        if key_id not in key_pool:
            key_pool[key_id] = generate_key(f"{seed}:{key_id}", key_bits)
        return key_pool[key_id]

    certificates: list[dict[str, Any]] = []
    cert_by_id: dict[str, dict[str, Any]] = {}
    cert_pem_by_fingerprint: dict[str, str] = {}
    roots_meta: list[dict[str, Any]] = []
    intermediates_meta: list[dict[str, Any]] = []
    leaves_meta: list[dict[str, Any]] = []

    def add_certificate(
        cert_id: str,
        role: str,
        subject_cn: str,
        issuer_cn: str,
        subject_key: RSAPrivateKey,
        issuer_key: RSAPrivateKey,
        not_before: int,
        not_after: int,
        is_ca: bool,
        path_len: int | None,
        parent_id: str | None,
        profile: str,
        corrupt_signature: bool = False,
    ) -> dict[str, Any]:
        serial = _serial(seed, cert_id)
        der = build_certificate(
            subject_cn=subject_cn,
            issuer_cn=issuer_cn,
            subject_key=subject_key,
            issuer_key=issuer_key,
            serial_number=serial,
            not_before_ms=not_before,
            not_after_ms=not_after,
            is_ca=is_ca,
            path_len=path_len,
            corrupt_signature=corrupt_signature,
        )
        record = parse_certificate(der)
        cert_pem = cert_dir / f"{cert_id}.pem"
        key_pem = key_dir / f"{cert_id}.key.pem"
        cert_pem_text = der_to_pem(der)
        _write_text(cert_pem, cert_pem_text)
        _write_text(key_pem, subject_key.private_pem())
        meta = {
            "id": cert_id,
            "role": role,
            "profile": profile,
            "subject": record.subject,
            "issuer": record.issuer,
            "serial_number": record.serial_number,
            "fingerprint": record.fingerprint,
            "not_before_ms": record.not_before_ms,
            "not_after_ms": record.not_after_ms,
            "ca": record.is_ca,
            "path_len": record.path_len,
            "key_usage": list(record.key_usage),
            "parent_id": parent_id,
            "certificate_file": str(cert_pem.relative_to(root_dir)).replace("\\", "/"),
            "key_file": str(key_pem.relative_to(root_dir)).replace("\\", "/"),
            "subject_key_id": _subject_key_id(subject_key.public_der()),
        }
        certificates.append(meta)
        cert_by_id[cert_id] = meta
        cert_pem_by_fingerprint[record.fingerprint] = cert_pem_text
        return meta

    def issuer_key_for(meta: dict[str, Any]) -> RSAPrivateKey:
        if meta["role"] == "root":
            root_number = int(meta["id"].rsplit("-", 1)[1])
            return key_for(f"root-key-{root_number:02d}")
        if meta["id"].startswith("int-overlong"):
            return overlong_key if "overlong_key" in locals() else key_for("int-key-overlong")
        level = int(meta["id"].rsplit("l", 1)[1])
        return key_for(f"int-key-l{level:02d}")

    root_valid_before = not_before_ms - 90 * DAY_MS
    ca_valid_after = not_before_ms + max(3650, validity_days * 4) * DAY_MS
    normal_leaf_after = not_before_ms + validity_days * DAY_MS
    expired_leaf_before = not_before_ms - (validity_days + 30) * DAY_MS
    expired_leaf_after = not_before_ms - DAY_MS

    root_meta: dict[str, dict[str, Any]] = {}
    for root_index in range(1, roots + 1):
        root_id = f"root-{root_index:02d}"
        root_key = key_for(f"root-key-{root_index:02d}")
        cn = f"TLSTRUST Root {root_index:02d}"
        meta = add_certificate(
            root_id,
            "root",
            cn,
            cn,
            root_key,
            root_key,
            root_valid_before,
            ca_valid_after,
            True,
            intermediate_levels,
            None,
            "trust_anchor",
        )
        root_meta[root_id] = meta
        roots_meta.append(meta)

    normal_chains: dict[str, list[str]] = {}
    for root_index in range(1, roots + 1):
        root_id = f"root-{root_index:02d}"
        parent = root_meta[root_id]
        for level in range(1, intermediate_levels + 1):
            int_id = f"int-{root_index:02d}-l{level:02d}"
            int_key = key_for(f"int-key-l{level:02d}")
            path_len = intermediate_levels - level
            meta = add_certificate(
                int_id,
                "intermediate",
                f"TLSTRUST Root {root_index:02d} Intermediate L{level:02d}",
                parent["subject"],
                int_key,
                issuer_key_for(parent),
                root_valid_before,
                ca_valid_after,
                True,
                path_len,
                parent["id"],
                "chain",
            )
            intermediates_meta.append(meta)
            parent = meta
        normal_chains[root_id] = []

    def chain_for_parent(parent_meta: dict[str, Any]) -> list[str]:
        chain = [parent_meta["fingerprint"]]
        current = parent_meta
        while current.get("parent_id"):
            current = cert_by_id[current["parent_id"]]
            chain.append(current["fingerprint"])
        return chain

    primary_root = "root-01"
    primary_parent = next(
        (item for item in intermediates_meta if item["id"].startswith("int-01-")),
        root_meta[primary_root],
    )
    if intermediate_levels:
        primary_parent = next(
            item for item in reversed(intermediates_meta) if item["id"].startswith("int-01-")
        )

    leaf_key = key_for("leaf-key")
    for leaf_index in range(1, leaves + 1):
        leaf_id = f"leaf-normal-{leaf_index:02d}"
        meta = add_certificate(
            leaf_id,
            "leaf",
            f"TLSTRUST Leaf Normal {leaf_index:02d}",
            primary_parent["subject"],
            leaf_key,
            issuer_key_for(primary_parent),
            not_before_ms,
            normal_leaf_after,
            False,
            None,
            primary_parent["id"],
            "normal",
        )
        chain = [meta["fingerprint"]] + chain_for_parent(primary_parent)
        chain_file = chain_dir / f"{leaf_id}.chain.pem"
        _write_chain(chain_file, cert_pem_by_fingerprint, chain)
        meta["chain_fingerprints"] = chain
        meta["chain_file"] = str(chain_file.relative_to(root_dir)).replace("\\", "/")
        meta["key_file"] = str((key_dir / f"{leaf_id}.key.pem").relative_to(root_dir)).replace("\\", "/")
        leaves_meta.append(meta)

    expired_id = "leaf-expired-01"
    expired_meta = add_certificate(
        expired_id,
        "leaf",
        "TLSTRUST Leaf Expired 01",
        primary_parent["subject"],
        leaf_key,
        issuer_key_for(primary_parent),
        expired_leaf_before,
        expired_leaf_after,
        False,
        None,
        primary_parent["id"],
        "expired",
    )
    chain = [expired_meta["fingerprint"]] + chain_for_parent(primary_parent)
    chain_file = chain_dir / f"{expired_id}.chain.pem"
    _write_chain(chain_file, cert_pem_by_fingerprint, chain)
    expired_meta["chain_fingerprints"] = chain
    expired_meta["chain_file"] = str(chain_file.relative_to(root_dir)).replace("\\", "/")
    expired_meta["key_file"] = str((key_dir / f"{expired_id}.key.pem").relative_to(root_dir)).replace("\\", "/")
    leaves_meta.append(expired_meta)

    badsig_id = "leaf-badsig-01"
    badsig_meta = add_certificate(
        badsig_id,
        "leaf",
        "TLSTRUST Leaf Bad Signature 01",
        primary_parent["subject"],
        leaf_key,
        issuer_key_for(primary_parent),
        not_before_ms,
        normal_leaf_after,
        False,
        None,
        primary_parent["id"],
        "bad_signature",
        corrupt_signature=True,
    )
    chain = [badsig_meta["fingerprint"]] + chain_for_parent(primary_parent)
    chain_file = chain_dir / f"{badsig_id}.chain.pem"
    _write_chain(chain_file, cert_pem_by_fingerprint, chain)
    badsig_meta["chain_fingerprints"] = chain
    badsig_meta["chain_file"] = str(chain_file.relative_to(root_dir)).replace("\\", "/")
    badsig_meta["key_file"] = str((key_dir / f"{badsig_id}.key.pem").relative_to(root_dir)).replace("\\", "/")
    leaves_meta.append(badsig_meta)

    overlong_parent = primary_parent
    overlong_key = key_for("int-key-overlong")
    extra_level = intermediate_levels + 1
    overlong_int_id = f"int-overlong-l{extra_level:02d}"
    overlong_meta = add_certificate(
        overlong_int_id,
        "intermediate",
        f"TLSTRUST Overlong Intermediate L{extra_level:02d}",
        overlong_parent["subject"],
        overlong_key,
        issuer_key_for(overlong_parent),
        root_valid_before,
        ca_valid_after,
        True,
        0,
        overlong_parent["id"],
        "overlong_chain",
    )
    intermediates_meta.append(overlong_meta)
    overlong_leaf_id = "leaf-overlong-01"
    overlong_leaf = add_certificate(
        overlong_leaf_id,
        "leaf",
        "TLSTRUST Leaf Overlong 01",
        overlong_meta["subject"],
        leaf_key,
        overlong_key,
        not_before_ms,
        normal_leaf_after,
        False,
        None,
        overlong_meta["id"],
        "overlong",
    )
    chain = [overlong_leaf["fingerprint"]] + chain_for_parent(overlong_meta)
    chain_file = chain_dir / f"{overlong_leaf_id}.chain.pem"
    _write_chain(chain_file, cert_pem_by_fingerprint, chain)
    overlong_leaf["chain_fingerprints"] = chain
    overlong_leaf["chain_file"] = str(chain_file.relative_to(root_dir)).replace("\\", "/")
    overlong_leaf["key_file"] = str((key_dir / f"{overlong_leaf_id}.key.pem").relative_to(root_dir)).replace("\\", "/")
    leaves_meta.append(overlong_leaf)

    for meta in leaves_meta:
        source = cert_by_id[meta["id"]]
        meta.update(
            {
                "subject": source["subject"],
                "issuer": source["issuer"],
                "serial_number": source["serial_number"],
                "fingerprint": source["fingerprint"],
                "not_before_ms": source["not_before_ms"],
                "not_after_ms": source["not_after_ms"],
                "ca": source["ca"],
                "path_len": source["path_len"],
                "key_usage": source["key_usage"],
                "parent_id": source["parent_id"],
                "certificate_file": source["certificate_file"],
            }
        )

    all_serial_numbers = sorted({item["serial_number"] for item in certificates})
    all_fingerprints = sorted({item["fingerprint"] for item in certificates})
    data = {
        "format_version": 1,
        "kind": "tlstrust.certificate_family",
        "generator_version": GENERATOR_VERSION,
        "workbench_version": __version__,
        "params": {
            "roots": roots,
            "intermediate_levels": intermediate_levels,
            "leaves": leaves,
            "validity_days": validity_days,
            "port": port,
            "seed": seed,
            "key_bits": key_bits,
            "not_before_ms": not_before_ms,
        },
        "roots": roots_meta,
        "intermediates": intermediates_meta,
        "leaves": leaves_meta,
        "certificates": certificates,
        "inventory": {
            "root_fingerprints": sorted(item["fingerprint"] for item in roots_meta),
            "certificate_fingerprints": all_fingerprints,
            "serial_numbers": all_serial_numbers,
        },
        "recommended_evaluation_ms": not_before_ms + DAY_MS,
    }
    write_canonical_json(root_dir / "family.json", data)
    return load_family(root_dir)


def _write_chain(path: Path, cert_pem_by_fingerprint: dict[str, str], chain: list[str]) -> None:
    pem_blocks: list[str] = []
    for fingerprint in chain:
        pem_blocks.append(cert_pem_by_fingerprint[fingerprint])
    _write_text(path, "".join(pem_blocks))


def load_family(path: str | Path) -> Family:
    root = Path(path)
    if root.is_dir():
        root = root / "family.json"
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkbenchError(E_CHAIN, f"cannot load certificate family: {exc}", field_path="family") from exc
    if data.get("format_version") != 1 or data.get("kind") != "tlstrust.certificate_family":
        raise WorkbenchError(E_CHAIN, "unsupported certificate family format", field_path="family.format_version")
    base = root.parent
    for cert in data.get("certificates", []):
        try:
            der = pem_to_der((base / cert["certificate_file"]).read_text(encoding="ascii"))
            record = parse_certificate(der)
        except Exception as exc:
            raise WorkbenchError(
                E_CHAIN,
                f"certificate family member cannot be parsed: {cert.get('id')}",
                field_path=f"family.certificates[{cert.get('id')}]",
            ) from exc
        if record.fingerprint != cert["fingerprint"]:
            raise WorkbenchError(
                E_CHAIN,
                "certificate fingerprint mismatch while loading family",
                field_path=f"family.certificates[{cert['id']}].fingerprint",
                details={"expected": cert["fingerprint"], "actual": record.fingerprint},
            )
    return Family(root=base, data=data)


def family_digest(family: Family) -> str:
    payload = {
        "params": family.data["params"],
        "roots": family.data["roots"],
        "leaves": family.data["leaves"],
        "inventory": family.data["inventory"],
    }
    from .canonical import digest_obj

    return digest_obj(payload)
