"""Fault injection configuration.

Injections only alter *input files or parameters*; the package source code is
identical for normal and failing runs.

* ``capacity``  -> override hot tier capacity to an impossibly small integer
  (parameter change), surfacing as ``E_CAPACITY``;
* ``truncate``  -> truncate one staged object file on disk (input file change),
  surfacing as ``E_INTEGRITY``;
* ``version``   -> pass a mismatched session policy/tier version (parameter
  change), surfacing as ``E_VERSION``;
* ``kill``      -> force the process to exit after the physical switch of one
  object but before its ledger commit, exercising crash recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Injection:
    kind: Optional[str] = None
    target_key: Optional[str] = None
    capacity_override: Optional[int] = None
    policy_version_override: Optional[str] = None
    tier_param_version_override: Optional[str] = None
    kill_after_commit: int = 1
    kill_exit_code: int = 9

    @property
    def active(self) -> bool:
        return self.kind is not None


def parse_injection(
    kind: Optional[str],
    target_key: Optional[str] = None,
    kill_after_commit: int = 1,
) -> Injection:
    allowed = {None, "capacity", "truncate", "version", "kill"}
    if kind not in allowed:
        raise ValueError(f"unknown injection kind {kind!r}; expected one of {sorted(a for a in allowed if a)}")
    return Injection(kind=kind, target_key=target_key, kill_after_commit=kill_after_commit)
