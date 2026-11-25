from __future__ import annotations

from evidence_diag.models import AppConfig, Session
from evidence_diag.sample import build_sample


def sample_objects() -> tuple[AppConfig, Session, list[dict]]:
    config_raw, session_raw, commands = build_sample()
    return (
        AppConfig.from_dict(config_raw),
        Session.from_dict(session_raw),
        commands,
    )
