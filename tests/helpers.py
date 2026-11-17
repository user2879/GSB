import json
from pathlib import Path

from graphidx import IndexDef, MaintenanceSession, PropertyGraph, ViewDef

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def load_sample_graph() -> PropertyGraph:
    return PropertyGraph.from_dict(json.loads((SAMPLES / "graph.json").read_text(encoding="utf-8")))


def sample_index_defs():
    return [IndexDef("by_label", "label_inverted", 1), IndexDef("by_weight", "weight_adjacency", 1)]


def sample_view_defs():
    return [
        ViewDef("reach_from_1", [1], []),
        ViewDef("reach_team", [4], ["reach_from_1"]),
    ]


def sample_changesets():
    return json.loads((SAMPLES / "changesets.json").read_text(encoding="utf-8"))["changesets"]


def make_session(**kwargs) -> MaintenanceSession:
    kwargs.setdefault("concurrency", 2)
    kwargs.setdefault("budget", 100)
    session = MaintenanceSession(load_sample_graph(), sample_index_defs(), sample_view_defs(), **kwargs)
    session.build_indexes()
    return session
