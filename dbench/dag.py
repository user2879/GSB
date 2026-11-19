"""Pipeline node model and deterministic dependency graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config

ENCODE_STAGE = "encode_prompt"
INIT_STAGE = "init_latent"
IMAGE_ENCODE_STAGE = "image_encode"
DECODE_STAGE = "decode"


def denoise_stage(step: int) -> str:
    return f"denoise_s{step:03d}"


def denoise_step_of(stage: str) -> int | None:
    if stage.startswith("denoise_s"):
        return int(stage[len("denoise_s"):])
    return None


@dataclass(frozen=True)
class Node:
    stage: str
    step: int | None
    sort_key: tuple[int, int]
    deps: tuple[str, ...]
    has_array: bool = True
    extra: dict = field(default_factory=dict)


def build_graph(config: Config) -> dict[str, Node]:
    """Return stage -> Node for both t2i and img2img branches.

    Denoise steps 1..K (K = image_replacement_steps) consume the image latent;
    steps K+1..N consume the previous denoise output. The initial latent stage
    stays in the graph as an audited upstream product even when fully replaced.
    """
    nodes: dict[str, Node] = {}
    nodes[ENCODE_STAGE] = Node(
        ENCODE_STAGE, None, (0, 0), tuple(), has_array=True,
        extra={"kind": "encode"},
    )
    nodes[INIT_STAGE] = Node(
        INIT_STAGE, None, (1, 0), (ENCODE_STAGE,), has_array=True,
        extra={"kind": "init"},
    )

    replace = config.image_replacement_steps if config.initial_image else 0
    if config.initial_image:
        nodes[IMAGE_ENCODE_STAGE] = Node(
            IMAGE_ENCODE_STAGE, None, (2, 0), tuple(), has_array=True,
            extra={"kind": "image_encode", "replacement_steps": replace},
        )

    for step in range(1, config.steps + 1):
        stage = denoise_stage(step)
        if step <= replace:
            # img2img early steps read the encoded initial-image latent.
            deps = (ENCODE_STAGE, IMAGE_ENCODE_STAGE)
        elif step == replace + 1:
            base = IMAGE_ENCODE_STAGE if replace else INIT_STAGE
            deps = (ENCODE_STAGE, base)
        else:
            deps = (ENCODE_STAGE, denoise_stage(step - 1))
        nodes[stage] = Node(
            stage, step, (3, step), deps, has_array=True,
            extra={"kind": "denoise", "replaced": step <= replace},
        )

    final_denoise = denoise_stage(config.steps)
    nodes[DECODE_STAGE] = Node(
        DECODE_STAGE, None, (4, 0), (ENCODE_STAGE, final_denoise),
        has_array=True, extra={"kind": "decode"},
    )
    return nodes


def topological_order(nodes: dict[str, Node]) -> list[str]:
    """Stable topological order using the precomputed sort keys as tiebreaker."""
    visited: set[str] = set()
    order: list[str] = []

    def visit(stage: str) -> None:
        if stage in visited:
            return
        for dep in sorted(nodes[stage].deps, key=lambda s: nodes[s].sort_key):
            visit(dep)
        visited.add(stage)
        order.append(stage)

    for stage in sorted(nodes, key=lambda s: nodes[s].sort_key):
        visit(stage)
    return order


def downstream_of(stage: str, nodes: dict[str, Node]) -> set[str]:
    """All transitive downstream stages of ``stage``."""
    result: set[str] = set()
    frontier = [stage]
    while frontier:
        current = frontier.pop()
        for name, node in nodes.items():
            if current in node.deps and name not in result:
                result.add(name)
                frontier.append(name)
    return result
