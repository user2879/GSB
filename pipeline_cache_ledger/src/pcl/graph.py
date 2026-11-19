"""Reference dependency graph, cycle detection and deterministic propagation."""

from __future__ import annotations

from .errors import CycleError, ReferenceError


class RefGraph:
    """Edges point upstream -> downstream (producer -> consumer)."""

    def __init__(self):
        self._down: dict[str, set[str]] = {}
        self._up: dict[str, set[str]] = {}
        self._graph_version = 0

    @property
    def graph_version(self):
        return self._graph_version

    def add_node(self, key):
        self._down.setdefault(key, set())
        self._up.setdefault(key, set())

    def add_edge(self, upstream, downstream):
        self.add_node(upstream)
        self.add_node(downstream)
        if downstream not in self._down[upstream]:
            self._down[upstream].add(downstream)
            self._up[downstream].add(upstream)
            self._graph_version += 1

    def nodes(self):
        return sorted(self._down)

    def downstream_of(self, key):
        return sorted(self._down.get(key, ()))

    def upstream_of(self, key):
        return sorted(self._up.get(key, ()))

    def detect_cycle(self):
        """Return one deterministic cycle (list of keys) or None."""
        white, gray, black = 0, 1, 2
        color = {n: white for n in self.nodes()}
        stack: list[str] = []

        def visit(node):
            color[node] = gray
            stack.append(node)
            for nxt in sorted(self._down[node]):
                if color[nxt] == gray:
                    start = stack.index(nxt)
                    return stack[start:] + [nxt]
                if color[nxt] == white:
                    found = visit(nxt)
                    if found:
                        return found
            stack.pop()
            color[node] = black
            return None

        for node in self.nodes():
            if color[node] == white:
                found = visit(node)
                if found:
                    return found
        return None

    def require_acyclic(self):
        cycle = self.detect_cycle()
        if cycle:
            raise CycleError("reference cycle detected", cycle)

    def transitive_downstream(self, changed_keys):
        """All direct + indirect downstream keys, excluding the roots."""
        seen: set[str] = set()
        frontier = list(changed_keys)
        while frontier:
            node = frontier.pop()
            for nxt in self._down.get(node, ()):
                if nxt not in seen and nxt not in changed_keys:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

    def invalidation_order(self, changed_keys):
        """Stable topological order: roots first, then consumers level by level.

        Kahn's algorithm with lexicographic ordering makes the result
        independent of manifest declaration / traversal order.
        """
        self.require_acyclic()
        roots = set(changed_keys)
        affected = set(roots) | self.transitive_downstream(roots)
        indeg = {
            n: sum(1 for up in self._up[n] if up in affected) for n in affected
        }
        ready = sorted(n for n in affected if indeg[n] == 0)
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for nxt in sorted(self._down.get(node, ())):
                if nxt in affected:
                    indeg[nxt] -= 1
                    if indeg[nxt] == 0:
                        ready.append(nxt)
                        ready.sort()
        if len(order) != len(affected):  # pragma: no cover - guarded above
            raise CycleError("reference cycle detected", self.detect_cycle())
        return order

    def referrers_of(self, key):
        return sorted(self._up.get(key, ()))

    def require_removable(self, key):
        referrers = self.referrers_of(key)
        if referrers:
            raise ReferenceError(
                "artifact is still referenced", key, referrers
            )

    def remove_node(self, key):
        self.require_removable(key)
        for up in list(self._up.get(key, ())):
            self._down[up].discard(key)
        for down in list(self._down.get(key, ())):
            self._up[down].discard(key)
        self._down.pop(key, None)
        self._up.pop(key, None)
        self._graph_version += 1

    def snapshot_edges(self):
        return tuple(
            sorted((up, down) for up, downs in self._down.items() for down in downs)
        )
