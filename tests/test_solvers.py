from __future__ import annotations

import unittest

from tactical_path.solvers.astar import AStarSolver
from tactical_path.solvers.chunkjump import ChunkJumpSolver, JumpDataBuilder
from tactical_path.solvers.flowfield import FlowFieldBuilder, flow_eligible

from .helpers import bundle


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.small = bundle("small")

    def test_astar_and_flow_match_on_every_intra_region_query(self) -> None:
        astar = AStarSolver(self.small.map_data)
        flow_builder = FlowFieldBuilder(self.small.map_data)
        checked = 0
        for query in self.small.queries:
            if not flow_eligible(self.small.map_data, query):
                continue
            astar_route = astar.solve(query)
            field = flow_builder.build(
                self.small.map_data.region_of_cell(query.start),
                query.target,
                query.mobility,
            )
            flow_route = field.solve(query, self.small.map_data.width)
            self.assertEqual(astar_route.cost, flow_route.cost, query.id)
            checked += 1
        self.assertEqual(checked, len(self.small.queries))

    def test_repeat_solves_are_byte_identical(self) -> None:
        astar = AStarSolver(self.small.map_data)
        flow_builder = FlowFieldBuilder(self.small.map_data)
        query = self.small.queries[0]
        first = astar.solve(query)
        second = astar.solve(query)
        self.assertEqual(first.path, second.path)
        self.assertEqual(first.cost, second.cost)
        field = flow_builder.build(
            self.small.map_data.region_of_cell(query.start),
            query.target,
            query.mobility,
        )
        flow_first = field.solve(query, self.small.map_data.width)
        flow_second = field.solve(query, self.small.map_data.width)
        self.assertEqual(
            flow_first.route_hash(self.small.map_data.width),
            flow_second.route_hash(self.small.map_data.width),
        )

    def test_chunk_jump_is_exact_on_small_and_medium_samples(self) -> None:
        for tier in ("small", "medium"):
            loaded = self.small if tier == "small" else bundle(tier)
            astar = AStarSolver(loaded.map_data)
            jump = ChunkJumpSolver(
                loaded.map_data, JumpDataBuilder(loaded.map_data)
            )
            for query in loaded.queries[:4]:
                expected = astar.solve(query)
                actual = jump.solve(query)
                self.assertEqual(expected.cost, actual.cost, f"{tier}:{query.id}")

    def test_chunk_jump_is_exact_on_1024_sample(self) -> None:
        loaded = bundle("large")
        query = loaded.queries[0]
        expected = AStarSolver(loaded.map_data).solve(query)
        actual = ChunkJumpSolver(
            loaded.map_data, JumpDataBuilder(loaded.map_data)
        ).solve(query)
        self.assertEqual(expected.cost, actual.cost)

