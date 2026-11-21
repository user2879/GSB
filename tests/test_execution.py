from __future__ import annotations

import unittest
from dataclasses import replace
from decimal import Decimal

from mlpipe_workbench.compute import (
    apply_stage,
    build_shards,
    saturate_int64,
    table_digest,
    table_from_dataset,
)
from mlpipe_workbench.demo_data import amplify_outliers
from mlpipe_workbench.errors import E_OUTLIER, E_PARAM, WorkbenchError
from mlpipe_workbench.evolution import classify_current_shards
from mlpipe_workbench.loader import prepare_from_objects
from mlpipe_workbench.models import INT64_MAX, INT64_MIN, ShardingSpec, Table, TableColumn

from tests.helpers import base_objects, prepared_base


class ExecutionTests(unittest.TestCase):
    def test_batch_boundaries_do_not_change_stage_or_final_digests(self) -> None:
        prepared = prepared_base()
        signatures: dict[int, list[str]] = {}
        for batch_size in (1, 7, 32, 137):
            table = table_from_dataset(prepared.dataset)
            digests = []
            for stage in prepared.pipeline.stages:
                table, computation = apply_stage(
                    table,
                    stage,
                    prepared.pipeline.fixed_scale,
                    prepared.pipeline.sharding,
                    batch_size,
                )
                digests.append(computation.output_digest)
            signatures[batch_size] = digests
        baseline = signatures[1]
        for batch_size in (7, 32, 137):
            self.assertEqual(signatures[batch_size], baseline)

    def test_fixed_point_truncation_and_saturation(self) -> None:
        prepared = prepared_base()
        stage = prepared.pipeline.stages[0]
        table = Table(
            row_count=2,
            columns=[
                TableColumn(
                    key="synthetic.value",
                    type="float",
                    values=[Decimal("-1.2349"), Decimal("123.9999")],
                    nulls=[False, False],
                )
            ],
        )
        output, _ = apply_stage(
            table,
            stage,
            prepared.pipeline.fixed_scale,
            prepared.pipeline.sharding,
            1,
        )
        self.assertEqual(output.columns[0].values, [-1234, 123999])
        self.assertEqual(saturate_int64(INT64_MAX + 1), INT64_MAX)
        self.assertEqual(saturate_int64(INT64_MIN - 1), INT64_MIN)

        overflow = Table(
            row_count=1,
            columns=[
                TableColumn(
                    key="synthetic.big",
                    type="float",
                    values=[Decimal("1e17")],
                    nulls=[False],
                )
            ],
        )
        with self.assertRaises(WorkbenchError) as captured:
            apply_stage(
                overflow,
                replace(stage, params={"saturate": 0}),
                prepared.pipeline.fixed_scale,
                prepared.pipeline.sharding,
                1,
            )
        self.assertEqual(captured.exception.code, E_PARAM)
        self.assertEqual(captured.exception.param, "saturate")

    def test_session_normalize_scale_is_independent_and_replayable_integer_context(self) -> None:
        prepared = prepared_base()
        outputs = {}
        for scale in (2, 4):
            table = table_from_dataset(prepared.dataset)
            for stage in prepared.pipeline.stages[:3]:
                table, computation = apply_stage(
                    table,
                    stage,
                    prepared.pipeline.fixed_scale,
                    prepared.pipeline.sharding,
                    17,
                    normalize_scale=scale,
                )
            outputs[scale] = computation.output_digest
            self.assertTrue(all(isinstance(value, int) for value in table.columns[1].values))
        self.assertNotEqual(outputs[2], outputs[4])

    def test_outlier_gate_has_dedicated_business_code(self) -> None:
        data, pipeline = base_objects()
        data = amplify_outliers(data, 1, 300)
        prepared = prepare_from_objects(data, pipeline)
        table = table_from_dataset(prepared.dataset)
        for stage in prepared.pipeline.stages[:3]:
            table, _ = apply_stage(
                table,
                stage,
                prepared.pipeline.fixed_scale,
                prepared.pipeline.sharding,
                17,
            )
        with self.assertRaises(WorkbenchError) as captured:
            apply_stage(
                table,
                prepared.pipeline.stages[3],
                prepared.pipeline.fixed_scale,
                prepared.pipeline.sharding,
                17,
            )
        error = captured.exception
        self.assertEqual(error.code, E_OUTLIER)
        self.assertEqual(error.stage_id, "clip_amount")
        self.assertEqual(error.param, "max_outlier_permille")
        self.assertGreater(error.affected_rows or 0, 0)

    def test_remainder_shards_keep_or_drop_explicitly(self) -> None:
        rows = 65
        table = Table(
            row_count=rows,
            columns=[TableColumn("source.value", "fixed", list(range(rows)), [False] * rows)],
        )
        stage = prepared_base().pipeline.shard_stage
        keep = build_shards(table, ShardingSpec(32, "keep_partial"), stage, table_digest(table))
        drop = build_shards(table, ShardingSpec(32, "drop_partial"), stage, table_digest(table))
        self.assertEqual([(shard["start_row"], shard["end_row"]) for shard in keep], [(0, 32), (32, 64), (64, 65)])
        self.assertEqual([(shard["start_row"], shard["end_row"]) for shard in drop], [(0, 32), (32, 64)])

    def test_shard_staleness_tracks_stage_version_and_spec_binding(self) -> None:
        prepared = prepared_base()
        table = table_from_dataset(prepared.dataset)
        for stage in prepared.pipeline.stages[:-1]:
            table, _ = apply_stage(
                table,
                stage,
                prepared.pipeline.fixed_scale,
                prepared.pipeline.sharding,
                32,
            )
        shards = build_shards(
            table,
            prepared.pipeline.sharding,
            prepared.pipeline.shard_stage,
            table_digest(table),
        )
        rows = [
            {
                **shard,
                "stage_id": shard["source_stage_id"],
                "fingerprint": prepared.fingerprint,
            }
            for shard in shards
        ]
        self.assertFalse(any(shard["stale"] for shard in classify_current_shards(rows, prepared)))
        rows[0]["source_stage_version"] += 1
        self.assertTrue(classify_current_shards(rows, prepared)[0]["stale"])
        rows[0]["source_stage_version"] -= 1
        rows[0]["fingerprint"] = "0" * 64
        self.assertTrue(classify_current_shards(rows, prepared)[0]["stale"])


if __name__ == "__main__":
    unittest.main()
