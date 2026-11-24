from __future__ import annotations

import json

from batch_diagnostic.gui import build_overview, build_plan_rows, render_overview_json
from batch_diagnostic.planning import build_plan
from batch_diagnostic.spec import load_spec


def test_gui_headless_builders_work_without_tk(system_spec):
    overview = dict(build_overview(system_spec))
    assert overview["requests"] == str(len(system_spec.requests))
    rows = build_plan_rows(system_spec)
    assert rows and all("batch_id" in row for row in rows)
    rendered = json.loads(render_overview_json(system_spec))
    assert rendered["plan"] == rows


def test_plan_independent_of_dict_and_dispatch_order(system_spec, tmp_path):
    requests_path = tmp_path / "requests.json"
    workers_path = tmp_path / "workers.json"
    budget_path = tmp_path / "budget.json"
    requests_path.write_text(
        json.dumps([request.__dict__ for request in system_spec.requests])
    )
    shuffled_workers = list(reversed(system_spec.workers))
    workers_path.write_text(
        json.dumps([worker.__dict__ for worker in shuffled_workers])
    )
    budget_path.write_text(json.dumps(system_spec.budget.__dict__))
    reloaded = load_spec(str(requests_path), str(workers_path), str(budget_path))
    direct = [batch.to_dict() for batch in build_plan(system_spec).batches]
    shuffled = [batch.to_dict() for batch in build_plan(reloaded).batches]
    assert direct == shuffled
