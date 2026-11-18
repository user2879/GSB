from __future__ import annotations

from pathlib import Path

from slowquery_workbench import Workbench


SAMPLE_QUERY = (
    "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id "
    "WHERE o.status = 'paid' AND o.amount >= 10 AND o.amount >= 100 "
    "AND c.city IN ('Shanghai', 'Beijing') ORDER BY o.id"
)


def new_sample_workbench(root: str | Path, name: str = "test-session") -> Workbench:
    path = Path(root) / f"{name}.sqlite"
    workbench = Workbench.create(path, name=name, overwrite=True)
    workbench.initialize_samples()
    return workbench
