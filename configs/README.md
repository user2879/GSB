# Generated specifications

These files are emitted by `python -m shard_diag generate --root .` with the
fixed integer seed `20261126` in `shard_diag.generator`:

- `local.json`: route table versions 1 and 2; physical databases are bound to
  version 2.
- `local_v3.json`: the same fixtures with databases bound to the evolved route
  table version 3.
- `local_edited.json`: an edited event stream used to demonstrate whole-session
  snapshot replacement and replay.

The files are intentionally committed so a reviewer can inspect or hand-edit
them. Regeneration is deterministic and never downloads external data.

