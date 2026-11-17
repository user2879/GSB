"""Tkinter UI and command loop showing the adjudication trace."""
from __future__ import annotations

import shlex

from .arbiter import Request
from .errors import PlatformError


def format_trace(platform, limit=200):
    lines = []
    for row in platform.store.list_adjudications()[-limit:]:
        lines.append(f"[adj] req={row['req_id']} attempt={row['attempt']} "
                     f"node={row['node_id']} sectors={row['sectors']} "
                     f"-> {row['result']} {row['detail']}")
    for row in platform.store.list_transitions()[-limit:]:
        lines.append(f"[lease] {row['lease_id']} {row['from']}->{row['to']} "
                     f"reason={row['reason']} progress={row['progress']} "
                     f"mono={row['mono']:.3f}")
    for row in platform.store.list_batches()[-limit:]:
        lines.append(f"[batch] lease={row['lease_id']} node={row['node_id']} "
                     f"op={row['op']} agg={row['agg']} value={row['value']} "
                     f"match={row['match']}")
    return lines


HELP = """commands:
  submit <node> <read|write> <agg> <sector,sector,...> [delta] [count]
  leases                      list leases and states
  sectors                     list sector versions and owners
  trace                       show adjudication / lease / batch trace
  snapshot                    persist a session snapshot
  recover                     run forced-kill recovery checks
  replay                      replay recorded batches and verify
  quit                        exit
"""


def run_cli(platform):
    print(f"session={platform.session} wl={platform.wl}")
    print(HELP)
    while True:
        try:
            line = input("platform> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        try:
            args = shlex.split(line)
        except ValueError as exc:
            print(f"parse error: {exc}")
            continue
        cmd, rest = args[0], args[1:]
        try:
            if cmd == "submit":
                node, op, agg, sectors = rest[0], rest[1], rest[2], rest[3]
                delta = float(rest[4]) if len(rest) > 4 else 0.0
                count = int(rest[5]) if len(rest) > 5 else 1
                req = Request(req_id=f"cli-{id(rest)}-{len(platform.store.list_adjudications())}",
                              node_id=node, op=op, agg=agg,
                              sectors=tuple(sectors.split(",")),
                              delta=delta, count=count)
                print(platform.submit_and_wait(req, timeout=30))
            elif cmd == "leases":
                for lease in platform.leases.all_leases():
                    print(lease.to_dict())
            elif cmd == "sectors":
                for sid in platform.sectors:
                    print(f"{sid} owner={platform.owners[sid]} "
                          f"version={platform.versions.get(sid)}")
            elif cmd == "trace":
                print("\n".join(format_trace(platform)))
            elif cmd == "snapshot":
                print(platform.snapshot())
            elif cmd == "recover":
                print(platform.recover())
            elif cmd == "replay":
                print(platform.replay())
            elif cmd in ("quit", "exit"):
                break
            else:
                print(HELP)
        except PlatformError as exc:
            print(f"rejected: {exc.code} {exc.detail}")
        except Exception as exc:
            print(f"error: {exc}")


def run_ui(platform):
    import tkinter as tk
    from tkinter import scrolledtext, ttk

    root = tk.Tk()
    root.title(f"sharedbuf platform {platform.session} ({platform.wl})")
    trace_box = scrolledtext.ScrolledText(root, width=110, height=28)
    trace_box.pack(fill="both", expand=True)
    status = tk.StringVar(value="ready")
    ttk.Label(root, textvariable=status).pack(fill="x")

    def refresh():
        trace_box.delete("1.0", "end")
        trace_box.insert("end", "\n".join(format_trace(platform)))
        root.after(1000, refresh)

    def do_submit(op):
        def handler():
            try:
                sectors = tuple(s for s, o in platform.owners.items()
                                if o == "n0")[:2]
                req = Request(req_id=f"ui-{op}-{len(platform.store.list_adjudications())}",
                              node_id="n0", op=op, agg="sum",
                              sectors=sectors, delta=0.25, count=2)
                platform.submit(req)
                status.set(f"submitted {req.req_id}")
            except Exception as exc:
                status.set(f"error: {exc}")
        return handler

    bar = ttk.Frame(root)
    bar.pack(fill="x")
    ttk.Button(bar, text="提交读批次", command=do_submit("read")).pack(side="left")
    ttk.Button(bar, text="提交写批次", command=do_submit("write")).pack(side="left")
    ttk.Button(bar, text="快照", command=lambda: status.set(str(platform.snapshot()))).pack(side="left")
    ttk.Button(bar, text="恢复校验", command=lambda: status.set(str(platform.recover()))).pack(side="left")
    refresh()
    root.mainloop()
