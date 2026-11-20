"""Tkinter diagnostic interface (read-only).

Shows the timeline, injection states and transitions, observation curves
with fixed-point error bounds, unknown-injection markers, the current
format version and the snapshot list. Rendering order is stable (ORDER BY
seq) and refreshing never writes to the ledger (read-only connection).
"""

import os
import sqlite3

PHASE_COLOR = {"pre": "#4477aa", "active": "#cc3311", "post": "#228833"}
STATE_COLOR = {"reverted": "black", "failed": "orange", "unknown": "red",
               "active": "blue", "observed": "blue", "pending": "gray"}


def _connect_ro(path):
    uri = "file:%s?mode=ro" % os.path.abspath(path)
    return sqlite3.connect(uri, uri=True)


def launch(ledger_path, snapshot_dir="snapshots"):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Chaos ledger diagnostics")
    root.geometry("980x720")

    meta_var = tk.StringVar()
    tk.Label(root, textvariable=meta_var, anchor="w", justify="left",
             font=("Consolas", 10)).pack(fill="x", padx=6, pady=4)

    tree = ttk.Treeview(root, columns=("seq", "type", "target", "state",
                                       "planned", "code"),
                        show="headings", height=10)
    for col, w in (("seq", 50), ("type", 90), ("target", 90),
                   ("state", 90), ("planned", 90), ("code", 160)):
        tree.heading(col, text=col)
        tree.column(col, width=w, anchor="center")
    tree.pack(fill="x", padx=6)

    trans = tk.Text(root, height=10, font=("Consolas", 9))
    trans.pack(fill="both", expand=False, padx=6, pady=4)

    canvas = tk.Canvas(root, bg="white", height=260)
    canvas.pack(fill="both", expand=True, padx=6, pady=4)

    def refresh():
        conn = _connect_ro(ledger_path)
        try:
            meta = dict(conn.execute("SELECT key,value FROM meta"))
            snaps = sorted(os.listdir(snapshot_dir)) \
                if os.path.isdir(snapshot_dir) else []
            meta_var.set(
                "format_version=%s seed=%s\nspec_digest=%s\ntimeline_digest=%s\n"
                "content_digest=%s\nsnapshots: %s"
                % (meta.get("format_version"), meta.get("seed"),
                   meta.get("spec_digest"), meta.get("timeline_digest"),
                   meta.get("content_digest"), ", ".join(snaps) or "(none)"))
            tree.delete(*tree.get_children())
            for row in conn.execute(
                    "SELECT seq,itype,target,state,planned_start_ms,"
                    "COALESCE(business_code,'') FROM injections ORDER BY seq"):
                tag = row[3] if row[3] in STATE_COLOR else "pending"
                tree.insert("", "end", values=row, tags=(tag,))
            for state, color in STATE_COLOR.items():
                tree.tag_configure(state, foreground=color)
            trans.delete("1.0", "end")
            for row in conn.execute(
                    "SELECT inj_seq,from_state,to_state,reason,mono_ms"
                    " FROM transitions ORDER BY id"):
                trans.insert("end", "seq=%s %s -> %s (%s) @%dms\n" % row)
            _draw_curves(conn, canvas)
        finally:
            conn.close()

    tk.Button(root, text="Refresh (read-only)", command=refresh).pack(pady=2)
    refresh()
    root.mainloop()


def _draw_curves(conn, canvas):
    canvas.delete("all")
    width = int(canvas["width"]) if canvas["width"].isdigit() else 940
    rows = conn.execute(
        "SELECT DISTINCT inj_seq FROM requests ORDER BY inj_seq").fetchall()
    if not rows:
        canvas.create_text(20, 20, anchor="nw", text="no observations yet")
        return
    y0 = 10
    for (seq,) in rows:
        scale_row = conn.execute(
            "SELECT scale, rate_err_bound FROM observations"
            " WHERE inj_seq=? LIMIT 1", (seq,)).fetchone()
        scale = scale_row[0] if scale_row else 10000
        err = scale_row[1] if scale_row else 1
        pts = conn.execute(
            "SELECT phase, req_idx, latency_ms, ok FROM requests"
            " WHERE inj_seq=? ORDER BY phase, req_idx", (seq,)).fetchall()
        max_lat = max([p[2] for p in pts] + [1])
        x = 40
        canvas.create_text(10, y0 + 10, anchor="nw", text="seq %d" % seq)
        for phase, _idx, lat, ok in pts:
            h = int(lat / max_lat * 40)
            color = PHASE_COLOR.get(phase, "black") if ok else "magenta"
            canvas.create_line(x, y0 + 50, x, y0 + 50 - h, fill=color)
            x += max(2, (width - 80) // max(1, len(pts)))
        bound = 50 - int((err / scale) * 50 * 1000 / max(max_lat, 1))
        canvas.create_line(40, y0 + bound, x, y0 + bound,
                           fill="gray", dash=(2, 2))
        canvas.create_text(x + 6, y0 + 10, anchor="nw",
                           text="err bound <= %d/%s" % (err, scale),
                           font=("Consolas", 8))
        y0 += 60
