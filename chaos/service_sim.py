"""Deterministic target-service simulator, run as a real child process.

    python -m chaos.service_sim --id svc --port P --control-port C \
        --latency-ms L --error-rate-ppm R --seed S

Data plane: one request per TCP connection on 127.0.0.1:P. Client sends
"REQ\n"; the service sleeps the fixed latency, then answers "OK\n" or
"ERR\n". Errors follow a deterministic seed rule:
    sha256(f"{seed}:{scope}:{counter}") % 10000 < rate_ppm
so a given (seed, rate, request sequence) always yields the same answers.

Control plane (127.0.0.1:C, line protocol):
    PING            -> PONG
    STATUS          -> override=<ppm|-1> base=<ppm> count=<n>
    SET_ERR <ppm>   -> OK   (activate error-rate override, counter reset)
    CLEAR_ERR       -> OK   (deactivate override)
"""

import argparse
import hashlib
import socketserver
import threading
import time


def _draw(seed, scope, counter):
    h = hashlib.sha256(f"{seed}:{scope}:{counter}".encode()).hexdigest()
    return int(h[:8], 16) % 10000


class _State:
    def __init__(self, latency_ms, base_ppm, seed):
        self.latency_ms = latency_ms
        self.base_ppm = base_ppm
        self.seed = seed
        self.override_ppm = -1
        self.base_count = 0
        self.override_count = 0
        self.lock = threading.Lock()

    def decide(self):
        with self.lock:
            if self.override_ppm >= 0:
                n = self.override_count
                self.override_count += 1
                ppm, scope = self.override_ppm, "override"
            else:
                n = self.base_count
                self.base_count += 1
                ppm, scope = self.base_ppm, "base"
        return _draw(self.seed, scope, n) < ppm


def _readline(conn):
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = conn.recv(1024)
        if not chunk:
            break
        buf += chunk
    return buf.decode().strip()


def _make_data_handler(state):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            line = _readline(self.request)
            if line != "REQ":
                return
            time.sleep(state.latency_ms / 1000.0)
            is_err = state.decide()
            self.request.sendall(b"ERR\n" if is_err else b"OK\n")
    return Handler


def _make_control_handler(state):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            line = _readline(self.request)
            parts = line.split()
            if not parts:
                return
            cmd = parts[0].upper()
            if cmd == "PING":
                self.request.sendall(b"PONG\n")
            elif cmd == "STATUS":
                with state.lock:
                    msg = (f"override={state.override_ppm} base={state.base_ppm} "
                           f"count={state.base_count}\n")
                self.request.sendall(msg.encode())
            elif cmd == "SET_ERR" and len(parts) == 2:
                ppm = int(parts[1])
                with state.lock:
                    state.override_ppm = ppm
                    state.override_count = 0
                self.request.sendall(b"OK\n")
            elif cmd == "CLEAR_ERR":
                with state.lock:
                    state.override_ppm = -1
                self.request.sendall(b"OK\n")
            else:
                self.request.sendall(b"E_BADCMD\n")
    return Handler


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(service_id, port, control_port, latency_ms, error_rate_ppm, seed,
          pidfile=None):
    if pidfile:
        import os
        with open(pidfile, "w") as fh:
            fh.write(str(os.getpid()))
    state = _State(latency_ms, error_rate_ppm, seed)
    data = _Server(("127.0.0.1", port), _make_data_handler(state))
    control = _Server(("127.0.0.1", control_port), _make_control_handler(state))
    threading.Thread(target=data.serve_forever, daemon=True).start()
    print(f"service {service_id} data={port} control={control_port}", flush=True)
    try:
        control.serve_forever()
    finally:
        data.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--control-port", type=int, required=True)
    ap.add_argument("--latency-ms", type=int, required=True)
    ap.add_argument("--error-rate-ppm", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pidfile", default=None)
    args = ap.parse_args(argv)
    serve(args.id, args.port, args.control_port, args.latency_ms,
          args.error_rate_ppm, args.seed, pidfile=args.pidfile)


if __name__ == "__main__":
    main()
