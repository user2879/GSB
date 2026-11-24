from __future__ import annotations

import socket
import sys
import time


def main() -> int:
    port = int(sys.argv[1])
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(8)
    while True:
        conn, _ = listener.accept()
        # Deliberately unresponsive peer: the client timeout must be real.
        time.sleep(30)
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
