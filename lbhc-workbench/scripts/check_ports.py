"""Confirm the reserved loopback high ports are free before a demo run."""

import socket
import sys

BASE, COUNT = 18601, 24


def main():
    busy = []
    for port in range(BASE, BASE + COUNT):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            busy.append(port)
        finally:
            s.close()
    if busy:
        print(f"occupied ports: {busy}")
        return 1
    print(f"ports {BASE}-{BASE + COUNT - 1} on 127.0.0.1 are free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
