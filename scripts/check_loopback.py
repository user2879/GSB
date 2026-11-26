import socket


def main() -> int:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    client = socket.create_connection(("127.0.0.1", port), timeout=2)
    accepted, _ = server.accept()
    payload = b"arg-review-loopback-ok"
    client.sendall(payload)
    received = accepted.recv(len(payload))
    client.close()
    accepted.close()
    server.close()

    if received != payload:
        raise SystemExit("loopback echo mismatch")
    print(f"127.0.0.1:{port} loopback ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
