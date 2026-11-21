from .runtime_control import install_signal_handlers


install_signal_handlers()

from .cli import main  # noqa: E402  (handler must be installed before heavy imports)


if __name__ == "__main__":
    raise SystemExit(main())
