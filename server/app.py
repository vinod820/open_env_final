from social_engineer_arena.server.app import app as app
from social_engineer_arena.server.app import main as _pkg_main


def main() -> None:
    _pkg_main()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
