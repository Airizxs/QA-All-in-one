"""Module entry point for `python -m seo_checker`."""

from __future__ import annotations

import sys

from .cli import main as cli_main
from .menu import run_menu


def entry() -> None:  # pragma: no cover - runtime wrapper
    if len(sys.argv) > 1:
        sys.exit(cli_main(sys.argv[1:]))
    run_menu()


if __name__ == "__main__":  # pragma: no cover - module execution
    entry()
