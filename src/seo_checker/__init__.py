"""SEO checker package."""

from importlib import metadata

try:
    __version__ = metadata.version("seo-checker")
except metadata.PackageNotFoundError:  # pragma: no cover - local dev fallback
    __version__ = "0.1.0.dev0"

from .cli import (  # noqa: F401
    compute_score,
    compute_section_scores,
    console_entry,
    main,
    parse_args,
    print_results_as_tables,
    run_all_checks,
)

__all__ = [
    "__version__",
    "compute_score",
    "compute_section_scores",
    "console_entry",
    "main",
    "parse_args",
    "print_results_as_tables",
    "run_all_checks",
]
