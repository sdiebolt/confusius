"""Datasets for confusius examples and tutorials."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._cybis_pereira_2026 import _BIDS_ROOT as _cybis_pereira_2026_bids_root
from ._cybis_pereira_2026 import _TOTAL_SIZE_BYTES as _cybis_pereira_2026_size
from ._cybis_pereira_2026 import fetch_cybis_pereira_2026
from ._nunez_elizalde_2022 import _BIDS_ROOT as _nunez_elizalde_2022_bids_root
from ._nunez_elizalde_2022 import _TOTAL_SIZE_BYTES as _nunez_elizalde_2022_size
from ._nunez_elizalde_2022 import fetch_nunez_elizalde_2022
from ._utils import format_bytes, get_datasets_dir

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "fetch_cybis_pereira_2026",
    "fetch_nunez_elizalde_2022",
    "get_datasets_dir",
    "list_datasets",
]

_REGISTRY: tuple[tuple[str, int, str], ...] = (
    (
        "fetch_cybis_pereira_2026",
        _cybis_pereira_2026_size,
        _cybis_pereira_2026_bids_root,
    ),
    (
        "fetch_nunez_elizalde_2022",
        _nunez_elizalde_2022_size,
        _nunez_elizalde_2022_bids_root,
    ),
)
"""Registry of (fetcher_name, total_size_bytes, bids_root_dirname) per dataset."""


def list_datasets(data_dir: str | Path | None = None) -> None:
    """Print a table of available datasets.

    Displays each fetch function, the total download size, and whether the
    dataset is already on disk in the cache directory.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Cache directory to check for existing datasets. Defaults to the
        directory resolved by
        [`get_datasets_dir`][confusius.datasets.get_datasets_dir].
    """
    import sys

    from rich.console import Console
    from rich.table import Table

    datasets_dir = get_datasets_dir(data_dir)
    console = Console()

    # Rich rewrites box-drawing chars when the terminal is ASCII-only, but cell
    # contents are passed through verbatim, so ✓/✗ must be guarded explicitly.
    unicode_ok = "utf" in (sys.stdout.encoding or "").lower()
    yes_mark = "[green]✓[/green]" if unicode_ok else "[green]yes[/green]"
    no_mark = "[red]✗[/red]" if unicode_ok else "[red]no[/red]"

    table = Table(title="Available Datasets")
    table.add_column("Fetch function")
    table.add_column("Size", justify="right")
    table.add_column("On disk", justify="center")

    for fetcher_name, size_bytes, bids_root in _REGISTRY:
        path = datasets_dir / bids_root
        is_cached = path.exists() and any(path.iterdir())
        table.add_row(
            fetcher_name,
            format_bytes(size_bytes),
            yes_mark if is_cached else no_mark,
        )

    console.print(table)
