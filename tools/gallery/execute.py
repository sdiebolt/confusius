"""Execute percent-format examples with an explicit ipykernel."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

import jupytext
import nbformat
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

_THEME_SETUP = {
    "light": [
        "import warnings",
        "import matplotlib as mpl",
        "from matplotlib import style as mpl_style",
        "warnings.filterwarnings('ignore', message='IProgress not found.*')",
        "mpl_style.use('default')",
        "mpl.rcParams['figure.dpi'] = 160",
        "mpl.rcParams['savefig.dpi'] = 200",
        "mpl.rcParams['figure.figsize'] = (7, 4)",
        "mpl.rcParams['figure.facecolor'] = '#ffffff'",
        "mpl.rcParams['axes.facecolor'] = '#ffffff'",
        "mpl.rcParams['savefig.facecolor'] = '#ffffff'",
        "try:",
        "    from matplotlib_inline.backend_inline import set_matplotlib_formats",
        "    set_matplotlib_formats('retina')",
        "except Exception:",
        "    pass",
    ],
    "dark": [
        "import warnings",
        "import matplotlib as mpl",
        "from matplotlib import style as mpl_style",
        "warnings.filterwarnings('ignore', message='IProgress not found.*')",
        "mpl_style.use('dark_background')",
        "mpl.rcParams['figure.dpi'] = 160",
        "mpl.rcParams['savefig.dpi'] = 200",
        "mpl.rcParams['figure.figsize'] = (7, 4)",
        "mpl.rcParams['figure.facecolor'] = '#111720'",
        "mpl.rcParams['axes.facecolor'] = '#111720'",
        "mpl.rcParams['savefig.facecolor'] = '#111720'",
        "try:",
        "    from matplotlib_inline.backend_inline import set_matplotlib_formats",
        "    set_matplotlib_formats('retina')",
        "except Exception:",
        "    pass",
    ],
}


def read_example(source: Path) -> nbformat.NotebookNode:
    """Read one percent-format example into a notebook without executing it."""
    return jupytext.read(source)


def _theme_setup_cell(theme: Literal["light", "dark"]) -> nbformat.NotebookNode:
    """Return a hidden setup cell for themed execution."""
    return nbformat.v4.new_code_cell(
        "\n".join(_THEME_SETUP[theme]),
        metadata={"tags": ["_gallery_internal"]},
    )


def execute_example(
    source: Path,
    *,
    timeout: int = 600,
    theme: Literal["light", "dark"] | None = None,
) -> tuple[nbformat.NotebookNode, float]:
    """Execute one percent-format ``.py`` example end-to-end."""
    notebook = read_example(source)
    if theme is not None:
        notebook.cells.insert(0, _theme_setup_cell(theme))

    kernel_manager = KernelManager()
    kernel_manager.kernel_cmd = [  # type: ignore[unresolved-attribute]
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]

    client = NotebookClient(notebook, km=kernel_manager, timeout=timeout)
    client.owns_km = True

    start = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as exc:
        raise RuntimeError(f"Failed to execute {source}: {exc}") from exc
    elapsed = time.perf_counter() - start
    return notebook, elapsed
