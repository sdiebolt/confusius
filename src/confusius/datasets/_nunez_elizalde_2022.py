"""Fetcher for the Nunez-Elizalde et al. (2022) fUSI-BIDS dataset."""

from __future__ import annotations

import re
from pathlib import Path

import pooch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ._osf import _OSF_DOWNLOAD_BASE, get_index
from ._pooch import quiet_pooch_logger, retrieve_with_retries
from ._utils import get_datasets_dir

_OSF_PROJECT_ID = "43skw"
_BIDS_ROOT = "nunez-elizalde-2022-bids"
_TOTAL_SIZE_BYTES = 6_982_575_320

_MISSING_INDEX_HINT = (
    "Run 'nunez-upload --index-only' from the nunez-elizalde-2022-bids "
    "repository to generate it."
)


def _filter_files(
    index: dict[str, str],
    subjects: list[str] | None,
    sessions: list[str] | None,
    tasks: list[str] | None,
    acqs: list[str] | None,
) -> dict[str, str]:
    """Filter the index to files matching the requested subjects/sessions/tasks.

    Top-level BIDS metadata files (dataset_description.json, participants.*,
    etc.) and subject-level files (sessions.tsv/.json) are always included.

    Parameters
    ----------
    index : dict[str, str]
        Full dataset index as returned by `get_index`.
    subjects : list[str] or None
        Subject IDs to include (without "sub-" prefix), e.g. `["CR020"]`. If `None`, all
        subjects are included.
    sessions : list[str] or None
        Session IDs to include (without "ses-" prefix), e.g. `["20191122"]`. If `None`,
        all sessions are included.
    tasks : list[str] or None
        Task names to include, e.g. `["kalatsky", "spontaneous"]`. If `None`, all tasks
        are included. Only applies to `fusi/` files; `angio/` files are always included.
    acqs : list[str] or None
        Acquisition labels to include (without `acq-`), e.g. `["slice03"]`. If `None`,
        all acquisitions are included. Only applies to `fusi/` files.

    Returns
    -------
    dict[str, str]
        Subset of the index matching the filters.
    """
    filtered: dict[str, str] = {}

    for path, osf_id in index.items():
        parts = Path(path).parts

        # Handle derivatives explicitly so subject/session filters apply there too.
        if parts[0] == "derivatives":
            derivative_sub = next((p for p in parts if p.startswith("sub-")), None)
            derivative_ses = next((p for p in parts if p.startswith("ses-")), None)

            # Derivative files without subject/session (e.g. dataset_description,
            # shared lookup tables) are always included.
            if derivative_sub is None:
                filtered[path] = osf_id
                continue

            sub_id = derivative_sub.removeprefix("sub-")
            if subjects is not None and sub_id not in subjects:
                continue

            if derivative_ses is not None:
                ses_id = derivative_ses.removeprefix("ses-")
                if sessions is not None and ses_id not in sessions:
                    continue

            filtered[path] = osf_id
            continue

        # Always include top-level BIDS files (no subject folder).
        if not parts[0].startswith("sub-"):
            filtered[path] = osf_id
            continue

        # Subject filter.
        sub_id = parts[0].removeprefix("sub-")
        if subjects is not None and sub_id not in subjects:
            continue

        # Subject-level files (e.g. sub-CR020_sessions.tsv).
        if len(parts) == 1 or not parts[1].startswith("ses-"):
            filtered[path] = osf_id
            continue

        # Session filter.
        ses_id = parts[1].removeprefix("ses-")
        if sessions is not None and ses_id not in sessions:
            continue

        # Task filter (only applies to fusi/ files).
        if tasks is not None and len(parts) >= 3 and parts[2] == "fusi":
            match = re.search(r"task-([^_]+)", parts[-1])
            if match is None or match.group(1) not in tasks:
                continue

        # Acquisition filter (only applies to fusi/ files).
        if acqs is not None and len(parts) >= 3 and parts[2] == "fusi":
            match = re.search(r"acq-([^_]+)", parts[-1])
            if match is None or match.group(1) not in acqs:
                continue

        filtered[path] = osf_id

    return filtered


def fetch_nunez_elizalde_2022(
    data_dir: str | Path | None = None,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    tasks: list[str] | None = None,
    acqs: list[str] | None = None,
    refresh: bool = False,
) -> Path:
    """Fetch the Nunez-Elizalde 2022 fUSI-BIDS dataset.

    Downloads simultaneous neural activity and cerebral blood volume recordings
    in awake mice, converted to fUSI-BIDS format from Nunez-Elizalde et al.
    (2022).

    Files are downloaded on first call and cached locally. Subsequent calls
    with the same `data_dir` return immediately for already-cached files.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory in which to cache the dataset. Defaults to the platform
        cache directory (e.g. `~/.cache/confusius` on Linux,
        `~/Library/Caches/confusius` on macOS,
        `%LOCALAPPDATA%\\confusius\\Cache` on Windows),
        overridable via the `CONFUSIUS_DATA` environment variable.
    subjects : list[str], optional
        Subject IDs to download (without "sub-" prefix), e.g. `["CR020"]`. If not
        provided, all subjects are downloaded.
    sessions : list[str], optional
        Session IDs to download (without "ses-" prefix), e.g. `["20191122"]`. If not
        provided, all sessions are downloaded.
    tasks : list[str], optional
        Task names to download, e.g. `["kalatsky", "spontaneous"]`. If not provided, all
        tasks are downloaded. Angiography files are always included regardless of this
        filter.
    acqs : list[str], optional
        Acquisition labels to download (without `acq-`), e.g. `["slice03"]`. If not
        provided, all acquisitions are downloaded. Only applies to `fusi/` files;
        angiography files are always included.
    refresh : bool, default: False
        Whether to re-fetch the dataset index from OSF and download any files that are
        missing locally. If `False` and all requested files are already cached, the
        function returns immediately without any network access.

    Returns
    -------
    pathlib.Path
        Path to the BIDS root directory of the cached dataset.

    References
    ----------
    [^1]:
        Nunez-Elizalde, A.O. et al. (2022). Neural correlates of blood flow measured by
        ultrasound. *Neuron*, 110(10), 1631–1640.
        [https://doi.org/10.1016/j.neuron.2022.02.012](https://doi.org/10.1016/j.neuron.2022.02.012)

    [^2]:
        fUSI-BIDS dataset on OSF: [https://osf.io/43skw/](https://osf.io/43skw/)
    """
    bids_dir = get_datasets_dir(data_dir) / _BIDS_ROOT
    bids_dir.mkdir(parents=True, exist_ok=True)

    index = get_index(
        bids_dir,
        _OSF_PROJECT_ID,
        _BIDS_ROOT,
        refresh=refresh,
        missing_index_hint=_MISSING_INDEX_HINT,
    )
    files = _filter_files(index, subjects, sessions, tasks, acqs)

    missing = {p: i for p, i in files.items() if not (bids_dir / p).exists()}
    if not missing:
        return bids_dir

    with quiet_pooch_logger():
        pooch_logger = pooch.get_logger()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Downloading dataset...", total=len(missing))

            for rel_path, osf_id in missing.items():
                dest = bids_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                progress.update(
                    task,
                    description=f"Downloading [bold]{Path(rel_path).name}[/bold]",
                )
                retrieve_with_retries(
                    url=_OSF_DOWNLOAD_BASE.format(osf_id.lstrip("/")),
                    dest=dest,
                    logger=pooch_logger,
                )
                progress.advance(task)

            progress.update(task, description="Download complete.")

    return bids_dir
