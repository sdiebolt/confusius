"""Shared dataclasses for the gallery builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExampleSpec:
    """A single discovered example file.

    Attributes
    ----------
    source : pathlib.Path
        Absolute path to the percent-format ``.py`` file.
    section : str
        Section folder name (e.g. ``"io"``).
    section_intro : str
        Contents of the sibling ``_section.md`` file, or empty string if none.
    """

    source: Path
    section: str
    section_intro: str
