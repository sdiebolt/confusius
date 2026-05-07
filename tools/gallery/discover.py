"""Discover example source files under ``docs/examples/``."""

from __future__ import annotations

from pathlib import Path

from ._types import ExampleSpec


def discover(root: Path) -> list[ExampleSpec]:
    """Return one [ExampleSpec][tools.gallery._types.ExampleSpec] per example file.

    Walks ``root`` looking for ``<section>/<example>.py`` files. Files whose
    name starts with ``_`` are skipped, as are sections without any non-skipped
    files.

    Parameters
    ----------
    root : pathlib.Path
        The ``docs/examples/`` directory to scan.

    Returns
    -------
    specs : list[tools.gallery._types.ExampleSpec]
        One entry per example, sorted by ``(section, source.name)``.
    """
    specs: list[ExampleSpec] = []
    if not root.is_dir():
        return specs

    for section_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        intro_path = section_dir / "_section.md"
        intro = intro_path.read_text() if intro_path.is_file() else ""

        for source in sorted(section_dir.glob("*.py")):
            if source.name.startswith("_"):
                continue
            specs.append(
                ExampleSpec(
                    source=source,
                    section=section_dir.name,
                    section_intro=intro,
                )
            )

    return specs
