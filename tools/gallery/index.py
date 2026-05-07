"""Generate the gallery index page."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ._types import ExampleSpec


@dataclass(frozen=True)
class RenderedExample:
    """An example after the renderer has produced its artifacts.

    Attributes
    ----------
    spec : tools.gallery._types.ExampleSpec
        The original spec from discovery.
    title : str
        First markdown heading found in the example.
    summary : str
        Second markdown line, used as the card description. May be empty.
    md_path : pathlib.Path
        Path to the rendered Markdown file (under ``docs/examples/_built/``).
    thumbnail : pathlib.Path or None
        Path to the thumbnail PNG, or ``None`` if the example produced no
        images.
    """

    spec: ExampleSpec
    title: str
    summary: str
    md_path: Path
    thumbnail: Path | None


_DEFAULT_THUMB = "_assets/default_thumb.png"


def build_index(rendered: list[RenderedExample], *, root: Path) -> str:
    """Return the Markdown text of the gallery index page.

    The index groups cards by section (folder name), shows the section intro
    above the cards, and uses Material's ``grid cards`` block.

    Parameters
    ----------
    rendered : list[tools.gallery.index.RenderedExample]
        All rendered examples.
    root : pathlib.Path
        ``docs/examples/`` — used to compute relative links.

    Returns
    -------
    text : str
        The full Markdown document.
    """
    by_section: dict[str, list[RenderedExample]] = defaultdict(list)
    for rex in rendered:
        by_section[rex.spec.section].append(rex)

    parts: list[str] = ["# Examples\n"]

    for section in sorted(by_section):
        items = by_section[section]
        intro = items[0].spec.section_intro.strip()
        parts.append(intro + "\n" if intro else f"## {section}\n")
        parts.append('<div class="grid cards" markdown>\n')
        for rex in sorted(items, key=lambda r: r.spec.source.name):
            href = rex.md_path.relative_to(root).as_posix()
            thumb = (
                rex.thumbnail.relative_to(root).as_posix()
                if rex.thumbnail is not None
                else _DEFAULT_THUMB
            )
            summary = f"\n  {rex.summary}" if rex.summary else ""
            parts.append(f"- ![]({thumb})\n\n  **[{rex.title}]({href})**{summary}\n")
        parts.append("</div>\n")

    return "\n".join(parts)
