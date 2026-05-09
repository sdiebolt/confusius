"""Generate the gallery index page."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ._types import ExampleSpec


@dataclass(frozen=True)
class RenderedExample:
    """An example after the renderer has produced its artifacts."""

    spec: ExampleSpec
    title: str
    summary: str
    md_path: Path
    thumbnail_light: Path | None
    thumbnail_dark: Path | None


_DEFAULT_THUMB = "_assets/default_thumb.svg"


def _demote_h1(text: str) -> str:
    """Demote a leading H1 heading to H2 so the index has a single page title."""
    lines = text.split("\n", 1)
    first = lines[0]
    if first.startswith("# ") and not first.startswith("## "):
        first = "#" + first
    return first + ("\n" + lines[1] if len(lines) > 1 else "")


def _card_image_markdown(rex: RenderedExample, *, root: Path, href: str) -> str:
    """Return theme-aware thumbnail markup for one example card."""
    if rex.thumbnail_light is None or rex.thumbnail_dark is None:
        return f"[![]({_DEFAULT_THUMB})]({href})"

    light = rex.thumbnail_light.relative_to(root).as_posix()
    dark = rex.thumbnail_dark.relative_to(root).as_posix()
    alt = rex.title.replace('"', '&quot;')
    return (
        f'[<img class="skip-lightbox" src="{light}#only-light" alt="{alt}">'
        f'<img class="skip-lightbox" src="{dark}#only-dark" alt="{alt}">]({href})'
    )


def build_index(rendered: list[RenderedExample], *, root: Path) -> str:
    """Return the Markdown text of the gallery index page."""
    by_section: dict[str, list[RenderedExample]] = defaultdict(list)
    for rendered_example in rendered:
        by_section[rendered_example.spec.section].append(rendered_example)

    parts: list[str] = [
        "# Examples\n\n"
        "This section collects runnable example workflows for ConfUSIus, organized by analysis\n"
        "task and learning level.\n\n"
        "Examples are authored as plain Python files, executed during docs generation, and\n"
        "rendered as notebook-style pages with outputs included.\n\n"
    ]

    for section in sorted(by_section):
        items = by_section[section]
        intro = _demote_h1(items[0].spec.section_intro.strip())
        parts.append((intro if intro else f"## {section}") + "\n\n")
        parts.append('<div class="grid cards examples-cards" markdown>\n\n')
        for rendered_example in sorted(items, key=lambda item: item.spec.source.name):
            href = rendered_example.md_path.relative_to(root).as_posix()
            image = _card_image_markdown(rendered_example, root=root, href=href)
            card = f"-   {image}\n\n    ---\n\n    **[{rendered_example.title}]({href})**"
            if rendered_example.summary:
                card += f"\n\n    {rendered_example.summary}"
            parts.append(card + "\n\n")
        parts.append("</div>\n\n")

    return "".join(parts)
