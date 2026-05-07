"""High-level orchestrator: discover, cache, execute, render, index."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nbformat

from . import cache, discover, execute, index, render
from ._types import ExampleSpec
from .index import RenderedExample


def _python_version() -> str:
    return sys.version.split()[0]


def _extract_title_and_summary(source: Path) -> tuple[str, str]:
    """Pull the first ``# Title`` and the next non-empty markdown line."""
    title = source.stem
    summary = ""
    text = source.read_text()
    in_md = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "# %% [markdown]":
            in_md = True
            continue
        if stripped.startswith("# %%"):
            in_md = False
            continue
        if not in_md:
            continue
        body = stripped.lstrip("#").strip()
        if body.startswith("# "):
            title = body[2:].strip()
        elif body and not summary:
            summary = body
        if title != source.stem and summary:
            break
    return title, summary


def _write_cached_artifacts(
    cache_dir: Path, dest_dir: Path, base_name: str
) -> Path | None:
    """Copy cached files for one example into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in cache_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copyfile(item, dest_dir / item.name)
    thumb = dest_dir / f"{base_name}_thumb.png"
    return thumb if thumb.is_file() else None


def _build_one(
    spec: ExampleSpec,
    *,
    built_dir: Path,
    cache_root: Path,
    deps_fingerprint: str,
) -> RenderedExample:
    base_name = spec.source.stem
    out_dir = built_dir / spec.section
    out_dir.mkdir(parents=True, exist_ok=True)

    key = cache.cache_key(
        spec, deps_fingerprint=deps_fingerprint, python_version=_python_version()
    )
    cache_entry = cache.cache_dir(cache_root, key) / spec.section / base_name

    title, summary = _extract_title_and_summary(spec.source)

    if cache_entry.is_dir():
        thumb = _write_cached_artifacts(cache_entry, out_dir, base_name)
        md_path = out_dir / f"{base_name}.md"
        return RenderedExample(
            spec=spec,
            title=title,
            summary=summary,
            md_path=md_path,
            thumbnail=thumb,
        )

    nb, seconds = execute.execute_example(spec.source)

    # Per-example artifacts get written into a temp scratch first, then copied
    # both to the built dir and into the cache, so the two copies are
    # bit-identical.
    scratch = cache_root / "_scratch" / spec.section / base_name
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    md_path, thumbnail = render.render_notebook(
        nb, out_dir=scratch, base_name=base_name, runtime_seconds=seconds
    )

    # Downloadable .py / .ipynb live next to the .md.
    shutil.copyfile(spec.source, scratch / f"{base_name}.py")
    nbformat.write(nb, scratch / f"{base_name}.ipynb")

    # Mirror scratch to built_dir/<section>/.
    for item in scratch.iterdir():
        target = out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(item, target)

    # Mirror scratch to cache.
    cache_entry.parent.mkdir(parents=True, exist_ok=True)
    if cache_entry.exists():
        shutil.rmtree(cache_entry)
    shutil.copytree(scratch, cache_entry)

    final_thumb = out_dir / f"{base_name}_thumb.png" if thumbnail is not None else None
    return RenderedExample(
        spec=spec,
        title=title,
        summary=summary,
        md_path=out_dir / f"{base_name}.md",
        thumbnail=final_thumb,
    )


def build_gallery(
    *,
    examples_root: Path,
    built_dir: Path,
    cache_root: Path,
    deps_fingerprint: str,
) -> None:
    """Run the full pipeline and write the gallery to disk.

    Parameters
    ----------
    examples_root : pathlib.Path
        ``docs/examples/`` (the source-of-truth tree).
    built_dir : pathlib.Path
        Where rendered Markdown + downloads go (typically
        ``docs/examples/_built/``).
    cache_root : pathlib.Path
        Cache directory (typically ``.cache/gallery/``).
    deps_fingerprint : str
        Locked-dependency fingerprint used in the cache key.
    """
    specs = discover.discover(examples_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    rendered: list[RenderedExample] = []
    for spec in specs:
        rendered.append(
            _build_one(
                spec,
                built_dir=built_dir,
                cache_root=cache_root,
                deps_fingerprint=deps_fingerprint,
            )
        )

    index_md = index.build_index(rendered, root=examples_root)
    (examples_root / "index.md").write_text(index_md)
