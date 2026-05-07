"""Cache key + cache directory helpers for the gallery builder."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ._types import ExampleSpec


def cache_key(
    spec: ExampleSpec,
    *,
    deps_fingerprint: str,
    python_version: str,
) -> str:
    """Compute a deterministic cache key for one example.

    The key is the sha256 of the source bytes, the locked-dependency
    fingerprint, and the Python version. Any change in these inputs produces a
    fresh key, forcing a re-execution.

    Parameters
    ----------
    spec : tools.gallery._types.ExampleSpec
        The example to hash.
    deps_fingerprint : str
        A textual representation of the relevant locked dependencies (typically
        a ``\n``-separated list of ``name==version`` lines).
    python_version : str
        ``sys.version`` or equivalent string identifying the interpreter.

    Returns
    -------
    digest : str
        Hex sha256 digest (64 characters).
    """
    h = hashlib.sha256()
    h.update(spec.source.read_bytes())
    h.update(b"\x00")
    h.update(deps_fingerprint.encode("utf-8"))
    h.update(b"\x00")
    h.update(python_version.encode("utf-8"))
    return h.hexdigest()


def cache_dir(root: Path, key: str) -> Path:
    """Return the cache directory for ``key`` rooted at ``root``.

    Parameters
    ----------
    root : pathlib.Path
        Root cache directory.
    key : str
        Cache key (a hex digest).

    Returns
    -------
    path : pathlib.Path
        ``root / key``.
    """
    return root / key
