from pathlib import Path
from typing import Sequence


def discover_files(
    root: Path, exts: set[str], recursive: bool = True
) -> Sequence[Path]:
    """Scan *root* for files whose suffix (lowercased) is in *exts*.

    Args:
        root: Directory to search.
        exts: Set of lowercase extensions to include, e.g. ``{".pdf", ".png"}``.
        recursive: When ``True`` (default) descend into sub-directories.

    Returns:
        Sorted list of matching :class:`~pathlib.Path` objects (stable order).
    """
    glob_fn = root.rglob if recursive else root.glob
    paths = [p for p in glob_fn("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)
