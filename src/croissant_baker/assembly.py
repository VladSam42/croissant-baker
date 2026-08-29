"""Turning what a handler described into distribution entries.

The handler names files logically and states format globs once; only the
generator knows what is on disk. Assembling ``encodingFormat`` and resolving
FileSet ``includes`` is that translation, and it is the generator's alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from croissant_baker import compression


def _encoding_formats(format_media_type: str, relative_path: str) -> List[str]:
    """The media types describing one file: its format, then its wrapper.

    A gzipped CSV is ``["text/csv", "application/gzip"]``. Croissant 1.1 gives
    ``sc:encodingFormat`` cardinality MANY, and this shape is proposal 2 of
    mlcommons/croissant#635, which is still open on how single-file compression
    should be expressed. The alternatives are worse: ``+``-concatenation
    (proposal 1) produces a string no consumer matches, and ``containedIn`` is
    the spec's mechanism for archives, whose extraction mlcroissant gates on
    tar/zip.

    Order matters. mlcroissant decides to decompress from the filename suffix,
    not from this list, and re-applies it on each iteration until a media type
    it recognises returns — so the format must come first. Reversed, a ``.gz``
    file is gunzipped twice.

    Handlers report only the format; the wrapper is the pipeline's to know.
    """
    wrapper = compression.compression_for(Path(relative_path).name)
    if wrapper is None:
        return [format_media_type]
    return [format_media_type, wrapper.media_type]


#: Characters that make an ``includes`` entry a pattern rather than a filename.
_GLOB_CHARS = ("*", "?", "[")


def _wrappers_in(entries: list) -> List:
    """The compressions among ``entries``, in registry order.

    Called per handler batch, so a FileSet declares only the wrappers its own
    files use. Registry order rather than set order keeps the output stable.
    """
    present = {
        comp.suffix
        for entry in entries
        if (comp := compression.compression_for(entry.path.name))
    }
    return [c for c in compression.compressions() if c.suffix in present]


def _resolve_file_sets(file_sets: list, stored_paths: dict, wrappers: list) -> list:
    """Point each FileSet at the files as stored, and at the wrappers they use.

    A handler names its files logically; only the generator knows what is on
    disk. An exact name becomes the stored path or paths it stands for, and a
    pattern gains one variant per compression present. Expanding an exact name
    as if it were a pattern would append a second suffix.

    Args:
        file_sets: FileSets from one handler batch. Mutated in place.
        stored_paths: Logical dataset-relative path to the stored paths sharing
            it. A plain file and its wrapper share one logical key.
        wrappers: Compressions present in this batch, in registry order.

    Returns:
        ``file_sets``, for chaining.
    """
    wrapper_types = [c.media_type for c in wrappers]
    for file_set in file_sets:
        formats = list(file_set.encoding_formats or [])
        file_set.encoding_formats = formats + [
            t for t in wrapper_types if t not in formats
        ]

        includes = getattr(file_set, "includes", None)
        if not includes:
            continue
        resolved: List[str] = []
        for pattern in includes:
            if any(char in pattern for char in _GLOB_CHARS):
                resolved.extend(compression.expand_globs([pattern], wrappers))
            else:
                # An exact name the scan did not find would be a phantom entry.
                resolved.extend(stored_paths.get(pattern, ()))
        file_set.includes = list(dict.fromkeys(resolved))
    return file_sets
