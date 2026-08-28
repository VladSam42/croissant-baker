"""Abstract base class for file type handlers."""

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Set

from croissant_baker.sources import FileSource


class InputKind(str, Enum):
    """What a handler needs in order to read a file.

    A handler declares the input it consumes rather than whether it supports
    compression; compression support follows from the declaration.
    """

    #: Reads bytes in order, from a :class:`~croissant_baker.sources.FileSource`.
    #: Works on a wrapped file and a plain one alike. The default.
    STREAM = "stream"
    #: Needs a real file on disk, because it reaches beyond the bytes at that
    #: path — WFDB reads a header together with its sibling .dat and .atr files.
    #: A compressed file is never routed here.
    PATH = "path"


_warned: Set[type] = set()
_warned_calls: Set[tuple] = set()


class FileTypeHandler(ABC):
    """Abstract base class for file type handlers.

    Each handler is responsible for three things:

    - ``claims``: decide if this handler owns a given file
    - ``extract``: read one file's structure from a source
    - ``build_croissant``: turn that metadata into FileSets + RecordSets

    The generator owns FileObject creation and @id assignment.

    ``claims`` and ``extract`` never see a compression wrapper: both are given
    a source built from the logical name.

    ``build_croissant`` is given both names. ``relative_path`` is logical and
    derives identifiers, so a wrapped file and its plain twin describe one
    table; ``stored_name`` is the file as it sits on disk and belongs in
    descriptions. Use
    :func:`~croissant_baker.handlers.utils.display_name` for the latter.

    ``can_handle(path)`` and ``extract_metadata(path)`` are the previous names
    for the first two. They still work, in both directions, and warn once per
    class.

    Adding a new format: subclass this, implement all three methods, add the
    instance to ``builtin_handlers()`` in registry.py.

    Subclasses set these class attributes for documentation and dispatch:

    - ``EXTENSIONS``: format suffixes this handler claims, e.g. ``(".csv",)``.
      Compression is stripped before a handler is asked, so ``".csv.gz"`` is
      never a valid entry.
    - ``FORMAT_NAME``, ``FORMAT_DESCRIPTION``: for the generated docs table.
    - ``INPUT_KIND``: see :class:`InputKind`. Defaults to ``STREAM``.
    """

    EXTENSIONS: tuple[str, ...] = ()
    FORMAT_NAME: str = ""
    FORMAT_DESCRIPTION: str = ""
    INPUT_KIND: InputKind = InputKind.STREAM

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def claims(self, source: FileSource) -> bool:
        """Whether this handler describes the given file.

        Match on ``source.suffix``, the logical suffix: ``.csv`` for
        ``data.csv`` and ``data.csv.gz`` alike. Use ``source.peek()`` when the
        extension alone is not enough.
        """
        raise NotImplementedError(
            f"{type(self).__name__} implements neither claims(source) nor the "
            "legacy can_handle(path)"
        )

    def extract(self, source: FileSource, **kwargs: object) -> dict:
        """Describe one file's structure.

        Read through ``source.open()`` or ``source.open_text()``, which are
        already decompressed, and take ``file_name``, ``file_size`` and
        ``sha256`` from the source. Report only the format's own media type in
        ``encoding_format``; the generator adds the compression one.

        Thread-safety: may be called concurrently across files on a single
        shared handler instance, so keep per-call state local.

        Returns:
            Extracted metadata. For tabular data this should include
            ``column_types`` mapping column names to Croissant types.
        """
        raise NotImplementedError(
            f"{type(self).__name__} implements neither extract(source) nor the "
            "legacy extract_metadata(path)"
        )

    # ------------------------------------------------------------------
    # Legacy contract — kept working, warned about once per class
    # ------------------------------------------------------------------

    def can_handle(self, file_path: Path) -> bool:
        """Deprecated. Implement and call :meth:`claims` instead."""
        _warn_legacy_call(self, "can_handle", "claims")
        return self.claims(_source_for(self, file_path))

    def extract_metadata(self, file_path: Path, **kwargs: object) -> dict:
        """Deprecated. Implement and call :meth:`extract` instead."""
        _warn_legacy_call(self, "extract_metadata", "extract")
        return self.extract(_source_for(self, file_path), **kwargs)

    @abstractmethod
    def build_croissant(
        self,
        file_metas: list[dict],
        file_ids: list[str],
    ) -> tuple[list, list]:
        """Build Croissant FileSets and RecordSets for every file this handler read.

        Called once per handler, after the FileObject loop.

        Args:
            file_metas: metadata dicts from ``extract``, one per file
            file_ids: FileObject @ids assigned by the generator, aligned by
                position with ``file_metas``

        Returns:
            ``(file_sets, record_sets)``. FileObjects are the generator's.
        """
        pass


def _overrides(handler: FileTypeHandler, method: str) -> bool:
    return getattr(type(handler), method) is not getattr(FileTypeHandler, method)


def uses_legacy_claims(handler: FileTypeHandler) -> bool:
    """Whether ``handler`` decides ownership through ``can_handle(path)``."""
    return not _overrides(handler, "claims") and _overrides(handler, "can_handle")


def uses_legacy_extract(handler: FileTypeHandler) -> bool:
    """Whether ``handler`` reads through ``extract_metadata(path)``."""
    return not _overrides(handler, "extract") and _overrides(
        handler, "extract_metadata"
    )


def warn_legacy_once(handler: FileTypeHandler) -> None:
    """Warn once per class that the registry is routing to a legacy method."""
    cls = type(handler)
    if cls in _warned:
        return
    _warned.add(cls)
    old = []
    if uses_legacy_claims(handler):
        old.append("can_handle(Path)")
    if uses_legacy_extract(handler):
        old.append("extract_metadata(Path)")
    warnings.warn(
        f"{cls.__module__}.{cls.__name__} implements {' / '.join(old)}. "
        "Deprecated: implement claims(FileSource) / extract(FileSource) "
        "instead. The handler still runs, but it is given the raw path, so it "
        "will not describe compressed files correctly.",
        DeprecationWarning,
        stacklevel=3,
    )


def _source_for(handler: FileTypeHandler, file_path: Path) -> FileSource:
    """Build the source a bare path implies, honouring the declared input kind."""
    from croissant_baker.sources import make_source

    return make_source(Path(file_path), with_path=handler.INPUT_KIND is InputKind.PATH)


def _warn_legacy_call(handler: FileTypeHandler, old: str, new: str) -> None:
    """Warn once per class that a caller used a deprecated method."""
    cls = type(handler)
    key = (cls, old)
    if key in _warned_calls:
        return
    _warned_calls.add(key)
    warnings.warn(
        f"{cls.__module__}.{cls.__name__}.{old}(Path) is deprecated; "
        f"call {new}(FileSource) instead. Building a source from the path for "
        "now, which resolves compression the same way the pipeline does.",
        DeprecationWarning,
        stacklevel=3,
    )
