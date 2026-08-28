"""The scan stage: one typed entry per file found, carrying what became of it.

``discover_files`` answers which paths exist; this module answers what happened
to each of them. Entries are mutated in place by the extraction stage, one
thread per entry, and read back in scan order, so output does not depend on how
many workers ran.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from croissant_baker import compression
from croissant_baker.files import discover_files

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime only
    from croissant_baker.handlers.base_handler import FileTypeHandler


class Outcome(str, Enum):
    """What became of a file the scan found.

    ``str`` mixin so an outcome serialises to its own value in the JSON report.
    ``PENDING``, ``READY`` and ``WOULD_PROCESS`` are working states; a completed
    bake leaves none of them behind.
    """

    #: Not yet resolved. Every entry starts here.
    PENDING = "pending"
    #: A handler claimed the file and extraction succeeded. Its
    #: ``build_croissant`` has still to run.
    READY = "ready"
    #: A handler described the file and its structure was assembled.
    DESCRIBED = "described"
    #: The file is another described file in a different form. Its bytes are
    #: still described; its structure is not.
    LINKED = "linked"
    #: No handler took the file. See :class:`Reason` for which way.
    UNCLAIMED = "unclaimed"
    #: The file was claimed and then lost at claim, extraction or assembly time.
    FAILED = "failed"
    #: ``--dry-run`` only: a handler claimed the file and nothing was read.
    WOULD_PROCESS = "would_process"


UNRESOLVED_OUTCOMES = frozenset({Outcome.PENDING, Outcome.READY, Outcome.WOULD_PROCESS})


class Reason(str, Enum):
    """Why a file was not described, as one of a finite set of categories.

    The entry's ``detail`` names the file and the exception; this is what the
    summary counts, so terminal output stays bounded by the number of *kinds*
    of problem.
    """

    #: Nothing claimed the file.
    NO_HANDLER = "no_handler"
    #: A multi-member archive. The baker reports archives and does not open them.
    ARCHIVE = "archive"
    #: A handler recognised the format but needs an uncompressed file on disk.
    UNSUPPORTED_INPUT = "unsupported_input"
    #: Deciding who owned the file raised — usually a corrupt wrapper.
    CLAIM_FAILED = "claim_failed"
    #: The handler took the file and failed to read it.
    EXTRACT_FAILED = "extract_failed"
    #: The handler read the file and failed to assemble its Croissant nodes.
    BUILD_FAILED = "build_failed"
    #: A plain file and its wrapper share a logical name. Linked on the naming
    #: convention alone; contents were not compared.
    DUPLICATE_BY_NAME = "duplicate_by_name"
    #: Two candidates decompressed to the same bounded prefix.
    PROBABLE_DUPLICATE = "probable_duplicate"


#: One short label per reason, for the fixed-size terminal summary.
REASON_LABELS: Dict[Reason, str] = {
    Reason.NO_HANDLER: "no registered handler",
    Reason.ARCHIVE: "archive, not opened",
    Reason.UNSUPPORTED_INPUT: "handler needs an uncompressed file on disk",
    Reason.CLAIM_FAILED: "unreadable while selecting a handler",
    Reason.EXTRACT_FAILED: "extraction failed",
    Reason.BUILD_FAILED: "could not be assembled",
    Reason.DUPLICATE_BY_NAME: "duplicate by naming convention",
    Reason.PROBABLE_DUPLICATE: "probable duplicate of another file",
}


@dataclass(eq=False)
class ScanEntry:
    """One file the scan found, and what became of it.

    The transition methods enforce the lifecycle rather than overwriting:

    .. code-block:: text

        PENDING -> READY -> DESCRIBED     handler read it, nodes assembled
                         -> LINKED        a duplicate of a described file
                                -> FAILED the file it duplicates was not described
                         -> FAILED        assembly raised
                -> UNCLAIMED              nothing took it
                -> FAILED                 claim or extraction raised
                -> WOULD_PROCESS          --dry-run stops here

    Attributes:
        path: Path relative to the dataset root, wrapper suffix included — the
            file *as stored*. It never crosses into a handler; see
            :mod:`croissant_baker.sources` for the logical view handlers get.
        reason: Which category of problem applied, for every outcome other than
            ``DESCRIBED``.
        detail: The human-readable explanation behind ``reason``.
        meta: The metadata dict the handler produced.
        duplicate_of: The entry this file duplicates. Set for ``LINKED``.

    Compared by identity, so entries can key the generator's staging dicts.
    """

    path: Path
    outcome: Outcome = Outcome.PENDING
    reason: Optional[Reason] = None
    detail: str = ""
    handler: Optional["FileTypeHandler"] = None
    meta: Optional[dict] = None
    error: Optional[BaseException] = None
    duplicate_of: Optional["ScanEntry"] = None

    @property
    def name(self) -> str:
        """The file's basename, wrapper suffix included."""
        return self.path.name

    def _move(self, to: Outcome, *allowed_from: Outcome) -> None:
        if self.outcome not in allowed_from:
            raise ValueError(
                f"{self.path}: cannot move from {self.outcome.value} to "
                f"{to.value}; expected one of "
                f"{', '.join(o.value for o in allowed_from)}"
            )
        self.outcome = to

    def ready(self, handler: "FileTypeHandler", meta: dict) -> None:
        """Record that ``handler`` read this file. Its nodes are not built yet."""
        self._move(Outcome.READY, Outcome.PENDING)
        self.handler = handler
        self.meta = meta

    def describe(self) -> None:
        """Record that this file's Croissant nodes were assembled."""
        self._move(Outcome.DESCRIBED, Outcome.READY)
        self.reason = None
        self.detail = ""

    def unclaimed(self, reason: Reason, detail: str) -> None:
        """Record that no handler took this file."""
        self._move(Outcome.UNCLAIMED, Outcome.PENDING)
        self.reason = reason
        self.detail = detail

    def would_process(self, handler: "FileTypeHandler") -> None:
        """Record that a handler claimed this file, without reading it."""
        self._move(Outcome.WOULD_PROCESS, Outcome.PENDING)
        self.handler = handler

    def failed(self, reason: Reason, error: BaseException) -> None:
        """Record that this file was lost, at whichever stage ``reason`` names.

        A linked duplicate can be lost too: its structure was its primary's, so
        it goes when the primary's does.
        """
        self._move(Outcome.FAILED, Outcome.PENDING, Outcome.READY, Outcome.LINKED)
        self.reason = reason
        self.detail = str(error) or type(error).__name__
        self.error = error
        self.meta = None

    def linked(self, primary: "ScanEntry", reason: Reason, detail: str) -> None:
        """Record that this file duplicates ``primary``.

        The entry keeps its own distribution entry — its bytes, size and
        checksum are its own — and gives up only its structure.
        """
        self._move(Outcome.LINKED, Outcome.READY)
        self.duplicate_of = primary
        self.reason = reason
        self.detail = detail


def scan_directory(
    dir_path: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[ScanEntry]:
    """One unresolved entry per file found, in discovery order.

    Discovery semantics are :func:`croissant_baker.files.discover_files`'s and
    are unchanged here.

    Raises:
        FileNotFoundError: If the directory does not exist or is not a directory.
        PermissionError: If the directory cannot be accessed.
    """
    return [
        ScanEntry(path=path)
        for path in discover_files(
            dir_path,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
    ]


#: How much decompressed data a duplicate check reads from each candidate.
PREFIX_BYTES = 64 * 1024


def resolve_duplicates(entries: List[ScanEntry], root: Path) -> None:
    """Collapse files that describe the same data onto a single description.

    Two files with one logical path would emit two record sets under one @id,
    which is not a Croissant document. The shapes and the evidence each needs
    are documented in ``docs/user-guide/supported-formats.md``.

    Args:
        entries: Scan entries after extraction. Only ``READY`` ones take part,
            so a file whose duplicate failed to extract is still described.
            Mutated in place.
        root: The dataset directory, for reading the prefixes.
    """
    groups: Dict[tuple, List[ScanEntry]] = {}
    for entry in entries:
        if entry.outcome is not Outcome.READY:
            continue
        logical = compression.logical_name(entry.name)
        # Directory plus stem, so every candidate lands in one group and is
        # decided against a single primary.
        key = (str(entry.path.parent), Path(logical).stem)
        groups.setdefault(key, []).append(entry)

    for members in groups.values():
        if len(members) > 1:
            _resolve_group(members, root)


def _resolve_group(members: List[ScanEntry], root: Path) -> None:
    """Link whichever members of one candidate group turn out to duplicate."""
    primary = choose_primary(members)
    for entry in members:
        if entry is primary:
            continue
        verdict = _compare(primary, entry, root)
        if verdict is not None:
            entry.linked(primary, *verdict)


def _compare(primary: ScanEntry, other: ScanEntry, root: Path) -> Optional[tuple]:
    """Decide whether ``other`` duplicates ``primary``, and on what evidence.

    Compressed sizes are never evidence: gzip and xz sizes are not comparable,
    and two files that differ can compress to the same length.
    """
    primary_logical = compression.logical_name(primary.name)
    other_logical = compression.logical_name(other.name)

    if primary_logical == other_logical:
        # One of the pair is the plain file the other wraps. Linked on the
        # naming convention alone: verifying it would cost a full decompression
        # pass, and the reason string says the content was not compared.
        if compression.is_compressed(primary.name) != compression.is_compressed(
            other.name
        ):
            return (
                Reason.DUPLICATE_BY_NAME,
                f"same logical name as {primary.path}; linked by naming "
                "convention, content not verified",
            )
    elif not compression.is_compressed(primary.name) and not compression.is_compressed(
        other.name
    ):
        # Two plain files of different size cannot match.
        if _stored_size(root / primary.path) != _stored_size(root / other.path):
            return None

    left = _read_prefix(root / primary.path)
    right = _read_prefix(root / other.path)
    if left is None or right is None or left != right:
        return None
    return (
        Reason.PROBABLE_DUPLICATE,
        f"first {len(left)} decompressed bytes are identical to "
        f"{primary.path}; linked as a probable duplicate",
    )


def _stored_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _read_prefix(path: Path) -> Optional[bytes]:
    """Up to :data:`PREFIX_BYTES` decompressed bytes, or ``None`` if unreadable."""
    try:
        with compression.open_binary(path) as stream:
            return stream.read(PREFIX_BYTES)
    except (OSError, EOFError, ValueError):
        return None


def choose_primary(members: Iterable[ScanEntry]) -> ScanEntry:
    """Pick the member of a duplicate group that keeps its structure.

    Uncompressed first, then registration order of the compression, then the
    stored path lexically, so the choice does not depend on ``rglob`` order.
    """
    order = {c.suffix: i for i, c in enumerate(compression.compressions())}

    def rank(entry: ScanEntry) -> tuple:
        wrapper = compression.compression_for(entry.name)
        return (
            wrapper is not None,
            order.get(wrapper.suffix, len(order)) if wrapper else -1,
            str(entry.path),
        )

    return min(members, key=rank)


@dataclass(frozen=True)
class ScanReport:
    """A view over resolved scan entries: what was described, and what was not.

    :meth:`summary_lines` is for the terminal, and its length depends on how
    many *kinds* of problem occurred, never on how many files did.
    :meth:`to_dict` carries the per-file detail, for ``--report`` and for
    downstream tools checking coverage.
    """

    entries: List[ScanEntry] = field(default_factory=list)

    @property
    def total(self) -> int:
        """How many files the scan found."""
        return len(self.entries)

    @property
    def described(self) -> List[ScanEntry]:
        """Entries a handler described and whose nodes were assembled."""
        return [e for e in self.entries if e.outcome is Outcome.DESCRIBED]

    @property
    def undescribed(self) -> List[ScanEntry]:
        """Entries that produced no record set, in scan order."""
        return [e for e in self.entries if e.outcome is not Outcome.DESCRIBED]

    @property
    def unresolved(self) -> List[ScanEntry]:
        """Entries still in a working state. Empty after a completed bake."""
        return [e for e in self.entries if e.outcome in UNRESOLVED_OUTCOMES]

    def counts(self) -> Dict[Reason, int]:
        """Number of undescribed entries per reason, in declaration order."""
        tally = Counter(e.reason for e in self.entries if e.reason is not None)
        return {r: tally[r] for r in Reason if tally[r]}

    def summary_lines(self) -> List[str]:
        """A header plus at most one line per reason."""
        if not self.entries:
            return ["Scanned 0 files."]

        n_described = len(self.described)
        lines = [
            f"Scanned {self.total} file(s): "
            f"{n_described} described, {self.total - n_described} not described."
        ]
        lines.extend(
            f"  {REASON_LABELS[reason]}: {count}"
            for reason, count in self.counts().items()
        )
        return lines

    def to_dict(self) -> dict:
        """Every discovered file with its outcome.

        ``reason`` is one of a finite set a caller can branch on; ``detail`` is
        the sentence for a human.
        """
        return {
            "total": self.total,
            "described": len(self.described),
            "undescribed": len(self.undescribed),
            "by_reason": {r.value: n for r, n in self.counts().items()},
            "files": [
                {
                    "path": str(e.path),
                    "outcome": e.outcome.value,
                    **({"reason": e.reason.value} if e.reason else {}),
                    **({"detail": e.detail} if e.detail else {}),
                    **(
                        {"duplicate_of": str(e.duplicate_of.path)}
                        if e.duplicate_of
                        else {}
                    ),
                }
                for e in self.entries
            ],
        }
