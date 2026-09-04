"""What a completed scan is worth telling someone.

A view over resolved entries, kept apart from the entries themselves: the
terminal wants a fixed-size summary and a machine consumer wants every file,
and neither is the scan stage's business.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from croissant_baker.entries import (
    REASON_LABELS,
    UNRESOLVED_OUTCOMES,
    Outcome,
    Reason,
    ScanEntry,
)


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
