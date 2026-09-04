"""Every file is accounted for, and one failure costs one file."""

import subprocess
import sys
from pathlib import Path

import pytest

from croissant_baker.handlers.base_handler import BuildResult, Declined, FileTypeHandler
from croissant_baker.handlers.csv_handler import CSVHandler
from croissant_baker.handlers.registry import HandlerRegistry
from croissant_baker.metadata_generator import MetadataGenerator
from croissant_baker.scan import Outcome, Reason, ScanReport

from tests.helpers import bake_with, bake_with_report, file_objects, write_wrapped

CSV = b"id,name\n1,Ada\n2,Grace\n"

HEADER = b"record 1 1 10\nrecord.dat 16 200\n"


@pytest.fixture
def refusals(dataset: Path) -> Path:
    """One directory holding every shape of undescribed file, plus two good ones."""
    write_wrapped(dataset, "good.csv", CSV)
    # Nothing claims this extension.
    write_wrapped(dataset, "mystery.xyz", b"?")
    # Claimed by extension, then fails once nibabel reads the header.
    write_wrapped(dataset, "broken.nii", b"not a nifti header" * 4)
    # A multi-member archive, which the baker reports and does not open.
    write_wrapped(dataset, "bundle.zip", b"PK\x03\x04not-really-a-zip")
    # WFDB reads a header together with its siblings, so it needs a real path.
    write_wrapped(dataset, "record.hea", HEADER, ".gz")
    # One file in two forms.
    write_wrapped(dataset, "twin.csv", CSV)
    write_wrapped(dataset, "twin.csv", CSV, ".gz")
    return dataset


@pytest.fixture
def report(refusals: Path) -> ScanReport:
    return bake_with_report(refusals)[1]


def _entry(report: ScanReport, name: str):
    return next(e for e in report.entries if e.name == name)


def test_the_rest_of_the_directory_is_still_described(report: ScanReport) -> None:
    """The loss is per-file, never the whole bake."""
    assert sorted(e.name for e in report.described) == ["good.csv", "twin.csv"]


def test_every_discovered_file_is_accounted_for(report: ScanReport) -> None:
    """PENDING, READY and WOULD_PROCESS are internal; none may reach a report."""
    assert report.total == 7
    assert not report.unresolved
    assert all(e.reason is not None for e in report.undescribed)


@pytest.mark.parametrize(
    ("name", "reason", "in_detail"),
    [
        ("mystery.xyz", Reason.NO_HANDLER, ["no handler"]),
        ("broken.nii", Reason.EXTRACT_FAILED, ["broken.nii"]),
        ("bundle.zip", Reason.ARCHIVE, ["does not open archives"]),
        ("record.hea.gz", Reason.UNSUPPORTED_INPUT, ["WFDB", "on disk", "gzip"]),
        ("twin.csv.gz", Reason.DUPLICATE_BY_NAME, ["twin.csv", "not verified"]),
    ],
)
def test_each_refusal_carries_its_own_reason_and_evidence(
    report: ScanReport, name: str, reason: Reason, in_detail: list
) -> None:
    """Folding any pair of these into one reason loses what a user acts on."""
    entry = _entry(report, name)
    assert entry.reason is reason
    for fragment in in_detail:
        assert fragment in entry.detail, entry.detail


def test_the_machine_readable_report_carries_all_of_them(report: ScanReport) -> None:
    """The report is machine-readable, so a tool can act on it."""
    files = {f["path"]: f for f in report.to_dict()["files"]}

    assert len(files) == 7
    assert all("reason" in files[e.name] for e in report.undescribed)
    assert files["good.csv"]["outcome"] == "described"
    assert files["bundle.zip"]["reason"] == "archive"
    assert files["twin.csv.gz"]["duplicate_of"] == "twin.csv"


def test_a_compressed_archive_is_still_an_archive(dataset: Path) -> None:
    """The wrapper is stripped first, so every compression composes with .tar."""
    write_wrapped(dataset, "good.csv", CSV)
    write_wrapped(dataset, "bundle.tar", b"not-really-a-tar", ".gz")

    _, report = bake_with_report(dataset)

    assert _entry(report, "bundle.tar.gz").reason is Reason.ARCHIVE


@pytest.fixture
def corrupt_wrapper(dataset: Path) -> Path:
    """A good CSV beside a file whose wrapper cannot be decompressed at all."""
    write_wrapped(dataset, "good.csv", CSV)
    write_wrapped(dataset, "bad.parquet.xz", b"not xz, just noise" * 8)
    return dataset


@pytest.mark.parametrize("workers", [1, 4])
def test_a_corrupt_wrapper_does_not_abort_the_bake(
    corrupt_wrapper: Path, workers: int
) -> None:
    """One unreadable file must not cost the whole directory."""
    metadata, report = bake_with_report(corrupt_wrapper, max_workers=workers)

    assert [rs["@id"] for rs in metadata["recordSet"]] == ["good"]
    assert _entry(report, "good.csv").outcome is Outcome.DESCRIBED
    assert _entry(report, "bad.parquet.xz").outcome is Outcome.FAILED


def test_a_claim_failure_is_reported_as_its_own_reason(corrupt_wrapper: Path) -> None:
    """Distinct from an extraction failure: the file never reached a handler."""
    entry = _entry(bake_with_report(corrupt_wrapper)[1], "bad.parquet.xz")

    assert entry.reason is Reason.CLAIM_FAILED
    assert entry.detail
    assert entry.error is not None


class _BrokenCSVHandler(CSVHandler):
    """Extracts happily, then fails to assemble — the gap the report missed."""

    def build_croissant(self, file_metas, file_ids):
        raise RuntimeError("assembly exploded")


@pytest.fixture
def broken_batch(dataset: Path) -> Path:
    write_wrapped(dataset, "a.csv", CSV)
    write_wrapped(dataset, "b.jsonl", b'{"x": 1}\n{"x": 2}\n')
    return dataset


def test_an_assembly_failure_costs_its_batch_and_nothing_else(
    broken_batch: Path,
) -> None:
    """Not described, no orphan FileObject, and the other handler unaffected."""
    metadata, report = bake_with([_BrokenCSVHandler()], broken_batch)

    assert _entry(report, "a.csv").outcome is Outcome.FAILED
    assert _entry(report, "a.csv").reason is Reason.BUILD_FAILED
    # Which files a failing batch took down, and why, with no log to read.
    assert "assembly exploded" in _entry(report, "a.csv").detail
    # A FileObject with no record set is worse than no entry at all.
    assert [d["name"] for d in file_objects(metadata)] == ["b.jsonl"]
    assert [rs["@id"] for rs in metadata["recordSet"]] == ["b.jsonl"]
    assert _entry(report, "b.jsonl").outcome is Outcome.DESCRIBED


def test_every_batch_failing_is_an_error_not_an_empty_document(
    dataset: Path,
) -> None:
    from croissant_baker.handlers.registry import builtin_handlers

    write_wrapped(dataset, "only.csv", CSV)
    generator = MetadataGenerator(
        dataset_path=str(dataset),
        name="t",
        handlers=HandlerRegistry([_BrokenCSVHandler(), *builtin_handlers()]),
    )

    with pytest.raises(ValueError, match="No supported files"):
        generator.generate_metadata()

    assert _entry(generator.scan_report, "only.csv").outcome is Outcome.FAILED


class _BadReturnHandler(FileTypeHandler):
    """Claims ``.brk`` and returns whatever shape the test asks for."""

    EXTENSIONS = (".brk",)
    FORMAT_NAME = "Broken"

    def __init__(self, built) -> None:
        self._built = built

    def claims(self, source) -> bool:
        return source.suffix == ".brk"

    def extract(self, source, **kwargs) -> dict:
        return {
            "file_name": source.name,
            "file_size": source.size,
            "sha256": source.sha256,
            "encoding_format": "application/x-brk",
        }

    def build_croissant(self, file_metas, file_ids):
        return self._built


@pytest.mark.parametrize(
    "built",
    [
        ([], [], "valid"),
        ([], [], [("not-an-index", "nope")]),
        ([], [], [(0,)]),
        "not a tuple at all",
        ([],),
        ([], [], [(99, Reason.EXTRACT_FAILED, "past the end")]),
        ([], [], [(-1, Reason.EXTRACT_FAILED, "counts backwards")]),
        ([], [], [(0, "not-a-reason", "unknown category")]),
        BuildResult([], [], (Declined(99, Reason.EXTRACT_FAILED, "past the end"),)),
    ],
)
def test_a_malformed_handler_return_costs_its_batch_only(built, dataset) -> None:
    """Shape, index and reason are all checked inside the guard. Outside it, a
    handler naming a file it was never given crashes the run, or rejects one it
    did not mean."""
    write_wrapped(dataset, "probe1.brk", b"payload")
    write_wrapped(dataset, "probe2.brk", b"payload")
    write_wrapped(dataset, "keep.csv", CSV)

    metadata, report = bake_with([_BadReturnHandler(built)], dataset)

    for name in ("probe1.brk", "probe2.brk"):
        assert _entry(report, name).outcome is Outcome.FAILED
        assert _entry(report, name).reason is Reason.BUILD_FAILED
    assert [rs["@id"] for rs in metadata["recordSet"]] == ["keep"]


def test_the_library_writes_nothing_to_stderr(tmp_path: Path) -> None:
    """A library that logs to a terminal it does not own is a library bug.

    Out of process, because pytest holds a root handler: in-process,
    ``logging.lastResort`` never fires and the check cannot fail.
    """
    dataset = tmp_path / "quiet"
    dataset.mkdir()
    write_wrapped(dataset, "good.csv", CSV)
    write_wrapped(dataset, "mystery.xyz", b"?")

    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "from croissant_baker.metadata_generator import MetadataGenerator;"
            f"MetadataGenerator(dataset_path={str(dataset)!r}, name='t')"
            ".generate_metadata()",
        ],
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stderr
    assert done.stderr == "", done.stderr
