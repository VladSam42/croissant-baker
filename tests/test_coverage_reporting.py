"""Every file is accounted for, and one failure costs one file."""

import logging
from pathlib import Path

import pytest

from croissant_baker.handlers.csv_handler import CSVHandler
from croissant_baker.handlers.registry import HandlerRegistry
from croissant_baker.handlers.tsv_handler import TSVHandler
from croissant_baker.metadata_generator import (
    MAX_UNDESCRIBED_WARNINGS,
    MetadataGenerator,
)
from croissant_baker.scan import Outcome, Reason, ScanEntry, ScanReport

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


# --------------------------------------------------------------------------
# Every file is named, with the reason it was not described
# --------------------------------------------------------------------------


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
        ("mystery.xyz", Reason.NO_HANDLER, ["no registered handler"]),
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


def test_the_five_reasons_stay_distinct(report: ScanReport) -> None:
    assert len(report.undescribed) == 5
    assert len({e.reason for e in report.undescribed}) == 5


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


# --------------------------------------------------------------------------
# Failing at claim time
# --------------------------------------------------------------------------


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
    """The whole directory used to be lost to one unreadable file."""
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


# --------------------------------------------------------------------------
# Failing at assembly time
# --------------------------------------------------------------------------


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
    # A FileObject with no record set is worse than no entry at all.
    assert [d["name"] for d in file_objects(metadata)] == ["b.jsonl"]
    assert [rs["@id"] for rs in metadata["recordSet"]] == ["b.jsonl"]
    assert _entry(report, "b.jsonl").outcome is Outcome.DESCRIBED


def test_every_batch_failing_is_an_error_not_an_empty_document(
    dataset: Path,
) -> None:
    from croissant_baker.handlers.registry import HandlerRegistry, builtin_handlers

    write_wrapped(dataset, "only.csv", CSV)
    generator = MetadataGenerator(
        dataset_path=str(dataset),
        name="t",
        handlers=HandlerRegistry([_BrokenCSVHandler(), *builtin_handlers()]),
    )

    with pytest.raises(ValueError, match="No supported files"):
        generator.generate_metadata()

    assert _entry(generator.scan_report, "only.csv").outcome is Outcome.FAILED


# --------------------------------------------------------------------------
# The lifecycle refuses illegal moves
# --------------------------------------------------------------------------


def test_an_entry_cannot_be_resolved_twice() -> None:
    entry = ScanEntry(path=Path("a.csv"))
    entry.unclaimed(Reason.NO_HANDLER, "nothing claims it")

    with pytest.raises(ValueError, match="unclaimed"):
        entry.ready(handler=object(), meta={})


def test_an_entry_cannot_be_described_before_it_is_ready() -> None:
    entry = ScanEntry(path=Path("a.csv"))

    with pytest.raises(ValueError, match="pending"):
        entry.describe()


def test_failing_clears_the_metadata_the_entry_was_carrying() -> None:
    """A FAILED entry must not still look like it has something to contribute."""
    entry = ScanEntry(path=Path("a.csv"))
    entry.ready(handler=object(), meta={"file_name": "a.csv"})
    entry.failed(Reason.BUILD_FAILED, RuntimeError("boom"))

    assert entry.outcome is Outcome.FAILED
    assert entry.meta is None


def test_an_undescribed_file_is_warned_about_as_it_is_scanned(tmp_path, caplog):
    """The summary arrives at the end; a long bake needs to say so as it goes."""
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "notes.md").write_text("free text")

    with caplog.at_level(logging.WARNING, logger="croissant_baker"):
        MetadataGenerator(dataset_path=str(tmp_path)).generate_metadata()

    assert any(
        "notes.md" in r.getMessage() and "no registered handler" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_a_file_that_fails_to_parse_names_itself(tmp_path, caplog):
    """The message that went missing in the refactor: which file, and why."""
    (tmp_path / "ok.csv").write_text("a,b\n1,2\n")
    (tmp_path / "broken.csv").write_text("# preamble\na,b,c,d,e\n1,2,3,4,5\n")

    with caplog.at_level(logging.WARNING, logger="croissant_baker"):
        MetadataGenerator(dataset_path=str(tmp_path)).generate_metadata()

    messages = [r.getMessage() for r in caplog.records]
    assert any("broken.csv" in m and "CSV parse error" in m for m in messages), messages


def test_per_file_warnings_stay_bounded(tmp_path, caplog):
    """The summary already counts these; a big directory must not bury the run."""
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    for i in range(MAX_UNDESCRIBED_WARNINGS + 20):
        (tmp_path / f"notes{i}.md").write_text("free text")

    with caplog.at_level(logging.WARNING, logger="croissant_baker"):
        MetadataGenerator(dataset_path=str(tmp_path)).generate_metadata()

    named = [r for r in caplog.records if ".md" in r.getMessage()]
    assert len(named) == MAX_UNDESCRIBED_WARNINGS
    assert "further per-file warnings suppressed" in named[-1].getMessage()


def test_a_batch_that_fails_to_build_names_its_files(tmp_path, caplog):
    """A handler failing as a whole must still say which files it took down."""

    class Exploding(CSVHandler):
        def build_croissant(self, file_metas, file_ids):
            raise RuntimeError("boom")

    (tmp_path / "one.csv").write_text("a,b\n1,2\n")
    (tmp_path / "two.tsv").write_text("a\tb\n1\t2\n")

    with caplog.at_level(logging.WARNING, logger="croissant_baker"):
        MetadataGenerator(
            dataset_path=str(tmp_path),
            handlers=HandlerRegistry([Exploding(), TSVHandler()]),
        ).generate_metadata()

    messages = [r.getMessage() for r in caplog.records]
    assert any("one.csv" in m and "boom" in m for m in messages), messages


def test_a_malformed_handler_return_costs_only_its_batch(tmp_path, caplog):
    """Unpacking under the guard: the other handlers' files are still described."""

    class Wrong(CSVHandler):
        def build_croissant(self, file_metas, file_ids):
            return "not", "a", "valid", "return"

    (tmp_path / "one.csv").write_text("a,b\n1,2\n")
    (tmp_path / "two.tsv").write_text("a\tb\n1\t2\n")

    with caplog.at_level(logging.WARNING, logger="croissant_baker"):
        result = MetadataGenerator(
            dataset_path=str(tmp_path),
            handlers=HandlerRegistry([Wrong(), TSVHandler()]),
        ).generate_metadata()

    assert [rs["name"] for rs in result["recordSet"]] == ["two"]
    assert any("one.csv" in r.getMessage() for r in caplog.records)
