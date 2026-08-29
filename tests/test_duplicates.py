"""Two files that want one @id, resolved on evidence."""

from pathlib import Path

import pytest

from croissant_baker.metadata_generator import MetadataGenerator
from croissant_baker.scan import PREFIX_BYTES, Outcome, Reason

from tests.helpers import (
    WRAPPER_SUFFIXES,
    bake,
    bake_with_report,
    by_name,
    file_objects,
    write_wrapped,
)

CSV = b"id,name\nid1,Ada\nid2,Grace\n"
OTHER_CSV = b"other,columns,entirely\n9,8,7\n"
TSV = b"id\tname\nid1\tAda\nid2\tGrace\n"
FHIR = (
    b'{"resourceType": "Patient", "id": "a", "name": [{"family": "L", "use": "x"}]}\n'
    b'{"resourceType": "Patient", "id": "b", "name": [{"family": "H", "use": "y"}]}\n'
)


def _entries(report) -> dict:
    return {e.name: e for e in report.entries}


def _ids(metadata: dict) -> list:
    return sorted(rs["@id"] for rs in metadata.get("recordSet", []))


@pytest.fixture
def twins(dataset: Path) -> Path:
    """``sample.csv`` and ``sample.csv.gz``: one file in two forms."""
    write_wrapped(dataset, "sample.csv", CSV)
    write_wrapped(dataset, "sample.csv", CSV, ".gz")
    return dataset


@pytest.fixture
def same_stem(dataset: Path) -> Path:
    """``sample.csv`` and ``sample.tsv``: two files, one stem."""
    write_wrapped(dataset, "sample.csv", CSV)
    write_wrapped(dataset, "sample.tsv", TSV)
    return dataset


# --------------------------------------------------------------------------
# Rule 1 — plain beside its wrapper
# --------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", WRAPPER_SUFFIXES)
def test_a_plain_file_and_its_wrapper_link_by_convention(
    suffix: str, dataset: Path
) -> None:
    write_wrapped(dataset, "sample.csv", CSV)
    write_wrapped(dataset, "sample.csv", CSV, suffix)

    metadata, report = bake_with_report(dataset)
    entries = _entries(report)

    assert entries["sample.csv"].outcome is Outcome.DESCRIBED
    assert entries[f"sample.csv{suffix}"].outcome is Outcome.LINKED
    assert entries[f"sample.csv{suffix}"].reason is Reason.DUPLICATE_BY_NAME
    assert _ids(metadata) == ["sample"]


def test_rule_one_links_without_reading_and_says_so(dataset: Path) -> None:
    """Different payloads still link — and the reason must not claim they match."""
    write_wrapped(dataset, "sample.csv", CSV)
    write_wrapped(dataset, "sample.csv", OTHER_CSV, ".gz")

    linked = _entries(bake_with_report(dataset)[1])["sample.csv.gz"]

    assert linked.reason is Reason.DUPLICATE_BY_NAME
    assert "content not verified" in linked.detail
    assert "same content" not in linked.detail


def test_the_uncompressed_member_wins_however_the_scan_ordered_them(
    dataset: Path,
) -> None:
    """Discovery order is the filesystem's, so the choice cannot depend on it."""
    write_wrapped(dataset, "sample.csv", CSV, ".gz")
    write_wrapped(dataset, "sample.csv", CSV)

    assert _entries(bake_with_report(dataset)[1])["sample.csv"].outcome is (
        Outcome.DESCRIBED
    )


# --------------------------------------------------------------------------
# Rule 2 — two wrappers, no plain file
# --------------------------------------------------------------------------


def test_two_wrappers_with_the_same_bytes_link(dataset: Path) -> None:
    write_wrapped(dataset, "sample.csv", CSV, ".gz")
    write_wrapped(dataset, "sample.csv", CSV, ".xz")

    metadata, report = bake_with_report(dataset)
    entries = _entries(report)

    assert entries["sample.csv.gz"].outcome is Outcome.DESCRIBED
    assert entries["sample.csv.xz"].outcome is Outcome.LINKED
    assert entries["sample.csv.xz"].reason is Reason.PROBABLE_DUPLICATE
    assert "probable duplicate" in entries["sample.csv.xz"].detail
    assert _ids(metadata) == ["sample"]
    # Both keep a distribution entry; only one carries the link.
    assert len(file_objects(metadata)) == 2
    assert len([d for d in file_objects(metadata) if "sameAs" in d]) == 1


def test_two_wrappers_with_different_bytes_stay_distinct(dataset: Path) -> None:
    write_wrapped(dataset, "sample.csv", CSV, ".gz")
    write_wrapped(dataset, "sample.csv", OTHER_CSV, ".xz")

    metadata, report = bake_with_report(dataset)

    assert all(e.outcome is Outcome.DESCRIBED for e in report.entries)
    assert len(_ids(metadata)) == 2


def test_the_primary_of_two_wrappers_follows_the_registry_not_the_filesystem(
    dataset: Path,
) -> None:
    """gzip is registered first, so it wins however the directory is read."""
    write_wrapped(dataset, "sample.csv", CSV, ".xz")
    write_wrapped(dataset, "sample.csv", CSV, ".gz")

    assert _entries(bake_with_report(dataset)[1])["sample.csv.gz"].outcome is (
        Outcome.DESCRIBED
    )


def test_compressed_size_is_never_taken_as_evidence(dataset: Path) -> None:
    """Two payloads that compress to the same length are still different files."""
    write_wrapped(dataset, "sample.csv", b"id,name\n" + b"1,aaaa\n" * 500, ".gz")
    write_wrapped(dataset, "sample.csv", b"id,name\n" + b"1,bbbb\n" * 500, ".xz")

    assert all(
        e.outcome is Outcome.DESCRIBED for e in bake_with_report(dataset)[1].entries
    )


# --------------------------------------------------------------------------
# Rule 3 — one stem, two format suffixes
# --------------------------------------------------------------------------


def test_a_csv_and_a_tsv_of_the_same_data_stay_distinct(same_stem: Path) -> None:
    """Same rows, different serialisation: two files, and two record sets."""
    metadata, report = bake_with_report(same_stem)

    assert all(e.outcome is Outcome.DESCRIBED for e in report.entries)
    assert _ids(metadata) == ["sample_csv", "sample_tsv"]


def test_byte_identical_files_under_two_suffixes_link(dataset: Path) -> None:
    write_wrapped(dataset, "sample.csv", CSV)
    write_wrapped(dataset, "sample.tsv", CSV)

    entries = _entries(bake_with_report(dataset)[1])

    assert entries["sample.csv"].outcome is Outcome.DESCRIBED
    assert entries["sample.tsv"].outcome is Outcome.LINKED
    assert entries["sample.tsv"].reason is Reason.PROBABLE_DUPLICATE


def test_two_plain_files_of_different_size_are_rejected_without_reading(
    dataset: Path, monkeypatch
) -> None:
    """The free half of rule 3: stat() settles it, so nothing is decompressed."""
    from croissant_baker import duplicates

    write_wrapped(dataset, "sample.csv", CSV)
    write_wrapped(dataset, "sample.tsv", CSV + b"id3,Alan\n")

    reads = []
    real = duplicates._read_prefix
    monkeypatch.setattr(
        duplicates, "_read_prefix", lambda p: reads.append(p) or real(p)
    )

    assert all(
        e.outcome is Outcome.DESCRIBED for e in bake_with_report(dataset)[1].entries
    )
    assert not reads


# --------------------------------------------------------------------------
# Scope and bounds
# --------------------------------------------------------------------------


def test_files_in_different_directories_never_link(dataset: Path) -> None:
    """The logical *path* has to match, not just the basename."""
    for sub in ("one", "two"):
        (dataset / sub).mkdir()
        write_wrapped(dataset / sub, "sample.csv", CSV)

    metadata, report = bake_with_report(dataset)

    assert all(e.outcome is Outcome.DESCRIBED for e in report.entries)
    assert not [d for d in file_objects(metadata) if "sameAs" in d]


def test_a_prefix_read_is_capped(tmp_path: Path) -> None:
    """A duplicate check must not turn into reading a terabyte."""
    from croissant_baker.duplicates import _read_prefix

    big = tmp_path / "big.csv"
    big.write_bytes(b"x" * (PREFIX_BYTES * 3))

    assert len(_read_prefix(big)) == PREFIX_BYTES


def test_an_unreadable_candidate_is_not_a_duplicate(tmp_path: Path) -> None:
    from croissant_baker.duplicates import _read_prefix

    assert _read_prefix(tmp_path / "absent.csv") is None


def test_a_three_member_group_points_at_one_primary(dataset: Path) -> None:
    """One equivalence class, not a chain of pairwise links."""
    write_wrapped(dataset, "sample.csv", CSV)
    write_wrapped(dataset, "sample.csv", CSV, ".gz")
    write_wrapped(dataset, "sample.csv", CSV, ".xz")

    metadata, report = bake_with_report(dataset)
    entries = _entries(report)

    assert entries["sample.csv"].outcome is Outcome.DESCRIBED
    for name in ("sample.csv.gz", "sample.csv.xz"):
        assert entries[name].duplicate_of is entries["sample.csv"]

    same_as = [d["sameAs"] for d in file_objects(metadata) if "sameAs" in d]
    assert len(same_as) == 2
    assert len(set(map(str, same_as))) == 1


def test_a_duplicate_of_a_file_that_failed_to_parse_is_still_described(
    dataset: Path,
) -> None:
    """Resolution runs after extraction, so a broken primary is not chosen."""
    write_wrapped(dataset, "sample.csv", b"")  # empty: extraction fails
    write_wrapped(dataset, "sample.csv", CSV, ".gz")

    metadata, report = bake_with_report(dataset)
    entries = _entries(report)

    assert entries["sample.csv"].outcome is Outcome.FAILED
    assert entries["sample.csv.gz"].outcome is Outcome.DESCRIBED
    assert _ids(metadata) == ["sample"]


def test_a_twin_does_not_cost_the_rest_of_the_directory(twins: Path) -> None:
    write_wrapped(twins, "other.csv", CSV)

    assert _ids(bake(twins)) == ["other", "sample"]


# --------------------------------------------------------------------------
# What the linked pair looks like in the document
# --------------------------------------------------------------------------


def test_both_twins_keep_their_own_bytes_and_the_wrapper_points_at_the_plain_file(
    twins: Path,
) -> None:
    """The files differ on disk, so checksums and sizes must differ too."""
    objects = by_name(file_objects(metadata := bake(twins)))
    plain, wrapped = objects["sample.csv"], objects["sample.csv.gz"]

    assert len(file_objects(metadata)) == 2
    assert wrapped["sameAs"] == plain["@id"]
    assert "sameAs" not in plain
    assert plain["sha256"] != wrapped["sha256"]
    assert plain["contentSize"] != wrapped["contentSize"]
    assert wrapped["encodingFormat"] == ["text/csv", "application/gzip"]


# --------------------------------------------------------------------------
# What the disambiguated pair looks like in the document
# --------------------------------------------------------------------------


def test_both_members_are_suffixed_not_just_one(same_stem: Path) -> None:
    """Suffixing only the loser would make the result depend on handler order."""
    assert _ids(bake(same_stem)) == ["sample_csv", "sample_tsv"]


def test_fields_follow_their_record_set_and_keep_their_column_names(
    same_stem: Path,
) -> None:
    """Only the identifier prefix moves — a reader matches on the name."""
    for record_set in bake(same_stem)["recordSet"]:
        assert [f["name"] for f in record_set["field"]] == ["id", "name"]
        for field in record_set["field"]:
            assert field["@id"].startswith(f"{record_set['@id']}/")


def test_sub_fields_follow_their_record_set(dataset: Path) -> None:
    """Nested identifiers carry the record set's own as a prefix at every depth."""
    write_wrapped(dataset, "sample.csv", CSV)
    write_wrapped(dataset, "sample.ndjson", FHIR)

    fhir = next(rs for rs in bake(dataset)["recordSet"] if rs["@id"] == "sample_fhir")
    name = next(f for f in fhir["field"] if f["name"] == "name")

    assert [sf["@id"] for sf in name["subField"]] == [
        "sample_fhir/name/family",
        "sample_fhir/name/use",
    ]


def test_a_suffixed_identifier_that_is_already_taken_falls_back(
    same_stem: Path,
) -> None:
    """A file genuinely named sample_csv must not be overwritten by the fix."""
    write_wrapped(same_stem, "sample_csv.csv", CSV)

    assert _ids(bake(same_stem)) == ["sample_csv", "sample_csv__2", "sample_tsv"]


def test_record_sets_that_do_not_collide_keep_their_bare_identifiers(
    dataset: Path,
) -> None:
    """The rename fires on collision only, so ordinary output does not churn."""
    write_wrapped(dataset, "one.csv", CSV)
    write_wrapped(dataset, "two.tsv", TSV)

    assert _ids(bake(dataset)) == ["one", "two"]


def test_the_disambiguated_document_still_validates(
    same_stem: Path, tmp_path_factory
) -> None:
    """mlcroissant is the arbiter of whether the rewritten identifiers are legal."""
    out = tmp_path_factory.mktemp("out") / "croissant.jsonld"
    MetadataGenerator(dataset_path=str(same_stem), name="collisions").save_metadata(
        str(out), validate=True
    )
    assert out.exists()
