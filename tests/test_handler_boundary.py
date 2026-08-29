"""A handler never observes a compression wrapper, watched from the outside."""

import gzip
import json
from pathlib import Path

import pytest

from croissant_baker.handlers.base_handler import FileTypeHandler
from croissant_baker.handlers.registry import HandlerRegistry, builtin_handlers
from croissant_baker.metadata_generator import MetadataGenerator

from tests.helpers import (
    PNG_1X1,
    SAMPLES,
    WRAPPER_SUFFIXES,
    bake,
    file_sets,
    includes,
    write_wrapped,
)


class SpyHandler(FileTypeHandler):
    """Claims ``.spy`` files and records every string it is shown."""

    EXTENSIONS = (".spy",)
    FORMAT_NAME = "Spy"
    FORMAT_DESCRIPTION = "records what the pipeline hands it"

    def __init__(self) -> None:
        self.seen: dict = {"claims": [], "extract": [], "build_croissant": []}

    def claims(self, source) -> bool:
        self.seen["claims"] += [source.name, str(source.relative_path)]
        return source.suffix == ".spy"

    def extract(self, source, **kwargs) -> dict:
        self.seen["extract"] += [source.name, str(source.relative_path)]
        return {
            "file_name": source.name,
            "file_size": source.size,
            "sha256": source.sha256,
            "encoding_format": "application/x-spy",
        }

    def build_croissant(self, file_metas, file_ids):
        for meta in file_metas:
            self.seen["build_croissant"] += [
                str(v) for v in meta.values() if isinstance(v, str)
            ]
        return [], []


def _bake_spy(tmp_path: Path, suffix: str) -> SpyHandler:
    nested = tmp_path / "sub"
    nested.mkdir()
    write_wrapped(nested, "probe.spy", b"payload", suffix)

    spy = SpyHandler()
    MetadataGenerator(
        dataset_path=str(tmp_path),
        name="spy",
        handlers=HandlerRegistry([spy, *builtin_handlers()]),
    ).generate_metadata()
    return spy


@pytest.mark.parametrize("suffix", WRAPPER_SUFFIXES)
def test_a_handler_never_decides_or_reads_through_a_wrapper(
    suffix: str, tmp_path: Path
) -> None:
    """Nothing a handler *acts on* names a wrapper."""
    spy = _bake_spy(tmp_path, suffix)

    for phase in ("claims", "extract"):
        strings = spy.seen[phase]
        assert strings, f"{phase} was never called"
        offenders = [s for s in strings if any(w in s for w in WRAPPER_SUFFIXES)]
        assert not offenders, f"{phase} was shown a wrapper: {offenders}"


@pytest.mark.parametrize("suffix", WRAPPER_SUFFIXES)
def test_build_croissant_is_given_both_names(suffix: str, tmp_path: Path) -> None:
    """Assembly gets the logical path *and* the stored name, for two jobs."""
    spy = _bake_spy(tmp_path, suffix)
    seen = spy.seen["build_croissant"]

    assert "sub/probe.spy" in seen, seen
    assert f"probe.spy{suffix}" in seen, seen


def test_the_handler_still_gets_its_directory(tmp_path: Path) -> None:
    """Logical does not mean flattened — only the wrapper suffix comes off."""
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    with gzip.open(nested / "probe.spy.gz", "wb") as fh:
        fh.write(b"payload")

    spy = SpyHandler()
    MetadataGenerator(
        dataset_path=str(tmp_path),
        name="spy",
        handlers=HandlerRegistry([spy, *builtin_handlers()]),
    ).generate_metadata()

    assert "a/b/probe.spy" in spy.seen["build_croissant"]


# --------------------------------------------------------------------------
# What the generator does with the logical names it gets back
# --------------------------------------------------------------------------


@pytest.fixture
def fhir_chunks(tmp_path: Path) -> Path:
    """The shape that produced the phantom includes: several wrapped chunks."""
    chunk = (
        b'{"resourceType": "Patient", "id": "a", "gender": "female"}\n'
        b'{"resourceType": "Patient", "id": "b", "gender": "male"}\n'
    )
    for i in range(3):
        with gzip.open(tmp_path / f"Patient.{i:03d}.ndjson.gz", "wb") as fh:
            fh.write(chunk)
    return tmp_path


def test_an_exact_include_names_the_file_that_exists(fhir_chunks: Path) -> None:
    metadata = MetadataGenerator(
        dataset_path=str(fhir_chunks), name="fhir"
    ).generate_metadata()

    resolved = includes(file_sets(metadata)[0])
    assert sorted(resolved) == [
        "Patient.000.ndjson.gz",
        "Patient.001.ndjson.gz",
        "Patient.002.ndjson.gz",
    ]


def test_no_include_is_wrapped_twice(fhir_chunks: Path) -> None:
    """The regression: an exact path expanded as though it were a pattern."""
    metadata = MetadataGenerator(
        dataset_path=str(fhir_chunks), name="fhir"
    ).generate_metadata()

    for file_set in file_sets(metadata):
        for pattern in includes(file_set):
            assert ".gz.gz" not in pattern
            assert ".bz2.bz2" not in pattern
            assert ".xz.xz" not in pattern


def test_every_include_matches_a_file_in_the_dataset(fhir_chunks: Path) -> None:
    """A pattern matching nothing is a claim about data that is not there."""
    metadata = MetadataGenerator(
        dataset_path=str(fhir_chunks), name="fhir"
    ).generate_metadata()

    on_disk = {p.name for p in fhir_chunks.iterdir()}
    for file_set in file_sets(metadata):
        for pattern in includes(file_set):
            matched = [n for n in on_disk if Path(n).match(pattern)]
            assert matched, f"{pattern!r} matches nothing in the dataset"


def test_every_compressed_member_is_covered_by_some_include(
    fhir_chunks: Path,
) -> None:
    """The other direction: a described file must be in its own FileSet."""
    metadata = MetadataGenerator(
        dataset_path=str(fhir_chunks), name="fhir"
    ).generate_metadata()

    patterns = [p for fs in file_sets(metadata) for p in includes(fs)]
    for stored in fhir_chunks.iterdir():
        assert any(Path(stored.name).match(p) for p in patterns), stored.name


def test_a_glob_gains_one_variant_per_wrapper_present(dataset: Path) -> None:
    """And in registry order, so the document does not depend on set hashing."""
    for stem, suffix in (("a.png", ".gz"), ("b.png", ".bz2"), ("c.png", ".xz")):
        write_wrapped(dataset, stem, PNG_1X1, suffix)

    assert includes(file_sets(bake(dataset))[0]) == [
        "**/*.png",
        "**/*.png.gz",
        "**/*.png.bz2",
        "**/*.png.xz",
    ]


def test_a_dataset_of_plain_files_keeps_its_bare_globs(dataset: Path) -> None:
    write_wrapped(dataset, "pixel.png", PNG_1X1)

    assert includes(file_sets(bake(dataset))[0]) == ["**/*.png"]


def test_a_glob_stays_scoped_to_the_directory_it_describes(dataset: Path) -> None:
    """Parquet shards group per directory, so the glob is not rooted at ``**/``."""
    ((_, payload),) = SAMPLES["ParquetHandler"]()
    part = dataset / "part"
    part.mkdir()
    write_wrapped(part, "part-00000.parquet", payload)
    write_wrapped(part, "part-00001.parquet", payload, ".gz")

    assert includes(file_sets(bake(dataset))[0]) == [
        "part/*.parquet",
        "part/*.parquet.gz",
    ]


def test_output_is_identical_across_hash_seeds(tmp_path: Path) -> None:
    """A set of wrappers used to reach the document, and sets are not ordered."""
    import subprocess
    import sys

    import bz2
    import lzma

    payload = b"id,name\n1,Ada\n"
    for name, opener in (
        ("a.csv.gz", gzip.open),
        ("b.csv.bz2", bz2.open),
        ("c.csv.xz", lzma.open),
    ):
        with opener(tmp_path / name, "wb") as fh:
            fh.write(payload)
    (tmp_path / "pic.png").write_bytes(
        __import__("base64").b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEh"
            b"QGAhKmMIQAAAABJRU5ErkJggg=="
        )
    )

    script = (
        "import json;"
        "from croissant_baker.metadata_generator import MetadataGenerator;"
        f"m=MetadataGenerator(dataset_path={str(tmp_path)!r}, name='seed')"
        ".generate_metadata();"
        "print(json.dumps(m['distribution'], sort_keys=False))"
    )
    outputs = set()
    for seed in ("0", "1", "42"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
            check=True,
        )
        outputs.add(result.stdout.strip())

    assert len(outputs) == 1, "output varies with PYTHONHASHSEED"
    assert json.loads(outputs.pop())


# --------------------------------------------------------------------------
# Descriptions name the file as stored
# --------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", WRAPPER_SUFFIXES)
def test_a_description_names_the_file_a_reader_can_find(
    suffix: str, tmp_path: Path
) -> None:
    """A manifest that names a file not on disk sends the reader nowhere."""
    write_wrapped(tmp_path, "sample.csv", b"id,name\n1,Ada\n", suffix)

    metadata = MetadataGenerator(
        dataset_path=str(tmp_path), name="described"
    ).generate_metadata()

    (record_set,) = metadata["recordSet"]
    stored = f"sample.csv{suffix}"

    assert record_set["@id"] == "sample"
    assert stored in record_set["description"], record_set["description"]
    for field in record_set["field"]:
        assert stored in field["description"], field["description"]


def test_a_plain_file_is_described_exactly_as_before(tmp_path: Path) -> None:
    """The uncompressed path is untouched: no dangling wrapper, no change."""
    (tmp_path / "sample.csv").write_text("id,name\n1,Ada\n")

    metadata = MetadataGenerator(
        dataset_path=str(tmp_path), name="described"
    ).generate_metadata()

    (record_set,) = metadata["recordSet"]
    assert record_set["description"] == "Records from sample.csv"
    assert record_set["field"][0]["description"] == "Column 'id' from sample.csv"
