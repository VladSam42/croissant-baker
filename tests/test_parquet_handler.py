"""Tests for Parquet handler."""

from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from croissant_baker.scan import Reason
from croissant_baker.handlers.parquet_handler import ParquetHandler
from croissant_baker.sources import make_source


@pytest.fixture
def handler() -> ParquetHandler:
    return ParquetHandler()


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Create a minimal Parquet file for testing."""
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "name": pa.array(["Alice", "Bob", "Charlie"], type=pa.string()),
            "score": pa.array([9.5, 8.3, 7.1], type=pa.float64()),
        }
    )
    path = tmp_path / "test.parquet"
    pq.write_table(table, str(path))
    return path


# ---------------------------------------------------------------------------
# can_handle — extension + magic bytes (issue #93)
#
# can_handle enforces the registry contract: True implies extract_metadata
# can read the file. Tests cover the failure modes (wrong extension, right
# extension/wrong content, truncated, missing) plus the happy path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["data.csv", "data.txt", "data"])
def test_can_handle_rejects_unsupported_extensions(
    handler: ParquetHandler, name: str
) -> None:
    """Non-.parquet extensions are rejected before any I/O."""
    assert handler.claims(make_source(Path(name))) is False


_PARQUET_LOGGER = "croissant_baker.handlers.parquet_handler"


def test_can_handle_missing_file_does_not_warn(
    handler: ParquetHandler, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing .parquet path is silently rejected (no spurious warning)
    since the caller, not the file, is at fault."""
    with caplog.at_level("WARNING", logger=_PARQUET_LOGGER):
        assert handler.claims(make_source(Path("/nonexistent/data.parquet"))) is False
    assert caplog.records == []


def test_can_handle_rejects_wrong_magic(
    handler: ParquetHandler, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A .parquet file without PAR1 magic is rejected AND a WARNING is
    logged identifying the file and the missing PAR1 header.

    Regression for #93: prevents the registry from dispatching a renamed
    file to ParquetHandler.extract_metadata and crashing inside pyarrow,
    while still surfacing the skip so the user knows what was dropped.
    """
    impostor = tmp_path / "fake.parquet"
    impostor.write_bytes(b"not a parquet file at all")
    with caplog.at_level("WARNING", logger=_PARQUET_LOGGER):
        assert handler.claims(make_source(impostor)) is False
    assert any(
        impostor.name in r.message and "PAR1 header" in r.message
        for r in caplog.records
    ), f"expected a WARNING naming {impostor} and 'PAR1 header', got {caplog.records}"


def test_can_handle_rejects_truncated_parquet(
    handler: ParquetHandler, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A truncated Parquet (start magic only, no footer magic) is rejected
    AND a WARNING explicitly mentions the missing footer / possible truncation."""
    truncated = tmp_path / "truncated.parquet"
    truncated.write_bytes(b"PAR1" + b"\x00" * 32)
    with caplog.at_level("WARNING", logger=_PARQUET_LOGGER):
        assert handler.claims(make_source(truncated)) is False
    assert any(
        truncated.name in r.message
        and ("footer" in r.message or "truncated" in r.message)
        for r in caplog.records
    ), (
        f"expected a WARNING naming {truncated} and footer/truncated, got {caplog.records}"
    )


def test_can_handle_rejects_too_small_parquet(
    handler: ParquetHandler, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A .parquet file under 8 bytes (cannot fit two PAR1 magics) is rejected
    with a WARNING that names the file and its size."""
    tiny = tmp_path / "tiny.parquet"
    tiny.write_bytes(b"PAR")  # 3 bytes, well under the 8-byte minimum
    with caplog.at_level("WARNING", logger=_PARQUET_LOGGER):
        assert handler.claims(make_source(tiny)) is False
    assert any(
        tiny.name in r.message and "too small" in r.message for r in caplog.records
    ), f"expected a WARNING naming {tiny} and 'too small', got {caplog.records}"


def test_can_handle_accepts_real_parquet(
    handler: ParquetHandler, sample_parquet: Path
) -> None:
    """A real Parquet file (PAR1 at start AND end) is accepted, including
    when the extension is uppercased."""
    assert handler.claims(make_source(sample_parquet)) is True

    upper = sample_parquet.with_name("test.PARQUET")
    upper.write_bytes(sample_parquet.read_bytes())
    assert handler.claims(make_source(upper)) is True


# ---------------------------------------------------------------------------
# extract_metadata
# ---------------------------------------------------------------------------


def test_extract_metadata(handler: ParquetHandler, sample_parquet: Path) -> None:
    """Test Parquet metadata extraction returns correct structure."""
    metadata = handler.extract(make_source(sample_parquet))

    assert metadata["file_name"] == "test.parquet"
    assert metadata["encoding_format"] == "application/vnd.apache.parquet"
    assert metadata["file_size"] > 0
    assert len(metadata["sha256"]) == 64
    assert metadata["num_rows"] == 3
    assert metadata["num_columns"] == 3
    assert metadata["columns"] == ["id", "name", "score"]

    column_types = metadata["column_types"]
    assert column_types["id"] == "cr:Int64"
    assert column_types["name"] == "sc:Text"
    assert column_types["score"] == "cr:Float64"


# ---------------------------------------------------------------------------
# Resource leak: file handle must be closed after extract_metadata (#54)
# ---------------------------------------------------------------------------


def test_parquet_file_handle_closed(
    handler: ParquetHandler, sample_parquet: Path
) -> None:
    """Verify the underlying file handle is closed after metadata extraction.

    Regression test for GitHub issue #54: ParquetFile was opened without a
    context manager, leaking file descriptors until garbage collection.
    """
    captured_handles: list[pq.ParquetFile] = []
    _OrigParquetFile = pq.ParquetFile

    def _spy(*args, **kwargs):
        pf = _OrigParquetFile(*args, **kwargs)
        captured_handles.append(pf)
        return pf

    with patch(
        "croissant_baker.handlers.parquet_handler.ParquetFile", side_effect=_spy
    ):
        handler.extract(make_source(sample_parquet))

    assert len(captured_handles) == 1, "ParquetFile should be opened exactly once"
    assert captured_handles[0].reader.closed, (
        "ParquetFile reader must be closed after extract_metadata returns"
    )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_extract_metadata_not_found(handler: ParquetHandler) -> None:
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        handler.extract(make_source(Path("/nonexistent/data.parquet")))


# ---------------------------------------------------------------------------
# build_croissant
# ---------------------------------------------------------------------------


def _pq_meta(name, rel_path, cols=None):
    return {
        "file_name": name,
        "relative_path": rel_path,
        "column_types": cols or {"id": "sc:Text", "value": "cr:Float64"},
        "num_rows": 10,
        "encoding_format": "application/vnd.apache.parquet",
    }


# --- Grouping is on evidence, not on directory membership -------------------

_BOUNDARIES = {"cell_id": "sc:Text", "vertex_x": "cr:Float32"}
_CELLS = {"cell_id": "sc:Text", "x_centroid": "cr:Float32", "counts": "cr:Int32"}


def test_distinct_names_are_distinct_tables_even_with_one_schema(
    handler: ParquetHandler,
) -> None:
    """Two 10x boundary tables share a schema but are not partitions of each other."""
    metas = [
        _pq_meta("cell_boundaries.parquet", "s1/cell_boundaries.parquet", _BOUNDARIES),
        _pq_meta(
            "nucleus_boundaries.parquet", "s1/nucleus_boundaries.parquet", _BOUNDARIES
        ),
    ]
    filesets, record_sets, conflicts = handler.build_croissant(
        metas, ["file_0", "file_1"]
    )

    assert filesets == []
    assert conflicts == []
    assert sorted(r.name for r in record_sets) == [
        "cell_boundaries",
        "nucleus_boundaries",
    ]


def test_a_vendor_directory_keeps_every_schema(handler: ParquetHandler) -> None:
    """Four tables in one sample directory stay four, with their own columns."""
    metas = [
        _pq_meta("cell_boundaries.parquet", "s1/cell_boundaries.parquet", _BOUNDARIES),
        _pq_meta("cells.parquet", "s1/cells.parquet", _CELLS),
        _pq_meta(
            "nucleus_boundaries.parquet", "s1/nucleus_boundaries.parquet", _BOUNDARIES
        ),
    ]
    _, record_sets, _ = handler.build_croissant(metas, ["f0", "f1", "f2"])

    by_name = {r.name: [f.name for f in r.fields] for r in record_sets}
    assert set(by_name) == {"cell_boundaries", "cells", "nucleus_boundaries"}
    assert by_name["cells"] == ["cell_id", "x_centroid", "counts"]
    assert by_name["cell_boundaries"] == ["cell_id", "vertex_x"]


def test_shards_that_disagree_on_schema_are_reported(handler: ParquetHandler) -> None:
    """Drift inside one partitioned table is declined, not folded into the first schema."""
    metas = [
        _pq_meta("part-00000.parquet", "events/part-00000.parquet", _BOUNDARIES),
        _pq_meta("part-00001.parquet", "events/part-00001.parquet", _BOUNDARIES),
        _pq_meta("part-00002.parquet", "events/part-00002.parquet", _CELLS),
    ]
    _, record_sets, conflicts = handler.build_croissant(metas, ["f0", "f1", "f2"])

    assert len(record_sets) == 1
    assert [f.name for f in record_sets[0].fields] == ["cell_id", "vertex_x"]

    ((index, reason, detail),) = conflicts
    assert index == 2
    assert reason is Reason.PARTITION_SCHEMA_CONFLICT
    assert "3 columns, expected 2" in detail


def test_same_named_directories_do_not_collide(handler: ParquetHandler) -> None:
    """Hive layouts repeat a directory name under different parents."""
    metas = [
        _pq_meta("part-00000.parquet", "a/events/part-00000.parquet"),
        _pq_meta("part-00001.parquet", "a/events/part-00001.parquet"),
        _pq_meta("part-00000.parquet", "b/events/part-00000.parquet"),
        _pq_meta("part-00001.parquet", "b/events/part-00001.parquet"),
    ]
    filesets, record_sets, _ = handler.build_croissant(metas, ["f0", "f1", "f2", "f3"])

    assert len({f.id for f in filesets}) == 2
    assert len({r.id for r in record_sets}) == 2
    # Each record set's fields must reach its own FileSet, not the first one.
    for record_set, fileset in zip(record_sets, filesets):
        assert record_set.fields[0].source.file_set == fileset.id


def test_a_shared_directory_names_tables_after_their_files(
    handler: ParquetHandler,
) -> None:
    """The parent directory only names a table while it is the directory's only one."""
    metas = [
        _pq_meta("part-00000.parquet", "events/part-00000.parquet", _BOUNDARIES),
        _pq_meta("part-00001.parquet", "events/part-00001.parquet", _BOUNDARIES),
        _pq_meta("lookup.parquet", "events/lookup.parquet", _CELLS),
    ]
    filesets, record_sets, _ = handler.build_croissant(metas, ["f0", "f1", "f2"])

    assert sorted(r.name for r in record_sets) == ["lookup", "part"]
    # The glob would reach the neighbouring table, so the shards are listed.
    assert filesets[0].includes == [
        "events/part-00000.parquet",
        "events/part-00001.parquet",
    ]


def test_parquet_build_croissant_standalone(handler: ParquetHandler) -> None:
    filesets, record_sets, _ = handler.build_croissant(
        [_pq_meta("data.parquet", "data.parquet")], ["file_0"]
    )
    assert filesets == []
    assert record_sets[0].name == "data"


def test_parquet_build_croissant_single_file_in_subdir(handler: ParquetHandler) -> None:
    filesets, record_sets, _ = handler.build_croissant(
        [_pq_meta("part-00000.parquet", "events/part-00000.parquet")], ["file_0"]
    )
    assert filesets == []
    assert record_sets[0].name == "events"


def test_parquet_build_croissant_partitioned(handler: ParquetHandler) -> None:
    metas = [
        _pq_meta("part-00000.parquet", "events/part-00000.parquet"),
        _pq_meta("part-00001.parquet", "events/part-00001.parquet"),
    ]
    filesets, record_sets, _ = handler.build_croissant(metas, ["file_0", "file_1"])
    assert len(filesets) == 1
    assert len(record_sets) == 1
    assert record_sets[0].name == "events"
    assert "events/*.parquet" in filesets[0].includes


def test_parquet_array_shape_fixed_vs_variable(
    handler: ParquetHandler, tmp_path: Path
) -> None:
    """Fixed-size lists report exact dim; variable-length lists report -1."""
    schema = pa.schema(
        [
            ("embedding", pa.list_(pa.float32(), 384)),  # fixed-size: dim 384
            ("tags", pa.list_(pa.string())),  # variable-length
        ]
    )
    table = pa.table(
        {"embedding": [[0.0] * 384], "tags": [["a", "b"]]},
        schema=schema,
    )
    path = tmp_path / "vectors.parquet"
    pq.write_table(table, str(path))

    meta = handler.extract(make_source(path))
    meta["relative_path"] = "vectors.parquet"
    _, record_sets, _ = handler.build_croissant([meta], ["file_0"])

    fields = {f.name: f for f in record_sets[0].fields}
    assert fields["embedding"].is_array is True
    assert fields["embedding"].array_shape == "384"
    assert fields["tags"].is_array is True
    assert fields["tags"].array_shape == "-1"


def test_a_declined_shard_is_not_left_inside_the_fileset(
    handler: ParquetHandler,
) -> None:
    """A glob over the directory would re-admit the shard the schema kept out."""
    metas = [
        _pq_meta("part-00000.parquet", "events/part-00000.parquet", _BOUNDARIES),
        _pq_meta("part-00001.parquet", "events/part-00001.parquet", _BOUNDARIES),
        _pq_meta("part-00002.parquet", "events/part-00002.parquet", _CELLS),
    ]
    filesets, _, conflicts = handler.build_croissant(metas, ["f0", "f1", "f2"])

    assert len(conflicts) == 1
    assert filesets[0].includes == [
        "events/part-00000.parquet",
        "events/part-00001.parquet",
    ]


def test_shards_are_grouped_on_the_arrow_schema_not_the_croissant_types(
    handler: ParquetHandler,
) -> None:
    """Two timestamp units are one Croissant type and two different tables."""
    metas = []
    for i, unit in enumerate(("s", "ns")):
        meta = _pq_meta(f"part-0000{i}.parquet", f"events/part-0000{i}.parquet")
        meta["column_types"] = {"t": "sc:DateTime"}
        meta["arrow_schema"] = pa.schema([pa.field("t", pa.timestamp(unit))])
        metas.append(meta)

    filesets, record_sets, conflicts = handler.build_croissant(metas, ["f0", "f1"])

    assert filesets == []
    assert conflicts == []
    assert len({r.id for r in record_sets}) == 2


def test_root_level_files_are_never_declined(handler: ParquetHandler) -> None:
    """Root files never pair, so a schema difference between them settles nothing."""
    metas = [
        _pq_meta("part-00000.parquet", "part-00000.parquet", _BOUNDARIES),
        _pq_meta("part-00001.parquet", "part-00001.parquet", _CELLS),
    ]
    filesets, record_sets, conflicts = handler.build_croissant(metas, ["f0", "f1"])

    assert filesets == []
    assert conflicts == []
    assert sorted(r.name for r in record_sets) == ["part-00000", "part-00001"]


def test_two_templates_reducing_to_one_stem_get_distinct_ids(
    handler: ParquetHandler,
) -> None:
    """``part-000`` and ``part_000`` share a directory, a stem, and every parent."""
    metas = [
        _pq_meta("part-000.parquet", "d/part-000.parquet", _BOUNDARIES),
        _pq_meta("part-001.parquet", "d/part-001.parquet", _BOUNDARIES),
        _pq_meta("part_000.parquet", "d/part_000.parquet", _CELLS),
        _pq_meta("part_001.parquet", "d/part_001.parquet", _CELLS),
    ]
    filesets, record_sets, _ = handler.build_croissant(metas, ["f0", "f1", "f2", "f3"])

    assert len({f.id for f in filesets}) == 2
    assert len({r.id for r in record_sets}) == 2


def test_a_number_inside_a_word_is_not_a_shard_index(handler: ParquetHandler) -> None:
    """``assay1`` and ``assay2`` are two tables; only a separated run is an index."""
    metas = [
        _pq_meta("assay1.parquet", "d/assay1.parquet", _BOUNDARIES),
        _pq_meta("assay2.parquet", "d/assay2.parquet", _BOUNDARIES),
    ]
    filesets, record_sets, conflicts = handler.build_croissant(metas, ["f0", "f1"])

    assert filesets == []
    assert conflicts == []
    assert sorted(r.name for r in record_sets) == ["assay1", "assay2"]


def test_differing_years_are_not_merged_or_declined(handler: ParquetHandler) -> None:
    """``report2024`` and ``report2025`` disagree on schema and both survive."""
    metas = [
        _pq_meta("report2024.parquet", "d/report2024.parquet", _BOUNDARIES),
        _pq_meta("report2025.parquet", "d/report2025.parquet", _CELLS),
    ]
    _, record_sets, conflicts = handler.build_croissant(metas, ["f0", "f1"])

    assert conflicts == []
    assert sorted(r.name for r in record_sets) == ["report2024", "report2025"]
