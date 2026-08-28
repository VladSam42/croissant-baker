"""Tests for Parquet handler."""

from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


def test_parquet_build_croissant_standalone(handler: ParquetHandler) -> None:
    filesets, record_sets = handler.build_croissant(
        [_pq_meta("data.parquet", "data.parquet")], ["file_0"]
    )
    assert filesets == []
    assert record_sets[0].name == "data"


def test_parquet_build_croissant_single_file_in_subdir(handler: ParquetHandler) -> None:
    filesets, record_sets = handler.build_croissant(
        [_pq_meta("part-00000.parquet", "events/part-00000.parquet")], ["file_0"]
    )
    assert filesets == []
    assert record_sets[0].name == "events"


def test_parquet_build_croissant_partitioned(handler: ParquetHandler) -> None:
    metas = [
        _pq_meta("part-00000.parquet", "events/part-00000.parquet"),
        _pq_meta("part-00001.parquet", "events/part-00001.parquet"),
    ]
    filesets, record_sets = handler.build_croissant(metas, ["file_0", "file_1"])
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
    _, record_sets = handler.build_croissant([meta], ["file_0"])

    fields = {f.name: f for f in record_sets[0].fields}
    assert fields["embedding"].is_array is True
    assert fields["embedding"].array_shape == "384"
    assert fields["tags"].is_array is True
    assert fields["tags"].array_shape == "-1"
