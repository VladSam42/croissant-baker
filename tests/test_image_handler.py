"""Tests for image file handler."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

import croissant_baker.handlers.image_handler as image_handler_module
from croissant_baker.handlers.image_handler import (
    ImageHandler,
    collect_image_summary,
)
from croissant_baker.sources import make_source


@pytest.fixture
def handler() -> ImageHandler:
    return ImageHandler()


# ---------------------------------------------------------------------------
# can_handle — extension + magic bytes (issue #93)
#
# can_handle enforces the registry contract: True implies extract_metadata
# can read the file. Tests cover the three failure modes (wrong extension,
# right extension/wrong content, missing file) plus the happy path per
# extension. Each accepted-extension case writes a minimal magic-byte stub
# so we exercise real files, not bare path strings.
# ---------------------------------------------------------------------------


# Minimal magic-byte stubs per supported extension. These are not full
# images — they only need enough bytes to satisfy can_handle's check.
_IMAGE_STUBS = {
    ".png": b"\x89PNG\r\n\x1a\n",
    ".jpg": b"\xff\xd8\xff\xe0",
    ".jpeg": b"\xff\xd8\xff\xe0",
    ".gif": b"GIF89a",
    ".bmp": b"BM\x00\x00\x00\x00",
    ".webp": b"RIFF\x00\x00\x00\x00WEBP",
    ".tiff": b"II*\x00",
    ".tif": b"MM\x00*",
    ".ico": b"\x00\x00\x01\x00",
}


@pytest.mark.parametrize(
    "filename",
    [
        "photo.jpg",
        "photo.jpeg",
        "photo.JPG",  # case-insensitive suffix
        "scan.png",
        "scan.PNG",
        "frame.gif",
        "icon.bmp",
        "hero.webp",
        "satellite.tiff",
        "satellite.tif",
        "satellite.TIFF",
        "image.ico",
    ],
)
def test_can_handle_accepts_supported_extensions_with_magic(
    handler: ImageHandler, tmp_path: Path, filename: str
) -> None:
    """Files whose extension is supported AND whose content matches the
    extension's magic bytes are accepted."""
    p = tmp_path / filename
    p.write_bytes(_IMAGE_STUBS[p.suffix.lower()])
    assert handler.claims(make_source(p)) is True


@pytest.mark.parametrize(
    "name", ["data.csv", "model.parquet", "readme.txt", "record.hea"]
)
def test_can_handle_rejects_unsupported_extensions(
    handler: ImageHandler, name: str
) -> None:
    """Non-image extensions are rejected before any I/O — bare path is fine."""
    assert handler.claims(make_source(Path(name))) is False


def test_can_handle_rejects_missing_file(handler: ImageHandler) -> None:
    """A path with an image extension but no file on disk is rejected.

    Without a file we cannot honor the contract that extract_metadata won't
    crash, so can_handle must say no.
    """
    assert handler.claims(make_source(Path("/nonexistent/photo.png"))) is False


def test_can_handle_rejects_wrong_magic(
    handler: ImageHandler, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Right extension, wrong content (e.g., HTML renamed to .png) is rejected
    AND a WARNING is logged naming the file so the user knows what was skipped.

    Regression for #93: prevents the registry from dispatching a renamed
    file to ImageHandler.extract_metadata and crashing inside Pillow, and
    surfaces the skip so the user is not blindsided by a missing file count.
    """
    impostor = tmp_path / "fake.png"
    impostor.write_bytes(b"<!DOCTYPE html><html></html>")
    with caplog.at_level("WARNING", logger="croissant_baker.handlers.image_handler"):
        assert handler.claims(make_source(impostor)) is False
    assert any(
        impostor.name in r.message and "magic bytes" in r.message
        for r in caplog.records
    ), f"expected a WARNING naming {impostor} and 'magic bytes', got {caplog.records}"


def test_can_handle_missing_file_does_not_warn(
    handler: ImageHandler, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing file is silently rejected (no spurious warnings) since the
    caller, not the file, is at fault."""
    with caplog.at_level("WARNING", logger="croissant_baker.handlers.image_handler"):
        assert handler.claims(make_source(Path("/nonexistent/photo.png"))) is False
    assert caplog.records == []


def test_can_handle_accepts_real_image(handler: ImageHandler, tmp_path: Path) -> None:
    """A fully-encoded PNG (not just a magic stub) is accepted."""
    from PIL import Image

    real_png = tmp_path / "real.png"
    Image.new("RGB", (4, 4), color="red").save(real_png)
    assert handler.claims(make_source(real_png)) is True


def test_can_handle_accepts_bigtiff(handler: ImageHandler, tmp_path: Path) -> None:
    """BigTIFF (TIFF variant for files >4GB, version byte 0x2b) is accepted.

    Regression for #93: Pillow and tifffile both read BigTIFF, so the
    contract requires can_handle to claim it.
    """
    bigtiff = tmp_path / "huge.tiff"
    tifffile.imwrite(
        str(bigtiff),
        np.zeros((4, 4, 3), dtype=np.uint8),
        bigtiff=True,
    )
    # Verify we wrote a real BigTIFF (version byte 0x2b, not 0x2a).
    assert bigtiff.read_bytes()[:4] in (b"II+\x00", b"MM\x00+")
    assert handler.claims(make_source(bigtiff)) is True
    # And extract_metadata must succeed — the contract.
    meta = handler.extract(make_source(bigtiff))
    assert meta["image_properties"]["width"] == 4
    assert meta["image_properties"]["height"] == 4


# ---------------------------------------------------------------------------
# extract_metadata — standard JPG images (glaucoma fundus)
# ---------------------------------------------------------------------------


@pytest.fixture
def glaucoma_image_path() -> Path:
    """Path to a sample JPG from the glaucoma fundus dataset."""
    p = (
        Path(__file__).parent
        / "data"
        / "input"
        / "glaucoma_fundus"
        / "Images"
        / "0_0.jpg"
    )
    if not p.exists():
        pytest.skip(f"Glaucoma fundus image not found at {p}")
    return p


def test_extract_metadata_jpg(handler: ImageHandler, glaucoma_image_path: Path) -> None:
    meta = handler.extract(make_source(glaucoma_image_path))

    assert meta["file_name"] == "0_0.jpg"
    assert meta["encoding_format"] == "image/jpeg"
    assert meta["file_size"] > 0
    assert len(meta["sha256"]) == 64

    props = meta["image_properties"]
    assert props["width"] > 0
    assert props["height"] > 0
    assert props["num_bands"] in (1, 3, 4)
    assert props["image_format"] == "JPEG"


# ---------------------------------------------------------------------------
# extract_metadata — multi-band TIFF images (satellite)
# ---------------------------------------------------------------------------


@pytest.fixture
def satellite_tiff_path() -> Path:
    """Path to a sample TIFF from the satellite dataset."""
    p = (
        Path(__file__).parent
        / "data"
        / "input"
        / "satellite_public_health"
        / "images"
        / "5001"
        / "image_2016-01-03.tiff"
    )
    if not p.exists():
        pytest.skip(f"Satellite TIFF not found at {p}")
    return p


def test_extract_metadata_tiff(
    handler: ImageHandler, satellite_tiff_path: Path
) -> None:
    meta = handler.extract(make_source(satellite_tiff_path))

    assert meta["file_name"] == "image_2016-01-03.tiff"
    assert meta["encoding_format"] == "image/tiff"
    assert meta["file_size"] > 0
    assert len(meta["sha256"]) == 64

    props = meta["image_properties"]
    assert props["width"] > 0
    assert props["height"] > 0
    # Sentinel-2 images have 12 bands
    assert props["num_bands"] == 12
    assert props["image_format"] == "TIFF"


@pytest.fixture
def separate_planar_tiff_path(tmp_path: Path) -> Path:
    """Create a multi-band TIFF that forces the tifffile fallback path."""
    path = tmp_path / "separate_planar.tiff"
    data = np.zeros((12, 5, 7), dtype=np.uint8)
    tifffile.imwrite(
        str(path),
        data,
        photometric="minisblack",
        planarconfig="separate",
    )
    return path


def test_extract_metadata_separate_planar_tiff(
    handler: ImageHandler,
    separate_planar_tiff_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for TIFFs whose band axis is stored first."""

    def _force_tifffile_fallback(_path: Path) -> None:
        raise RuntimeError("force tifffile fallback")

    monkeypatch.setattr(
        image_handler_module,
        "_read_with_pillow",
        _force_tifffile_fallback,
    )

    meta = handler.extract(make_source(separate_planar_tiff_path))

    assert meta["file_name"] == "separate_planar.tiff"
    assert meta["encoding_format"] == "image/tiff"
    assert meta["file_size"] > 0
    assert len(meta["sha256"]) == 64

    props = meta["image_properties"]
    assert props["width"] == 7
    assert props["height"] == 5
    assert props["num_bands"] == 12
    assert props["image_format"] == "TIFF"


# ---------------------------------------------------------------------------
# extract_metadata — error cases
# ---------------------------------------------------------------------------


def test_extract_metadata_not_found(handler: ImageHandler) -> None:
    with pytest.raises(FileNotFoundError):
        handler.extract(make_source(Path("/nonexistent/image.jpg")))


def test_extract_metadata_corrupt_file(handler: ImageHandler, tmp_path: Path) -> None:
    bad_img = tmp_path / "corrupt.jpg"
    bad_img.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Failed to read image"):
        handler.extract(make_source(bad_img))


# ---------------------------------------------------------------------------
# collect_image_summary
# ---------------------------------------------------------------------------


def test_collect_image_summary_empty() -> None:
    assert collect_image_summary([]) == {}


def test_collect_image_summary() -> None:
    metas = [
        {
            "image_properties": {
                "width": 100,
                "height": 200,
                "num_bands": 3,
                "image_format": "JPEG",
            }
        },
        {
            "image_properties": {
                "width": 640,
                "height": 480,
                "num_bands": 3,
                "image_format": "JPEG",
            }
        },
        {
            "image_properties": {
                "width": 256,
                "height": 256,
                "num_bands": 12,
                "image_format": "TIFF",
            }
        },
    ]
    summary = collect_image_summary(metas)

    assert summary["num_images"] == 3
    assert summary["width_range"] == (100, 640)
    assert summary["height_range"] == (200, 480)
    assert summary["num_bands_range"] == (3, 12)
    assert summary["format_counts"] == {"JPEG": 2, "TIFF": 1}


def test_collect_image_summary_missing_properties() -> None:
    metas = [
        {
            "image_properties": {
                "width": 100,
                "height": 200,
                "num_bands": 3,
                "image_format": "JPEG",
            }
        },
        {},  # Missing entirely
        {
            "image_properties": {
                "width": 640,  # Missing some keys
                "image_format": "JPEG",
            }
        },
    ]
    summary = collect_image_summary(metas)

    assert summary["num_images"] == 2
    assert summary["width_range"] == (100, 640)
    assert summary["height_range"] == (200, 200)
    assert summary["num_bands_range"] == (3, 3)
    assert summary["format_counts"] == {"JPEG": 2}


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------


def test_image_handler_registered(tmp_path: Path) -> None:
    """ImageHandler should be discoverable via the global registry for real
    image files (i.e. extension AND magic bytes match)."""
    from croissant_baker.handlers.registry import find_handler, register_all_handlers

    register_all_handlers()
    for name, magic in [
        ("photo.jpg", _IMAGE_STUBS[".jpg"]),
        ("scan.png", _IMAGE_STUBS[".png"]),
        ("satellite.tiff", _IMAGE_STUBS[".tiff"]),
    ]:
        p = tmp_path / name
        p.write_bytes(magic)
        assert find_handler(p) is not None, f"no handler dispatched for {name}"


# ---------------------------------------------------------------------------
# build_croissant
# ---------------------------------------------------------------------------


def _img_meta(name, fmt="JPEG", mime="image/jpeg", w=100, h=100, bands=3):
    return {
        "file_name": name,
        "encoding_format": mime,
        "image_properties": {
            "width": w,
            "height": h,
            "num_bands": bands,
            "image_format": fmt,
        },
    }


def test_image_build_croissant(handler: ImageHandler) -> None:
    metas = [_img_meta("a.jpg"), _img_meta("b.jpg")]
    filesets, record_sets = handler.build_croissant(metas, ["file_0", "file_1"])

    assert len(filesets) == 1
    assert len(record_sets) == 1
    assert record_sets[0].name == "images"
    assert "**/*.jpg" in filesets[0].includes


def test_image_build_croissant_multiband(handler: ImageHandler) -> None:
    metas = [
        _img_meta(f"tile_{i}.tif", fmt="TIFF", mime="image/tiff", bands=12)
        for i in range(3)
    ]
    _, record_sets = handler.build_croissant(metas, [f"file_{i}" for i in range(3)])

    assert "band" in record_sets[0].description
