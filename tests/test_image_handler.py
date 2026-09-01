"""Tests for image file handler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from croissant_baker.handlers.image_handler import (
    _IMAGE_MAGIC_CHECKS,
    _MIME_TYPES,
    SUPPORTED_EXTENSIONS,
    ImageHandler,
    collect_image_summary,
)
from croissant_baker.handlers.registry import builtin_handlers
from croissant_baker.sources import make_source

from tests.helpers import (
    PNG_1X1,
    WRAPPER_SUFFIXES,
    tiff_bytes,
    write_wrapped,
)


@pytest.fixture
def handler() -> ImageHandler:
    return ImageHandler()


# Minimal magic-byte stubs per supported extension. These are not full
# images — they only need enough bytes to satisfy claims()'s check.
_IMAGE_STUBS = {
    ".png": b"\x89PNG\r\n\x1a\n",
    ".jpg": b"\xff\xd8\xff\xe0",
    ".jpeg": b"\xff\xd8\xff\xe0",
    ".gif": b"GIF89a",
    ".bmp": b"BM\x00\x00\x00\x00",
    ".webp": b"RIFF\x00\x00\x00\x00WEBP",
    ".tiff": b"II*\x00",
    ".tif": b"MM\x00*",
    ".btf": b"II+\x00",
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
        "tissue.btf",
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


def test_a_renamed_file_is_declined_at_debug(
    handler: ImageHandler, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The generator owns the user-facing warning, so the handler's note is
    debug — asserted, or a regression to WARNING doubles every skip."""
    impostor = tmp_path / "fake.png"
    impostor.write_bytes(b"<!DOCTYPE html><html></html>")

    with caplog.at_level("DEBUG", logger="croissant_baker.handlers.image_handler"):
        assert handler.claims(make_source(impostor)) is False

    assert [
        r
        for r in caplog.records
        if r.levelname == "DEBUG"
        and impostor.name in r.message
        and "magic bytes" in r.message
    ], caplog.records


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


def test_extract_metadata_separate_planar_tiff(
    handler: ImageHandler, tmp_path: Path
) -> None:
    """Regression test for TIFFs whose band axis is stored first."""
    path = tmp_path / "separate_planar.tiff"
    tifffile.imwrite(
        str(path),
        np.zeros((12, 5, 7), dtype=np.uint8),
        photometric="minisblack",
        planarconfig="separate",
    )

    props = handler.extract(make_source(path))["image_properties"]

    assert (props["width"], props["height"]) == (7, 5)
    assert props["num_bands"] == 12
    assert props["image_format"] == "TIFF"


# --------------------------------------------------------------------------
# BigTIFF
# --------------------------------------------------------------------------

BIGTIFF = tiff_bytes(size=16, bigtiff=True)


def test_every_supported_extension_is_declared_typed_and_sniffed() -> None:
    """The four tables that have to agree, and the reason BigTIFF went
    unclaimed for so long: ``_TIFF_MAGICS`` already accepted its version byte,
    but the magic check is keyed by extension first, and ``.btf`` was in none
    of the tables that key it."""
    assert set(ImageHandler.EXTENSIONS) == SUPPORTED_EXTENSIONS
    assert set(_MIME_TYPES) == SUPPORTED_EXTENSIONS
    assert set(_IMAGE_MAGIC_CHECKS) == SUPPORTED_EXTENSIONS


def test_a_bigtiff_carries_the_magic_the_handler_checks_for() -> None:
    """Guards the fixture: BigTIFF differs from classic TIFF in one version
    byte, 0x2b against 0x2a, and a fixture written as classic TIFF would let
    the claim below pass for the wrong reason."""
    assert BIGTIFF[:4] == b"II+\x00"


def test_a_bigtiff_is_claimed_and_described(
    handler: ImageHandler, tmp_path: Path
) -> None:
    """Any TIFF writer switches to BigTIFF at 4 GiB, which is where whole-slide
    imaging, EM volumes and geospatial rasters all live."""
    path = tmp_path / "tissue.btf"
    path.write_bytes(BIGTIFF)
    source = make_source(path)

    assert handler.claims(source) is True

    props = handler.extract(source)["image_properties"]

    assert (props["width"], props["height"]) == (16, 16)
    assert props["num_bands"] == 1
    # BigTIFF is a TIFF variant, and a second token here would land in the
    # format breakdown that the record-set description reports.
    assert props["image_format"] == "TIFF"


def test_a_btf_without_bigtiff_magic_is_declined(
    handler: ImageHandler, tmp_path: Path
) -> None:
    path = tmp_path / "impostor.btf"
    path.write_bytes(PNG_1X1)

    assert handler.claims(make_source(path)) is False


@pytest.mark.parametrize(
    "handler_name", sorted(type(h).__name__ for h in builtin_handlers())
)
def test_only_the_image_handler_claims_a_bigtiff(
    handler_name: str, tmp_path: Path
) -> None:
    """The shared exclusive-format sweep cannot reach this: ``.btf`` resolves to
    ImageHandler, and the sweep then writes that owner's first sample, which is
    ``pixel.png``. So no handler is ever asked about a ``.btf`` but here."""
    path = tmp_path / "tissue.btf"
    path.write_bytes(BIGTIFF)
    other = next(h for h in builtin_handlers() if type(h).__name__ == handler_name)

    claimed = other.claims(make_source(path))

    assert claimed is (handler_name == "ImageHandler")


@pytest.mark.parametrize("suffix", WRAPPER_SUFFIXES)
def test_a_wrapped_bigtiff_is_read_through_the_wrapper(
    handler: ImageHandler, dataset: Path, suffix: str
) -> None:
    """tifffile seeks to the end of a BigTIFF to reach its offsets, and on a
    compressed stream that is a decompression of the whole file. It has to
    work, and it is the dearest read this handler does."""
    path = write_wrapped(dataset, "tissue.btf", BIGTIFF, suffix)

    props = handler.extract(make_source(path, Path("tissue.btf")))["image_properties"]

    assert (props["width"], props["height"]) == (16, 16)


# --------------------------------------------------------------------------
# Which backend reads a TIFF
# --------------------------------------------------------------------------

PLAIN_TIFF = tiff_bytes()


def test_a_tiff_is_read_through_tifffile_and_not_pillow(
    handler: ImageHandler, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pillow is a display library that happens to open TIFFs: it reports one
    band for a three-channel image, and decodes tag 270 as latin-1, so ``µm``
    comes back mojibake. tifffile is the reference reader and the only one that
    exposes the OME header at all."""
    from croissant_baker.handlers import image_handler as module

    def fail(_source):
        raise AssertionError("a TIFF was read through Pillow")

    monkeypatch.setattr(module, "_read_with_pillow", fail)
    path = tmp_path / "scan.tif"
    path.write_bytes(PLAIN_TIFF)

    props = handler.extract(make_source(path))["image_properties"]

    assert props["image_format"] == "TIFF"


def test_a_png_is_still_read_through_pillow(
    handler: ImageHandler, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the TIFF extensions move. Nothing else changes backend."""
    from croissant_baker.handlers import image_handler as module

    def fail(_source):
        raise AssertionError("a PNG was read through tifffile")

    monkeypatch.setattr(module, "_read_with_tifffile", fail)
    path = tmp_path / "pixel.png"
    path.write_bytes(PNG_1X1)

    assert handler.extract(make_source(path))["image_properties"]["width"] == 1


def test_an_unreadable_tiff_raises_a_value_error_naming_the_file(
    handler: ImageHandler, tmp_path: Path
) -> None:
    """The message becomes the reason detail a user reads in ``--report``. The
    shared garbage-bytes sweep writes this handler's first sample name, which
    is a PNG, so a broken TIFF is only covered here."""
    path = tmp_path / "truncated.tif"
    path.write_bytes(BIGTIFF[:20])

    with pytest.raises(ValueError, match="truncated.tif"):
        handler.extract(make_source(path))


# --------------------------------------------------------------------------
# The batch summary
# --------------------------------------------------------------------------


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
