"""Tests for image file handler."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import tifffile

from croissant_baker.handlers import ome
from croissant_baker.handlers.image_handler import (
    _IMAGE_MAGIC_CHECKS,
    _MIME_TYPES,
    SUPPORTED_EXTENSIONS,
    ImageHandler,
    collect_image_summary,
)
from croissant_baker.handlers.registry import builtin_handlers
from croissant_baker.sources import FileSource, make_source

from tests.helpers import (
    OME_TIFF,
    PNG_1X1,
    WRAPPER_SUFFIXES,
    ome_bomb,
    ome_image,
    ome_xml,
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
    """``_TIFF_MAGICS`` already accepted BigTIFF's version byte. The magic check
    is keyed by extension first, so being in three of these tables is no use."""
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
    """Pillow opens some TIFFs, and reads them worse — see
    ``_read_image_metadata``."""
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
# No pixel data, at any size
# --------------------------------------------------------------------------

TILED_OME = tiff_bytes(ome_xml(ome_image()), planes=4, size=64, tile=(16, 16))


class ReadLog(io.BytesIO):
    """A stream that records the byte interval of every read it serves."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.intervals: list = []
        self.back_seeks = 0

    def read(self, size=-1):
        start = self.tell()
        chunk = super().read(size)
        self.intervals.append((start, start + len(chunk)))
        return chunk

    def seek(self, offset, whence=0):
        before = self.tell()
        position = super().seek(offset, whence)
        if position < before:
            self.back_seeks += 1
        return position


def pixel_intervals(data: bytes) -> list:
    """Where the pixels are, from the offsets and byte counts the TIFF declares."""
    out = []
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        for page in tif.pages:
            tags = page.tags
            offsets = tags.get("TileOffsets") or tags.get("StripOffsets")
            counts = tags.get("TileByteCounts") or tags.get("StripByteCounts")
            out += [(o, o + c) for o, c in zip(offsets.value, counts.value) if c]
    return out


def test_describing_a_tiled_ome_tiff_reads_no_pixel_data(
    handler: ImageHandler,
) -> None:
    """A byte cap would be the wrong assertion: the pull scales with the
    ImageDescription and the IFD count, so 8 KiB holds at 3 channels and
    breaches at 40. Overlap, not "no read starts at a pixel offset", which a
    read beginning earlier and spanning into one would satisfy."""
    log = ReadLog(TILED_OME)
    source = FileSource(
        name="tiled.ome.tif",
        relative_path=Path("tiled.ome.tif"),
        size=len(TILED_OME),
        exists=True,
        _open_binary=lambda: log,
        _digest=lambda: "0" * 64,
    )

    handler.extract(source)

    assert log.intervals, "nothing was read at all, so the check proves nothing"
    pixels = pixel_intervals(TILED_OME)
    assert pixels, "the fixture declares no pixel data to avoid"
    overlaps = [
        (read, pixel)
        for read in log.intervals
        for pixel in pixels
        if read[0] < pixel[1] and pixel[0] < read[1]
    ]
    assert not overlaps, overlaps
    # A backward seek on a wrapped file costs a decompression from offset 0.
    assert log.back_seeks <= 3, log.back_seeks


# --------------------------------------------------------------------------
# The OME header
# --------------------------------------------------------------------------
def test_an_ome_tiff_keeps_its_tiff_tags_alongside_its_header(
    handler: ImageHandler, tmp_path: Path
) -> None:
    """``num_bands`` is TIFF SamplesPerPixel and stays so. It is genuinely 1 for
    a three-channel OME stored as three IFDs, and ``size_c`` is the channel
    count — reporting one of them as the other would lose both."""
    path = tmp_path / "morphology.ome.tif"
    path.write_bytes(OME_TIFF)

    meta = handler.extract(make_source(path))

    assert meta["image_properties"]["num_bands"] == 1
    assert meta["ome"].size_c == 3


BOMB_TIFF = tiff_bytes(ome_bomb())


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("entity declaration", BOMB_TIFF),
        (
            "oversized",
            tiff_bytes(ome_xml(f"<!--{'x' * (ome.MAX_DESCRIPTION_BYTES + 1)}-->")),
        ),
        # Closed at the root, so tifffile still calls it OME, but not
        # well-formed. A description truncated before ``</OME>`` is a
        # different case: nothing identifies it as OME, so it is not refused.
        ("malformed", tiff_bytes(ome_xml("<Image>"))),
    ],
)
def test_a_refused_description_is_warned_about_once(
    handler: ImageHandler,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    label: str,
    payload: bytes,
) -> None:
    """The record-set description carries the count, but a described file has
    no scan-report entry to hang a reason on, so this is the only runtime sign
    that metadata was dropped."""
    path = tmp_path / "a.ome.tif"
    path.write_bytes(payload)

    with caplog.at_level("DEBUG", logger="croissant_baker.handlers"):
        handler.extract(make_source(path))

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, caplog.records
    assert "a.ome.tif" in warnings[0].getMessage()


def test_a_sound_ome_file_is_not_warned_about(
    handler: ImageHandler, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Or every microscopy bake would warn on every file."""
    path = tmp_path / "a.ome.tif"
    path.write_bytes(OME_TIFF)

    with caplog.at_level("DEBUG", logger="croissant_baker.handlers"):
        handler.extract(make_source(path))

    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


# --------------------------------------------------------------------------
# The OME collection
# --------------------------------------------------------------------------

IMAGEJ_TIFF = tiff_bytes("ImageJ=1.53t\nimages=1\nslices=1\n")
BINARY_ONLY_TIFF = tiff_bytes(
    ome_xml('<BinaryOnly UUID="urn:uuid:9c1b" MetadataFile="plate.companion.ome"/>')
)
OME_40 = tiff_bytes(
    ome_xml(
        ome_image(
            pixels='DimensionOrder="XYCZT" Type="uint8" SizeX="8" SizeY="8"'
            ' SizeC="40" SizeZ="1" SizeT="1"',
            channels=("CD3", "CD8"),
        )
    )
)
OME_NO_PHYSICAL_SIZE = tiff_bytes(
    ome_xml(
        ome_image(
            pixels='DimensionOrder="XYCZT" Type="uint16" SizeX="8" SizeY="8"'
            ' SizeC="3" SizeZ="1" SizeT="1"'
        )
    )
)
OME_TWO_IMAGES = tiff_bytes(
    ome_xml(ome_image() + ome_image(identifier="Image:1", channels=("CD3",)))
)
OME_NAMED = tiff_bytes(
    ome_xml(
        ome_image(attrs=' Name="Patient 3 slide 2"'),
        attrs=' UUID="urn:uuid:9c1bde0e-dead-beef" Creator="Acme Scanner 4.2"',
    )
)


def ome_partner(other: str) -> bytes:
    """One file of a multi-file OME set, naming its partner the way OME does."""
    return tiff_bytes(
        ome_xml(
            ome_image(
                trailing=f'<TiffData IFD="0"><UUID FileName="{other}">'
                "urn:uuid:9c1b</UUID></TiffData>"
            )
        )
    )


def described(handler: ImageHandler, directory: Path, files: dict) -> tuple:
    """Write ``files``, extract each, and stamp what the generator stamps."""
    metas = []
    for name, payload in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        meta = handler.extract(make_source(path, Path(name)))
        meta["relative_path"] = name
        metas.append(meta)
    return metas, [f"file_{i}" for i in range(len(metas))]


def build(handler: ImageHandler, directory: Path, files: dict):
    return handler.build_croissant(*described(handler, directory, files))


def nodes_by_name(nodes) -> dict:
    return {node.name: node for node in nodes}


def fields_of(record_set) -> dict:
    return {field["name"]: field for field in record_set.to_json().get("field", [])}


def as_json(result) -> str:
    """Everything the handler contributes to the document, as one string."""
    import json

    return json.dumps(
        [node.to_json() for node in (*result.file_sets, *result.record_sets)],
        ensure_ascii=False,
    )


def test_a_batch_with_no_ome_file_describes_one_collection(
    handler: ImageHandler, dataset: Path
) -> None:
    """The gate on every corpus already committed: no OME file, no change."""
    result = build(
        handler, dataset, {"a.png": PNG_1X1, "b.tif": PLAIN_TIFF, "c.btf": BIGTIFF}
    )

    assert [fs.id for fs in result.file_sets] == ["image-files"]
    assert [rs.name for rs in result.record_sets] == ["images"]
    assert sorted(result.file_sets[0].includes) == ["**/*.btf", "**/*.png", "**/*.tif"]


def test_ome_files_are_described_as_their_own_collection(
    handler: ImageHandler, dataset: Path
) -> None:
    """OME fields on the shared record set would land on every PNG beside them."""
    result = build(handler, dataset, {"a.ome.tif": OME_TIFF, "b.png": PNG_1X1})

    record_sets = nodes_by_name(result.record_sets)
    assert sorted(record_sets) == ["images", "ome_images"]
    assert sorted(fs.id for fs in result.file_sets) == [
        "image-files",
        "ome-image-files",
    ]
    assert list(fields_of(record_sets["images"])) == ["image"]
    assert nodes_by_name(result.file_sets)["OME-TIFF files"].includes == ["a.ome.tif"]


def test_the_ome_file_set_lists_its_files_rather_than_globbing(
    handler: ImageHandler, dataset: Path
) -> None:
    """A ``**/*.tif`` glob would re-admit the plain TIFF beside it."""
    result = build(handler, dataset, {"a.ome.tif": OME_TIFF, "plain.tif": PLAIN_TIFF})

    ome_files = nodes_by_name(result.file_sets)["OME-TIFF files"]
    assert ome_files.includes == ["a.ome.tif"]
    assert not any("*" in pattern for pattern in ome_files.includes)


def resolve(includes, directory: Path) -> set:
    found = set()
    for pattern in includes:
        if "*" in pattern:
            found |= {str(p.relative_to(directory)) for p in directory.glob(pattern)}
        else:
            found.add(pattern)
    return found


def test_the_two_collections_partition_the_batch(
    handler: ImageHandler, dataset: Path
) -> None:
    """The OME files leave ``images``, which is an existing public record set,
    so every image must still be in exactly one collection."""
    files = {
        "a.ome.tif": OME_TIFF,
        "b.ome.tif": OME_40,
        "plain.tif": PLAIN_TIFF,
        "photo.png": PNG_1X1,
        "tissue.btf": BIGTIFF,
    }

    result = build(handler, dataset, files)

    plain, ome_files = (
        resolve(nodes_by_name(result.file_sets)[name].includes, dataset)
        for name in ("Image files", "OME-TIFF files")
    )
    assert plain & ome_files == set()
    assert plain | ome_files == set(files)
    assert ome_files == {"a.ome.tif", "b.ome.tif"}


def test_an_extension_no_ome_file_uses_keeps_its_glob(
    handler: ImageHandler, dataset: Path
) -> None:
    """Listing every PNG in a photo archive beside one OME file would be a
    FileSet the length of the dataset."""
    result = build(handler, dataset, {"a.ome.tif": OME_TIFF, "photo.png": PNG_1X1})

    assert nodes_by_name(result.file_sets)["Image files"].includes == ["**/*.png"]


def test_every_field_is_typed_and_only_the_image_field_extracts(
    handler: ImageHandler, dataset: Path
) -> None:
    """mlcroissant's ``fileProperty: content`` for ``image/tiff`` is the decoded
    pixels, so putting that extract on ``size_c`` would ask a consumer to cast
    an image to an integer."""
    result = build(handler, dataset, {"a.ome.tif": OME_TIFF})

    fields = fields_of(nodes_by_name(result.record_sets)["ome_images"])
    # mlcroissant hands a dataType back as an rdflib URIRef, which does not
    # compare equal to a plain str from the left.
    assert str(fields["image"]["dataType"]) == "sc:ImageObject"
    assert str(fields["size_c"]["dataType"]) == "sc:Integer"
    assert str(fields["physical_size_x"]["dataType"]) == "sc:Float"
    assert str(fields["dimension_order"]["dataType"]) == "sc:Text"
    assert fields["channel_names"]["cr:isArray"] is True
    assert fields["channel_names"]["cr:arrayShape"] == "-1"

    assert fields["image"]["source"]["extract"] == {"fileProperty": "content"}
    for name, field in fields.items():
        assert "value" not in field, name
        assert field["source"]["fileSet"] == {"@id": "ome-image-files"}
        if name != "image":
            assert "extract" not in field["source"], name


def test_what_the_batch_observed_reaches_each_field_description(
    handler: ImageHandler, dataset: Path
) -> None:
    """No ``Field.value`` anywhere: the per-file numbers stay in the files, and
    only the aggregate over the batch reaches the document."""
    result = build(handler, dataset, {"a.ome.tif": OME_TIFF})

    fields = fields_of(nodes_by_name(result.record_sets)["ome_images"])
    assert "3" in fields["size_c"]["description"]
    assert "µm" in fields["physical_size_unit"]["description"]
    assert "0.2125" in fields["physical_size_x"]["description"]
    assert "uint16" in fields["pixel_type"]["description"]
    assert "XYCZT" in fields["dimension_order"]["description"]
    assert "2016-06" in fields["ome_version"]["description"]


def test_files_that_disagree_are_reported_as_a_range(
    handler: ImageHandler, dataset: Path
) -> None:
    """One shared field describes the whole batch, so one file's value would be
    a false statement about the others."""
    result = build(handler, dataset, {"a.ome.tif": OME_TIFF, "b.ome.tif": OME_40})

    fields = fields_of(nodes_by_name(result.record_sets)["ome_images"])
    assert "3-40" in fields["size_c"]["description"]
    assert "uint16" in fields["pixel_type"]["description"]
    assert "uint8" in fields["pixel_type"]["description"]


def test_a_field_no_file_declares_is_not_emitted(
    handler: ImageHandler, dataset: Path
) -> None:
    """``PhysicalSizeX`` is optional in the schema, and a field naming
    something no file declares is noise."""
    result = build(handler, dataset, {"a.ome.tif": OME_NO_PHYSICAL_SIZE})

    fields = fields_of(nodes_by_name(result.record_sets)["ome_images"])
    assert "physical_size_x" not in fields
    assert "physical_size_unit" not in fields
    assert "size_c" in fields


def test_the_record_set_says_its_rows_are_files(
    handler: ImageHandler, dataset: Path
) -> None:
    """A FileSet yields one record per file, and one OME-XML document may
    declare several images. The count makes the gap visible instead of leaving
    a consumer to assume rows are images."""
    result = build(handler, dataset, {"a.ome.tif": OME_TWO_IMAGES})

    record_set = nodes_by_name(result.record_sets)["ome_images"]
    fields = fields_of(record_set)
    assert "ome_image_count" in fields
    assert "2" in fields["ome_image_count"]["description"]
    assert "file" in record_set.description.lower()
    assert "Image[0]" in record_set.description


def test_a_multi_file_ome_set_is_counted_as_files(
    handler: ImageHandler, dataset: Path
) -> None:
    """Grouping the files of one logical image is a separate change. Reporting
    two rows for what may be one image, without saying so, is not."""
    result = build(
        handler,
        dataset,
        {"a.ome.tif": ome_partner("b.ome.tif"), "b.ome.tif": ome_partner("a.ome.tif")},
    )

    record_set = nodes_by_name(result.record_sets)["ome_images"]
    assert "2 OME-TIFF file" in record_set.description
    assert "Image[0]" in record_set.description


def test_channel_names_are_the_only_vocabulary_that_reaches_the_document(
    handler: ImageHandler, dataset: Path
) -> None:
    """The schema defines ``Channel/@Name`` as an acquisition channel's label,
    so it names an antibody or a fluorophore. It says nothing about
    ``Image/@Name``, which in practice holds slide labels and operator notes."""
    result = build(handler, dataset, {"a.ome.tif": OME_NAMED})

    fields = fields_of(nodes_by_name(result.record_sets)["ome_images"])
    channels = fields["channel_names"]["description"]
    document = as_json(result)
    for name in ("DAPI", "ATP1A1", "18S"):
        assert name in channels
        assert document.count(name) == 1, f"{name} reached a node of its own"
    for secret in ("Patient 3 slide 2", "Acme Scanner 4.2", "9c1bde0e-dead-beef"):
        assert secret not in document


def test_a_refused_description_is_counted_and_never_expanded(
    handler: ImageHandler, dataset: Path
) -> None:
    """``ScanEntry.describe()`` clears the reason and the detail, so a described
    file has nowhere else to record a partial refusal."""
    result = build(handler, dataset, {"a.ome.tif": BOMB_TIFF, "b.ome.tif": OME_TIFF})

    record_set = nodes_by_name(result.record_sets)["ome_images"]
    fields = fields_of(record_set)
    assert "1 of 2" in record_set.description
    assert "not parsed" in record_set.description
    # The refused file contributed nothing, and the sound one still did.
    assert fields["size_c"]["description"].endswith("(3)")
    assert "lol" not in as_json(result)


def test_an_image_j_description_leaves_the_file_a_plain_tiff(
    handler: ImageHandler, dataset: Path
) -> None:
    """Tag 270 carries all sorts of things. Only OME-XML is OME."""
    result = build(handler, dataset, {"a.tif": IMAGEJ_TIFF, "b.tif": PLAIN_TIFF})

    assert [rs.name for rs in result.record_sets] == ["images"]
    assert [fs.id for fs in result.file_sets] == ["image-files"]


def test_a_binary_only_file_names_its_companion_and_declares_nothing(
    handler: ImageHandler, dataset: Path
) -> None:
    """The schema forbids a place-holder any other content, so it carries no
    header field — not even the zero images it declares, which is a fact about
    the stub rather than about the image the file holds."""
    result = build(handler, dataset, {"a.ome.tif": BINARY_ONLY_TIFF})

    record_set = nodes_by_name(result.record_sets)["ome_images"]
    assert "plate.companion.ome" in record_set.description
    assert list(fields_of(record_set)) == ["image"]


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
