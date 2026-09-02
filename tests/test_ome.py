"""What an OME-TIFF's header says about itself, read from the XML alone."""

from __future__ import annotations

import io

import pytest
import tifffile

from croissant_baker.handlers import ome

from tests.helpers import (
    OME_NAMESPACE,
    ome_bomb as bomb,
    ome_image as image,
    ome_xml,
    tiff_bytes,
)


def read_bytes(data: bytes):
    """The header the handler would get for a TIFF holding ``data``."""
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        return ome.read(tif)


# --------------------------------------------------------------------------
# What the parser reads
# --------------------------------------------------------------------------


def test_every_pixels_attribute_reaches_the_header() -> None:
    header = ome.parse(ome_xml(image()))

    assert header is not None
    assert (header.size_c, header.size_z, header.size_t) == (3, 1, 1)
    assert header.dimension_order == "XYCZT"
    assert header.pixel_type == "uint16"
    assert (header.physical_size_x, header.physical_size_y) == (0.2125, 0.2125)
    assert header.physical_size_unit == "µm"
    assert header.channel_names == ("DAPI", "ATP1A1", "18S")
    assert header.image_count == 1
    assert header.refusal == ""


@pytest.mark.parametrize("version", ["2016-06", "2013-06"])
def test_the_schema_version_comes_from_the_root_element(version: str) -> None:
    """The namespace is versioned, so matching a constant would read one year
    of files and silently decline the rest."""
    namespace = f"http://www.openmicroscopy.org/Schemas/OME/{version}"

    header = ome.parse(ome_xml(image(), namespace=namespace))

    assert header is not None
    assert header.version == version
    assert header.size_c == 3


def test_a_root_that_is_not_ome_is_not_an_ome_header() -> None:
    """Well-formed XML in tag 270 is common — ImageJ, MetaSeries, Leica SCN."""
    assert ome.parse("<MetaData><plane/></MetaData>") is None


def test_an_absent_attribute_is_absent_rather_than_guessed() -> None:
    """``PhysicalSizeX`` is optional in the schema, and a default would be a
    made-up measurement."""
    header = ome.parse(
        ome_xml(image(pixels='DimensionOrder="XYCZT" Type="uint8" SizeC="1"'))
    )

    assert header is not None
    assert header.physical_size_x is None
    assert header.physical_size_y is None
    assert header.physical_size_unit is None
    assert header.size_z is None
    assert (header.size_c, header.pixel_type) == (1, "uint8")


def test_an_unreadable_number_is_dropped_rather_than_raised() -> None:
    """One malformed attribute costs that attribute, not the whole header."""
    header = ome.parse(
        ome_xml(image(pixels='SizeC="lots" SizeZ="2" PhysicalSizeX="wide"'))
    )

    assert header is not None
    assert header.size_c is None
    assert header.physical_size_x is None
    assert header.size_z == 2


def test_channel_names_keep_document_order_and_skip_the_unnamed() -> None:
    """``Name`` is optional, and a gap in the list would misalign the rest."""
    header = ome.parse(
        ome_xml(
            '<Image ID="Image:0"><Pixels ID="Pixels:0" SizeC="3">'
            '<Channel ID="Channel:0:0" Name="DAPI"/>'
            '<Channel ID="Channel:0:1"/>'
            '<Channel ID="Channel:0:2" Name="18S"/>'
            "</Pixels></Image>"
        )
    )

    assert header is not None
    assert header.channel_names == ("DAPI", "18S")


def test_the_pixels_fields_describe_the_first_image() -> None:
    """A multi-position acquisition declares several images in one file. One
    row per file means the row can only describe one of them, so it says which."""
    header = ome.parse(
        ome_xml(
            image()
            + image(
                identifier="Image:1",
                pixels='SizeC="40" Type="uint8"',
                channels=("CD3",),
            )
        )
    )

    assert header is not None
    assert header.image_count == 2
    assert header.size_c == 3
    assert header.pixel_type == "uint16"
    assert header.channel_names == ("DAPI", "ATP1A1", "18S")


def test_a_binary_only_file_names_its_companion() -> None:
    """The attribute is ``MetadataFile``. ``FileName`` is TiffData/UUID's, for
    the multi-file case, and reading the wrong one loses a valid name in
    silence."""
    header = ome.parse(
        ome_xml('<BinaryOnly UUID="urn:uuid:9c1b" MetadataFile="plate.companion.ome"/>')
    )

    assert header is not None
    assert header.binary_only is True
    assert header.companion == "plate.companion.ome"
    assert header.image_count == 0
    assert header.size_c is None


# --------------------------------------------------------------------------
# What the parser refuses
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "document"),
    [
        ("entity bomb", bomb(6)),
        ("bare doctype", f'<!DOCTYPE OME><OME xmlns="{OME_NAMESPACE}"/>'),
    ],
)
def test_a_declaration_is_refused_before_the_document_is_parsed(
    label: str, document: str
) -> None:
    """OME-XML carries no DTD — its root is ``<OME xmlns=…>`` behind at most an
    XML declaration — so refusing one loses nothing legitimate, and it does not
    depend on which Expat the user happens to have linked."""
    header = ome.parse(document)

    assert header is not None
    assert header.refusal
    assert header.size_c is None
    assert header.channel_names == ()


def test_the_entity_bomb_would_otherwise_have_expanded() -> None:
    """Guards the fixture, not the parser: were Expat to start refusing a
    six-level bomb, the test above would pass without the check it exists for."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(bomb(6))

    assert len(root[0].get("ID")) > 100_000


def test_malformed_xml_is_refused_rather_than_raised() -> None:
    header = ome.parse(f'<OME xmlns="{OME_NAMESPACE}"><Image')

    assert header is not None
    assert header.refusal
    assert header.size_c is None


def test_an_oversized_description_is_not_parsed(monkeypatch) -> None:
    """A high-content-screening plate's OME-XML reaches this size, and
    describing 384 wells is not what this handler is for."""

    def fail(_document: str):
        raise AssertionError("the document was parsed despite exceeding the cap")

    monkeypatch.setattr(ome.ET, "fromstring", fail)
    oversized = ome_xml(f"<!--{'x' * (ome.MAX_DESCRIPTION_BYTES + 1)}-->")

    header = read_bytes(tiff_bytes(oversized))

    assert header is not None
    assert header.refusal
    assert header.size_c is None


# --------------------------------------------------------------------------
# What a TIFF has to carry before any of it applies
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "description"),
    [
        ("no description at all", None),
        ("ImageJ", "ImageJ=1.53t\nimages=1\nslices=1\n"),
        ("XML that is not OME", "<MetaData><PlaneInfo/></MetaData>"),
    ],
)
def test_a_tiff_carrying_no_ome_xml_has_no_header(
    label: str, description: str | None
) -> None:
    """None, not a refusal: these files are described as plain TIFFs and
    nothing about them was declined."""
    assert read_bytes(tiff_bytes(description)) is None


def test_a_real_ome_tiff_is_read_through_the_tag() -> None:
    header = read_bytes(tiff_bytes(ome_xml(image()), planes=3))

    assert header is not None
    assert header.size_c == 3
    assert header.version == "2016-06"
