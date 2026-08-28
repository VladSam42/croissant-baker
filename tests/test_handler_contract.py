"""One contract, asserted over every built-in handler."""

from __future__ import annotations

from pathlib import Path

import pytest

from croissant_baker import compression
from croissant_baker.handlers.base_handler import (
    FileTypeHandler,
    InputKind,
    uses_legacy_claims,
    uses_legacy_extract,
)
from croissant_baker.handlers.registry import builtin_handlers, select_handler
from croissant_baker.sources import make_source

from tests.helpers import EXEMPT, SAMPLES, WRAPPER_SUFFIXES, write_wrapped

HANDLERS = {type(h).__name__: h for h in builtin_handlers()}
NAMES = sorted(HANDLERS)
SAMPLED = sorted(SAMPLES)

#: Extensions only one handler declares. ``.json`` is claimed by both FHIR and
#: JSON, so it cannot serve as a negative case for either.
EXCLUSIVE = {
    ext: names[0]
    for ext, names in (
        (
            ext,
            [n for n in NAMES if ext in HANDLERS[n].EXTENSIONS],
        )
        for ext in {e for h in HANDLERS.values() for e in h.EXTENSIONS}
    )
    if len(names) == 1
}


def source_for(handler: FileTypeHandler, path: Path, relative: Path | None = None):
    """The source the pipeline would hand this handler for ``path``."""
    return make_source(path, relative, with_path=handler.INPUT_KIND is InputKind.PATH)


def probe_name(name: str) -> str:
    """A filename carrying this handler's own extension."""
    if name in SAMPLES:
        return SAMPLES[name]()[0][0]
    return f"probe{HANDLERS[name].EXTENSIONS[0]}"


# --------------------------------------------------------------------------
# The sample table must keep up with the registry
# --------------------------------------------------------------------------


def test_every_handler_has_a_sample_or_a_recorded_exemption() -> None:
    """Adding a handler without test data fails here, not silently in the wild."""
    missing = [n for n in NAMES if n not in SAMPLES and n not in EXEMPT]
    assert not missing, (
        f"handlers with no sample: {missing}. Add one to tests.helpers.SAMPLES, "
        "or record in EXEMPT why it cannot take a stream."
    )


def test_only_non_stream_handlers_are_exempt() -> None:
    """An exemption has to be earned by the handler's declared input."""
    for name in EXEMPT:
        assert HANDLERS[name].INPUT_KIND is not InputKind.STREAM, (
            f"{name} reads a stream, so it has no excuse for an exemption"
        )


# --------------------------------------------------------------------------
# Claiming
# --------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", ["", *WRAPPER_SUFFIXES])
@pytest.mark.parametrize("name", SAMPLED)
def test_a_handler_claims_its_own_format_wrapped_or_not(
    name: str, suffix: str, dataset: Path
) -> None:
    handler = HANDLERS[name]
    for logical, payload in SAMPLES[name]():
        path = write_wrapped(dataset, logical, payload, suffix)
        assert handler.claims(source_for(handler, path, Path(logical))), (
            f"{name} did not claim its own {logical}{suffix}"
        )


@pytest.mark.parametrize("name", NAMES)
def test_a_handler_declines_another_handlers_exclusive_format(
    name: str, dataset: Path
) -> None:
    handler = HANDLERS[name]
    for ext, owner in EXCLUSIVE.items():
        if owner == name:
            continue
        sample = SAMPLES.get(owner)
        logical, payload = sample()[0] if sample else (f"other{ext}", b"payload")
        path = write_wrapped(dataset, logical, payload)
        assert not handler.claims(source_for(handler, path, Path(logical))), (
            f"{name} claimed {logical}, which belongs to {owner}"
        )


@pytest.mark.parametrize("name", SAMPLED)
def test_selection_routes_a_file_back_to_its_own_handler(
    name: str, dataset: Path
) -> None:
    logical, payload = SAMPLES[name]()[0]
    path = write_wrapped(dataset, logical, payload)
    selection = select_handler(path, Path(logical))
    assert type(selection.handler).__name__ == name


# --------------------------------------------------------------------------
# Failing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_a_missing_file_raises_file_not_found(name: str, tmp_path: Path) -> None:
    """One contract, so the pipeline reports one reason category."""
    handler = HANDLERS[name]
    logical = probe_name(name)
    with pytest.raises(FileNotFoundError):
        handler.extract(source_for(handler, tmp_path / "gone" / logical, Path(logical)))


@pytest.mark.parametrize("name", NAMES)
def test_garbage_bytes_raise_a_value_error_naming_the_file(
    name: str, dataset: Path
) -> None:
    """The message becomes the reason detail a user reads in ``--report``."""
    handler = HANDLERS[name]
    logical = probe_name(name)
    path = write_wrapped(dataset, logical, b"\x00\xff not a real file \xfe\x00")

    with pytest.raises(ValueError) as caught:
        handler.extract(source_for(handler, path, Path(logical)))

    assert logical in str(caught.value), (
        f"{name} raised {caught.value!r}, which does not name the file"
    )


# --------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_extensions_are_declared_and_carry_no_wrapper(name: str) -> None:
    handler = HANDLERS[name]
    assert handler.EXTENSIONS, f"{name} declares no EXTENSIONS"
    assert handler.FORMAT_NAME, f"{name} declares no FORMAT_NAME"
    for ext in handler.EXTENSIONS:
        assert ext.startswith("."), f"{name}.EXTENSIONS has {ext!r}"
        assert compression.compression_for(ext) is None, (
            f"{name}.EXTENSIONS contains the compound suffix {ext!r}; "
            "compression is stripped before a handler is asked"
        )


@pytest.mark.parametrize("name", NAMES)
def test_no_builtin_is_still_on_the_legacy_contract(name: str) -> None:
    handler = HANDLERS[name]
    assert not uses_legacy_claims(handler), f"{name} still implements can_handle"
    assert not uses_legacy_extract(handler), f"{name} still implements extract_metadata"
