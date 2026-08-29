"""Every handler describes a wrapped file exactly as it describes its twin."""

import ast
import copy
import gzip
from pathlib import Path

import pytest

from croissant_baker import compression
from croissant_baker.handlers.base_handler import InputKind
from croissant_baker.handlers.registry import (
    get_registered_handlers,
    register_all_handlers,
    select_handler,
)
from croissant_baker.scan import Reason

from tests.helpers import SAMPLES, bake, write_all, write_wrapped

_HANDLER_DIR = Path(__file__).parent.parent / "src" / "croissant_baker" / "handlers"


# --------------------------------------------------------------------------
# One fixture per handler, keyed by class name
# --------------------------------------------------------------------------


def _handlers():
    register_all_handlers()
    return get_registered_handlers()


def _formats(metadata: dict) -> list:
    """The media types on each FileObject, in document order."""
    out = []
    for node in metadata.get("distribution", []):
        if node.get("@type") != "cr:FileObject":
            continue
        value = node.get("encodingFormat")
        out.append(value if isinstance(value, list) else [value])
    return out


def _rename_ids(node, rename: dict):
    """Replace every FileObject identifier with the name of the file it describes."""
    if isinstance(node, dict):
        return {k: _rename_ids(v, rename) for k, v in node.items()}
    if isinstance(node, list):
        return [_rename_ids(v, rename) for v in node]
    return rename.get(node, node) if isinstance(node, str) else node


def _normalise(metadata: dict) -> dict:
    """The described document, with transport differences removed and nothing else."""
    doc = copy.deepcopy(metadata)
    wrapper_types = {c.media_type for c in compression.compressions()}

    rename = {
        node["@id"]: f"file:{compression.logical_name(node['contentUrl'])}"
        for node in doc.get("distribution", [])
        if node.get("@type") == "cr:FileObject"
    }
    doc = _rename_ids(doc, rename)

    distribution = []
    for node in doc.get("distribution", []):
        formats = node.get("encodingFormat")
        formats = formats if isinstance(formats, list) else [formats]
        node["encodingFormat"] = [f for f in formats if f not in wrapper_types]
        if node.get("@type") == "cr:FileObject":
            node["name"] = compression.logical_name(node["name"])
            node["contentUrl"] = compression.logical_name(node["contentUrl"])
            node.pop("contentSize", None)
            node.pop("sha256", None)
        else:
            node["includes"] = _logical_includes(node.get("includes"))
        distribution.append(node)

    distribution.sort(key=lambda n: (n.get("@type", ""), n.get("@id", "")))
    # Record sets follow rglob order too, which differs between two freshly
    # created directories holding the same files.
    record_sets = sorted(doc.get("recordSet", []), key=lambda n: n.get("@id", ""))
    out = {"recordSet": record_sets, "distribution": distribution}
    _logicalise_descriptions(out)
    return out


def _logicalise_descriptions(node) -> None:
    """Reduce every filename inside a ``description`` to its logical form."""
    if isinstance(node, dict):
        text = node.get("description")
        if isinstance(text, str):
            node["description"] = " ".join(
                compression.logical_name(word) for word in text.split(" ")
            )
        for value in node.values():
            _logicalise_descriptions(value)
    elif isinstance(node, list):
        for item in node:
            _logicalise_descriptions(item)


def _logical_includes(includes) -> list:
    """Reduce a FileSet's includes to the form a plain dataset would emit."""
    patterns = includes if isinstance(includes, list) else [includes]
    reduced = {compression.logical_name(pattern) for pattern in patterns}
    return sorted(reduced)


# --------------------------------------------------------------------------
# A wrapped file is described exactly as the plain one is
# --------------------------------------------------------------------------


@pytest.mark.parametrize("handler_name", sorted(SAMPLES))
@pytest.mark.parametrize("suffix", ["", ".gz", ".bz2", ".xz"])
def test_wrapped_is_described_like_plain(
    handler_name: str, suffix: str, tmp_path: Path
) -> None:
    files = SAMPLES[handler_name]()

    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    write_all(plain_dir, files, "")

    wrapped_dir = tmp_path / "wrapped"
    wrapped_dir.mkdir()
    write_all(wrapped_dir, files, suffix)

    plain, wrapped = bake(plain_dir), bake(wrapped_dir)

    assert _normalise(wrapped) == _normalise(plain), (
        f"{handler_name}: {suffix} described differently from the plain files"
    )

    # The media types are the one thing that legitimately differs, so they are
    # asserted rather than normalised away. A handler fusing the two — the
    # application/x-nifti+gzip bug — fails here even if it never writes a
    # suffix literal for the grep check to find.
    expected = _formats(plain)
    if suffix:
        wrapper = next(c for c in compression.compressions() if c.suffix == suffix)
        expected = [formats + [wrapper.media_type] for formats in expected]
    assert _formats(wrapped) == expected, (
        f"{handler_name}: {suffix} reported {_formats(wrapped)}"
    )

    _assert_every_stored_file_is_covered(wrapped, wrapped_dir)


def _assert_every_stored_file_is_covered(metadata: dict, directory: Path) -> None:
    """A described file must be matched by an include in some FileSet."""
    file_sets = [n for n in metadata["distribution"] if n.get("@type") == "cr:FileSet"]
    if not file_sets:
        return  # this handler describes per-file, with no FileSet
    patterns = [
        p
        for fs in file_sets
        for p in (
            fs["includes"] if isinstance(fs["includes"], list) else [fs["includes"]]
        )
    ]
    for stored in sorted(directory.rglob("*")):
        if not stored.is_file():
            continue
        rel = stored.relative_to(directory)
        assert any(_matches(rel, pattern) for pattern in patterns), (
            f"{rel} is described but matched by none of {patterns}"
        )


def _matches(relative: Path, pattern: str) -> bool:
    """Whether ``relative`` is covered by a Croissant ``includes`` pattern."""
    return relative.match(pattern) or (
        pattern.startswith("**/") and relative.match(pattern[3:])
    )


# --------------------------------------------------------------------------
# A new compression needs no handler edit
# --------------------------------------------------------------------------


@pytest.fixture
def fauxzip():
    """A fourth compression, registered for one test. gzip bytes, new suffix."""
    extra = compression.Compression(
        "fauxzip", ".fz", "application/x-fauxzip", gzip.open
    )
    compression.register_compression(extra)
    yield extra


@pytest.mark.parametrize("handler_name", sorted(SAMPLES))
def test_a_newly_registered_compression_works_for_every_handler(
    handler_name: str, fauxzip, tmp_path: Path
) -> None:
    """No handler was edited to make this pass."""
    files = SAMPLES[handler_name]()

    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    write_all(plain_dir, files, "")

    wrapped_dir = tmp_path / "wrapped"
    wrapped_dir.mkdir()
    write_all(wrapped_dir, files, ".fz")

    assert _normalise(bake(wrapped_dir)) == _normalise(bake(plain_dir))


def test_a_newly_registered_compression_reaches_the_media_types(
    fauxzip, tmp_path: Path
) -> None:
    write_all(tmp_path, SAMPLES["CSVHandler"](), ".fz")

    entry = next(
        d for d in bake(tmp_path)["distribution"] if d.get("@type") == "cr:FileObject"
    )
    assert entry["encodingFormat"] == ["text/csv", "application/x-fauxzip"]


# --------------------------------------------------------------------------
# WFDB: exempt, and reported rather than silently skipped
# --------------------------------------------------------------------------


def test_a_compressed_wfdb_header_is_refused_with_its_own_reason(
    tmp_path: Path,
) -> None:
    write_wrapped(tmp_path, "record.hea", b"record 1 1 10\nrecord.dat 16 200\n", ".gz")

    selection = select_handler(tmp_path / "record.hea.gz")

    assert selection.handler is None
    assert "gzip" in selection.refusal
    assert "on disk" in selection.refusal
    assert selection.refusal != "no handler for this file type"


def test_a_plain_wfdb_header_is_still_claimed(tmp_path: Path) -> None:
    (tmp_path / "record.hea").write_text("record 1 1 10\nrecord.dat 16 200\n")
    assert type(select_handler(tmp_path / "record.hea").handler).__name__ == (
        "WFDBHandler"
    )


def test_a_newly_registered_compression_reaches_the_cli(
    fauxzip, tmp_path: Path
) -> None:
    """The CLI resolves compression the same way the generator does."""
    from typer.testing import CliRunner

    from croissant_baker.__main__ import app

    write_all(tmp_path, SAMPLES["CSVHandler"](), ".fz")

    result = CliRunner().invoke(
        app,
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.jsonld"),
            "--creator",
            "Tester",
            "--no-validate",
            "--count-csv-rows",
        ],
    )

    assert result.exit_code == 0
    assert "no CSV files found" not in result.stderr


# --------------------------------------------------------------------------
# Archives: reported, never opened, and composing with every compression
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "bundle.zip",
        "bundle.tar",
        "bundle.tgz",
        "bundle.tar.gz",
        "bundle.tar.bz2",
        "bundle.tar.xz",
    ],
)
def test_every_archive_spelling_is_recognised(name: str) -> None:
    assert compression.is_archive(name)


def test_an_archive_composes_with_a_runtime_compression(fauxzip) -> None:
    """A new compression must not need adding to the archive list as well."""
    assert compression.is_archive("bundle.tar.fz")


def test_an_archive_alias_survives_a_compression_claiming_its_suffix() -> None:
    """.tgz is its own archive suffix, not a wrapper to strip first."""
    extra = compression.Compression("tgzip", ".tgz", "application/x-tgzip", gzip.open)
    compression.register_compression(extra)
    assert compression.is_archive("bundle.tgz")


@pytest.mark.parametrize("name", ["bundle.zip", "bundle.tgz", "bundle.tar.xz"])
def test_an_archive_is_refused_with_the_archive_reason(name: str, tmp_path) -> None:
    (tmp_path / name).write_bytes(b"PK\x03\x04 not really")

    selection = select_handler(tmp_path / name)

    assert selection.handler is None
    assert selection.reason is Reason.ARCHIVE


# --------------------------------------------------------------------------
# Compression knowledge lives outside the handlers
# --------------------------------------------------------------------------


def _handler_modules() -> list:
    return sorted(_HANDLER_DIR.glob("*_handler.py"))


def test_no_format_handler_imports_compression_or_a_codec() -> None:
    """A tripwire over the obvious spellings, not a proof."""
    forbidden = (
        "from croissant_baker import compression",
        "from croissant_baker.compression",
        "import croissant_baker.compression",
        "import gzip",
        "import bz2",
        "import lzma",
    )
    offenders = [
        f"{path.name}: {line.strip()}"
        for path in _handler_modules()
        for line in path.read_text().splitlines()
        if any(line.strip().startswith(f) for f in forbidden)
    ]
    assert not offenders, offenders


def _executable_strings(path: Path):
    """Every string literal in a module except the docstrings."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.lineno, node.value


def test_no_handler_branches_on_a_wrapper_suffix_or_media_type() -> None:
    """A dead ``.json.gz`` branch is how a reader learns the wrong contract."""
    literals = {c.suffix for c in compression.compressions()}
    literals |= {c.media_type for c in compression.compressions()}

    offenders = [
        f"{path.name}:{lineno}: {value!r}"
        for path in _handler_modules()
        for lineno, value in _executable_strings(path)
        for literal in literals
        if literal in value
    ]
    assert not offenders, offenders


#: Methods that exist only on ``Path``, so a call to one is a handler reaching
#: the filesystem. ``.open()`` is deliberately absent: ``Image.open(stream)``
#: and ``TiffFile(stream)`` take the stream the source already decompressed,
#: and no static rule separates those from ``path.open()`` without guessing.
_PATH_ONLY_READS = frozenset({"read_bytes", "read_text"})

#: Module-level functions that open a path.
_PATH_OPENERS = frozenset({"open"})


def _reads_outside_the_source(node: ast.AST) -> bool:
    """Whether ``node`` is a call that gets bytes behind the source's back."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _PATH_OPENERS
    if isinstance(func, ast.Attribute):
        if func.attr in _PATH_ONLY_READS:
            return True
        # io.open / os.open, the aliases for the builtin.
        return (
            func.attr in _PATH_OPENERS
            and isinstance(func.value, ast.Name)
            and (func.value.id in ("io", "os"))
        )
    return False


def test_no_stream_handler_reads_bytes_the_pipeline_did_not_resolve() -> None:
    """Opening a path is how a handler gets bytes behind the source's back."""
    streaming = {
        f"{type(h).__module__.rsplit('.', 1)[-1]}.py"
        for h in _handlers()
        if h.INPUT_KIND is InputKind.STREAM
    }
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in _handler_modules()
        if path.name in streaming
        for node in ast.walk(ast.parse(path.read_text()))
        if _reads_outside_the_source(node)
    ]
    assert not offenders, (
        f"a STREAM handler reads bytes itself at {offenders}; "
        "read through source.open() / source.open_text() instead"
    )


def test_no_handler_declares_a_compound_extension() -> None:
    """EXTENSIONS carry format suffixes only; compression is stripped first."""
    for handler in _handlers():
        for ext in handler.EXTENSIONS:
            assert compression.compression_for(ext) is None, (
                f"{type(handler).__name__}.EXTENSIONS contains {ext!r}"
            )
