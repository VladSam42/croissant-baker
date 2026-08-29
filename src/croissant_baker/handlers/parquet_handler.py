"""Parquet file handler for tabular event streams (e.g., MEDS)."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

import mlcroissant as mlc
from pyarrow.parquet import ParquetFile

from croissant_baker.handlers.base_handler import FileTypeHandler
from croissant_baker.scan import Reason
from croissant_baker.sources import FileSource
from croissant_baker.handlers.utils import (
    _build_fields,
    _disambiguate_ids,
    display_name,
    get_clean_record_name,
    infer_column_types_from_arrow_schema,
    make_field_id,
    DIGIT_MASK,
    partition_template,
    sanitize_id,
)

logger = logging.getLogger(__name__)

# Apache Parquet file format spec: every Parquet file begins AND ends with
# the 4-byte ASCII magic "PAR1" — the trailing copy is the footer marker.
# A valid file is therefore at least 8 bytes long.
# Reference: https://parquet.apache.org/docs/file-format/
_PARQUET_MAGIC = b"PAR1"
_PARQUET_MAGIC_LEN = len(_PARQUET_MAGIC)


def _tail(stream, size: int) -> bytes:
    """Return the last ``size`` bytes of ``stream`` from its current position.

    Seeks from the end when the stream allows it. A stream that refuses — gzip
    in read mode — is read forward keeping a rolling tail: one pass, constant
    memory.
    """
    try:
        stream.seek(-size, 2)
        return stream.read(size)
    except (OSError, ValueError):
        tail = b""
        while chunk := stream.read(1 << 20):
            tail = (tail + chunk)[-size:]
        return tail


def _has_parquet_magic(source: FileSource) -> bool:
    """Return True iff ``source`` has the Parquet magic at start and end.

    Files that fail any check (too small, missing PAR1 header, missing PAR1
    footer) are rejected and a WARNING is logged with the specific reason so
    the user can see which files were skipped and why. Missing or unreadable
    files return False without logging (caller errors, not impostors). The
    footer check costs a full pass on a compressed file.
    """
    if not source.exists:
        return False
    if source.size < _PARQUET_MAGIC_LEN * 2:
        logger.warning(
            "Skipping %s: file is too small (%d bytes) to be a valid Parquet file",
            source.relative_path,
            source.size,
        )
        return False
    try:
        with source.open() as f:
            if f.read(_PARQUET_MAGIC_LEN) != _PARQUET_MAGIC:
                logger.warning(
                    "Skipping %s: missing Parquet PAR1 header magic",
                    source.relative_path,
                )
                return False
            if _tail(f, _PARQUET_MAGIC_LEN) != _PARQUET_MAGIC:
                logger.warning(
                    "Skipping %s: missing Parquet PAR1 footer magic "
                    "(file may be truncated)",
                    source.relative_path,
                )
                return False
            return True
    except OSError:
        return False


class _File(NamedTuple):
    index: int
    file_id: str
    meta: dict


@dataclass(frozen=True)
class _Table:
    """One logical table: the shards of a partition, or a standalone file."""

    dir_path: str
    files: list
    name: str
    stem: str
    disambig_parents: list
    is_partitioned: bool
    alone_in_dir: bool

    @property
    def first_meta(self) -> dict:
        return self.files[0].meta


def _schema_of(meta: dict) -> tuple:
    return tuple(meta["column_types"].items())


def _group_tables(file_metas: list, file_ids: list) -> tuple:
    """Split Parquet files into logical tables, and the shards that would not fit.

    Spark and Arrow write one table as a directory of ``part-*.parquet``; a
    vendor output directory holds several unrelated ones. Directory membership
    cannot tell those apart, so files group on evidence instead: the same
    schema, and the same name once digit runs are masked.
    """
    by_dir: dict = defaultdict(list)
    for index, (file_id, meta) in enumerate(zip(file_ids, file_metas)):
        by_dir[str(Path(meta["relative_path"]).parent)].append(
            _File(index, file_id, meta)
        )

    tables: list = []
    conflicts: list = []
    for dir_path in sorted(by_dir):
        dir_tables, dir_conflicts = _tables_in_dir(dir_path, by_dir[dir_path])
        tables.extend(dir_tables)
        conflicts.extend(dir_conflicts)
    return tables, conflicts


def _tables_in_dir(dir_path: str, files: list) -> tuple:
    """The tables one directory holds, and the shards it had to decline."""
    by_template: dict = defaultdict(list)
    for file in sorted(files, key=lambda f: str(f.meta["relative_path"])):
        by_template[partition_template(file.meta["file_name"])].append(file)

    groups: list = []
    conflicts: list = []
    for template in sorted(by_template):
        by_schema: dict = defaultdict(list)
        for file in by_template[template]:
            by_schema[_schema_of(file.meta)].append(file)

        if len(by_schema) > 1:
            kept, declined = _resolve_schema_conflict(by_schema)
            conflicts.extend(_describe_conflicts(kept, declined))
            shard_groups = [kept]
        else:
            shard_groups = list(by_schema.values())

        for shards in shard_groups:
            # The dataset root is not a table, so its own files never pair.
            if len(shards) >= 2 and dir_path != ".":
                groups.append((shards, True))
            else:
                groups.extend(([file], False) for file in shards)

    alone = len(groups) == 1
    return [_make_table(dir_path, g, p, alone) for g, p in groups], conflicts


def _resolve_schema_conflict(by_schema: dict) -> tuple:
    """The largest shard group wins; the rest are declined rather than merged."""
    ordered = sorted(
        by_schema.values(),
        key=lambda g: (-len(g), min(str(f.meta["relative_path"]) for f in g)),
    )
    return ordered[0], [file for group in ordered[1:] for file in group]


def _describe_conflicts(kept: list, declined: list) -> list:
    expected = len(kept[0].meta["column_types"])
    reference = kept[0].meta["file_name"]
    return [
        (
            file.index,
            Reason.PARTITION_SCHEMA_CONFLICT,
            f"{len(file.meta['column_types'])} columns, "
            f"expected {expected} from {reference}",
        )
        for file in declined
    ]


def _make_table(dir_path: str, files: list, partitioned: bool, alone: bool) -> _Table:
    meta = files[0].meta
    parents = list(Path(meta["relative_path"]).parts[:-1])

    # Parquet basenames are usually partition labels, so a directory holding one
    # table names it better than its own files do. Once it holds several, the
    # basenames are what tell them apart.
    if alone and parents:
        name, disambig_parents = parents[-1], parents[:-1]
    elif partitioned:
        name, disambig_parents = _shared_stem(meta["file_name"]), parents
    else:
        name, disambig_parents = get_clean_record_name(meta["file_name"]), parents

    return _Table(
        dir_path=dir_path,
        files=files,
        name=name,
        stem=sanitize_id(name),
        disambig_parents=disambig_parents,
        is_partitioned=partitioned,
        alone_in_dir=alone,
    )


def _shared_stem(file_name: str) -> str:
    """What a table's shards have in common, so two do not become ``part-00000``."""
    common = get_clean_record_name(partition_template(file_name))
    return common.replace(DIGIT_MASK, "").rstrip("-_.") or get_clean_record_name(
        file_name
    )


class ParquetHandler(FileTypeHandler):
    """
    Handler for Parquet files (.parquet) with schema-based type inference.

    - Uses pyarrow to read schema and row count without loading full data
    - Emits Croissant-compatible column types via shared map_arrow_type()
    - Computes SHA256 for reproducibility
    - Keeps memory usage minimal (schema-only)
    """

    EXTENSIONS = (".parquet",)
    FORMAT_NAME = "Parquet"
    FORMAT_DESCRIPTION = "Arrow schema, column names and types, row count"

    def claims(self, source: FileSource) -> bool:
        if source.suffix != ".parquet":
            return False
        return _has_parquet_magic(source)

    def extract(self, source: FileSource, **kwargs) -> dict:
        """Extract metadata from a Parquet file via pyarrow schema inspection."""
        if not source.exists:
            raise FileNotFoundError(f"Parquet file not found: {source.relative_path}")

        try:
            with source.open() as stream, ParquetFile(stream) as pq:
                schema = pq.schema_arrow
                num_rows = pq.metadata.num_rows if pq.metadata is not None else 0

                # Use the shared Arrow type mapper (same as CSV handler)
                column_types = infer_column_types_from_arrow_schema(schema)
                columns = [field.name for field in schema]

            return {
                "file_name": source.name,
                "file_size": source.size,
                "sha256": source.sha256,
                "encoding_format": "application/vnd.apache.parquet",
                "column_types": column_types,
                "arrow_schema": schema,
                "num_rows": num_rows,
                "num_columns": len(columns),
                "columns": columns,
            }
        except Exception as e:
            raise ValueError(
                f"Failed to process Parquet file {source.relative_path}: {e}"
            ) from e

    def build_croissant(self, file_metas: list, file_ids: list) -> tuple:
        """Build FileSets and RecordSets for all Parquet files in this dataset.

        Returns ``(file_sets, record_sets, conflicts)``, the last being
        ``(index, Reason, detail)`` per file this declined to describe.
        """
        tables, conflicts = _group_tables(file_metas, file_ids)
        rs_ids = _disambiguate_ids([(t.stem, t.disambig_parents) for t in tables])

        file_sets = []
        record_sets = []
        for table, rs_id in zip(tables, rs_ids):
            if table.is_partitioned:
                file_sets.append(self._partition_file_set(table, rs_id))
                record_sets.append(self._partition_record_set(table, rs_id))
            else:
                record_sets.append(self._standalone_record_set(table, rs_id))

        return file_sets, record_sets, conflicts

    def _partition_file_set(self, table: _Table, rs_id: str) -> mlc.FileSet:
        """One FileSet over a table's shards.

        The id follows the record set's, which is already disambiguated: two
        directories sharing a name would otherwise claim the same one.
        """
        if table.alone_in_dir:
            suffix = "".join(Path(table.first_meta["file_name"]).suffixes)
            includes = [f"{table.dir_path}/*{suffix}"]
        else:
            # A directory-wide glob would reach into the neighbouring tables.
            includes = [f.meta["relative_path"] for f in table.files]

        return mlc.FileSet(
            id=f"{rs_id}-fileset",
            name=f"{table.name} partition files",
            description=f"{len(table.files)} Parquet partition files for table '{table.name}'",
            encoding_formats=["application/vnd.apache.parquet"],
            includes=includes,
        )

    def _partition_record_set(self, table: _Table, rs_id: str) -> mlc.RecordSet:
        fields = self._fields(
            table.first_meta,
            rs_id,
            {"file_set": f"{rs_id}-fileset"},
            f"table '{table.name}'",
        )
        num_rows = sum(f.meta.get("num_rows", 0) for f in table.files)
        return mlc.RecordSet(
            id=rs_id,
            name=table.name,
            description=(
                f"Partitioned table '{table.name}' "
                f"({len(table.files)} Parquet files, {num_rows} total rows)"
            ),
            fields=fields,
        )

    def _standalone_record_set(self, table: _Table, rs_id: str) -> mlc.RecordSet:
        file = table.files[0]
        fields = self._fields(
            file.meta, rs_id, {"file_object": file.file_id}, display_name(file.meta)
        )
        num_rows = file.meta.get("num_rows")
        row_desc = f" ({num_rows} rows)" if num_rows is not None else ""
        return mlc.RecordSet(
            id=rs_id,
            name=table.name,
            description=f"Records from {display_name(file.meta)}{row_desc}",
            fields=fields,
        )

    def _fields(self, meta: dict, rs_id: str, source: dict, origin: str) -> list:
        if "arrow_schema" in meta:
            return _build_fields(meta["arrow_schema"], rs_id, source)

        used_field_ids: set = set()
        fields = []
        for col_name, col_type in meta["column_types"].items():
            fields.append(
                mlc.Field(
                    id=make_field_id(rs_id, col_name, used_field_ids),
                    name=col_name,
                    description=f"Column '{col_name}' from {origin}",
                    data_types=[col_type],
                    source=mlc.Source(**source, extract=mlc.Extract(column=col_name)),
                )
            )
        return fields
