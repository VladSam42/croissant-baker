# Supported Formats

croissant-baker detects file types automatically. Handlers are checked in the order listed below — the first match wins.

Nothing is skipped in silence. A file no handler describes is counted in the run's coverage summary with the reason it was not, named individually under `--verbose`, and recorded in the machine-readable `--report` file:

```text
Scanned 8 file(s): 5 described, 3 not described.
  no registered handler: 1
  archive, not opened: 1
  duplicate by naming convention: 1
```

`--report FILE` writes one JSON object per run. `outcome` and `reason` come from
fixed vocabularies, so a tool can branch on them without parsing English:

```json
{
  "total": 8,
  "described": 5,
  "undescribed": 3,
  "by_reason": {"no_handler": 1, "archive": 1, "duplicate_by_name": 1},
  "files": [
    {"path": "notes.txt", "outcome": "unclaimed", "reason": "no_handler",
     "detail": "no registered handler claims this file"},
    {"path": "sample.csv.gz", "outcome": "linked", "reason": "duplicate_by_name",
     "detail": "same logical name as sample.csv; linked by naming convention, content not verified",
     "duplicate_of": "sample.csv"}
  ]
}
```

`outcome` is one of `described`, `linked`, `unclaimed`, `failed`, or
`would_process` (`--dry-run` only). `reason` is one of `no_handler`, `archive`,
`unsupported_input`, `claim_failed`, `extract_failed`, `build_failed`,
`duplicate_by_name`, or `probable_duplicate`, and is absent on `described`.

## File types

--8<-- "_generated/formats-table.md"

## Compression

`.gz`, `.bz2` and `.xz` are transport, not format. They are resolved before any handler is consulted, so every format in the table above is described the same way whether or not it arrived wrapped: `cells.parquet.gz` produces the record set that `cells.parquet` produces.

What the wrapper changes is the distribution entry, which addresses the bytes on disk. `encodingFormat` carries both media types, and `contentUrl`, `contentSize` and `sha256` are of the file as stored:

```json
{
  "@type": "cr:FileObject",
  "name": "cells.parquet.gz",
  "contentUrl": "cells.parquet.gz",
  "encodingFormat": ["application/vnd.apache.parquet", "application/gzip"]
}
```

Record set identifiers are derived from the logical file — `cells.parquet` — so a compressed file and its plain twin describe one table rather than two. Descriptions name the file as stored (`Records from cells.parquet.gz`), because a description has to name something you can find on disk. FileSet `includes` are resolved against the files actually present, so a compressed file is covered by the FileSet describing it.

WFDB is the exception. A WFDB record is a `.hea` header read together with its sibling `.dat` and `.atr` files, so no single stream carries it; a compressed `.hea` is reported with that as its reason rather than described.

`.zip`, `.tar` and `.tgz` hold several members. They are reported as archives and not opened.

Registering another compression is a library call and needs no handler change:

```python
import zstandard
from croissant_baker import compression

compression.register_compression(
    compression.Compression("zstd", ".zst", "application/zstd", zstandard.open)
)
```

## Duplicate files

Two files in one directory can describe the same data, which would otherwise put two record sets under one identifier and abort the bake. One is described and the others link to it with `sameAs`, keeping their own distribution entries — their bytes, sizes and checksums are their own.

| Shape | How it is decided |
|-------|-------------------|
| `sample.csv` and `sample.csv.gz` | Linked on the naming convention. Contents are not compared, and the reason says so |
| `sample.csv.gz` and `sample.csv.xz` | Linked only if the first 64 KiB decompress identically |
| `sample.csv` and `sample.tsv` | Two plain files of different size are rejected outright; otherwise the same 64 KiB comparison. If they differ, both are described under distinct identifiers (`sample_csv`, `sample_tsv`) |

The file that keeps its structure is chosen deterministically: uncompressed first, then the order compressions were registered in, then the path.

## CSV and TSV

CSV and TSV files are read with PyArrow's streaming reader — memory is constant regardless of file size. Type inference runs in two passes: an initial sweep, then per-column promotion when the first pass hits a type conflict.

Row counts are omitted by default for speed. Pass `--count-csv-rows` to do a full scan for exact counts (slow on large datasets).

## FHIR (`.ndjson`, `.json` Bundle)

Two FHIR serialization formats are supported:

- **NDJSON bulk export** (`.ndjson`): one resource per line, all of the same `resourceType`. Produced by FHIR Bulk Data servers.
- **JSON Bundle** (`.json`): a FHIR Bundle whose `entry[]` may contain mixed resource types.

Field names and types are inferred from a sample of resources. `OperationOutcome` resources (error markers) are skipped.

!!! note
    FHIR `.json` files are detected by content — the handler looks for `"resourceType": "<UpperCase…"` before accepting. Plain JSON files that happen to use `.json` are handled by the JSON handler instead.

## JSON and JSONL

- **JSON** (`.json`): an array of objects (one object per record) or a single object (treated as one record).
- **JSONL** (`.jsonl`): newline-delimited JSON, one object per line.

Schema is inferred from a sample of records. FHIR `.json` files are excluded — they go to the FHIR handler.

## Parquet

Schema is read from Parquet metadata only — the file data is never loaded. Partitioned datasets (a directory containing two or more `.parquet` files) are grouped into a single logical `cr:FileSet` and `cr:RecordSet`.

## WFDB

WFDB (WaveForm DataBase) is the standard physiological signal format on PhysioNet. The handler reads the `.hea` header file and records signal channel names, sampling frequency, number of samples, and duration. Associated `.dat` binary files are listed as related files.

Because a record spans several files located by path, this is the one handler that needs a real file on disk. A compressed `.hea` is reported rather than described.

## Images

Standard images are read with Pillow. Multi-band or scientific TIFFs fall back to `tifffile`. All images in a dataset are grouped into one `cr:FileSet` with a single summary `cr:RecordSet` covering width, height, color mode, and encoding format.

## DICOM

DICOM (`.dcm`, `.dicom`) is the standard format for medical imaging (CT, MRI, PET, etc.). The handler uses `pydicom` with `stop_before_pixels=True` — only the file header is read, so large pixel arrays are never loaded into memory.

Extracted metadata: image dimensions (rows, columns), number of frames, bits allocated per pixel, photometric interpretation, pixel spacing, slice thickness, modality, study/series description, manufacturer, and SOP class UID.

Files with no extension are also accepted if they carry the DICOM magic bytes (`DICM` at byte offset 128), which is common in PACS exports.

All DICOM files in a dataset are grouped into one `cr:FileSet` with a summary `cr:RecordSet` covering modality counts and dimension ranges.

## NIfTI

NIfTI (`.nii`) is the standard format for neuroimaging data (structural MRI, fMRI, CT). The handler uses `nibabel` and reads the header only — the voxel data array is never loaded.

Extracted metadata: spatial dimensions (x, y, z), number of timepoints for 4D volumes, voxel spacing in mm, stored data type, NIfTI version (1 or 2), and repetition time (TR) for fMRI data.

All NIfTI files in a dataset are grouped into one `cr:FileSet` with a summary `cr:RecordSet`. The `tr_seconds` field is only added when at least one 4D volume is present.

## Hidden files and directories

Files inside hidden directories (any path component starting with `.`) are always skipped, and do not appear in the coverage report. Use `--include` and `--exclude` glob patterns to further control which files are processed.
