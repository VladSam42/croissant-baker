# Supported Formats

croissant-baker detects file types automatically. Handlers are checked in the order listed below — the first match wins.

Nothing is skipped in silence, and nothing is reported one line per file by default. The closing coverage summary counts every file the document does not carry, under the reason it was passed over. The summary is a fixed size — a header plus at most one line per reason — so a directory with one undescribed file and one with ten thousand print the same shape. The full list is under `--verbose`, or in the machine-readable `--report` file:

```text
Scanned 8 file(s): 4 described, 1 linked, 1 referenced, 2 not described.
  no handler: 1
  archive, not opened: 1
```

A file reaches the document three ways, and the header names each one that
happened. `described` means a handler read the file and built its record set.
`linked` means the file is another described file in a different form — a
`.csv.gz` beside its `.csv` — so its bytes are described and its structure is
its twin's. `referenced` means another file's handler put it there: a WFDB
header is read together with its `.dat` and `.atr`, and each gets a FileObject
though nothing claims a `.dat` on its own. Only `not described` counts files the
document does not carry, and the reason lines account for exactly those.

`--report FILE` writes one JSON object per run. `outcome` and `reason` come from
fixed vocabularies, so a tool can branch on them without parsing English:

```json
{
  "total": 8,
  "described": 4,
  "linked": 1,
  "referenced": 1,
  "undescribed": 2,
  "by_reason": {"no_handler": 1, "archive": 1},
  "files": [
    {"path": "notes.txt", "outcome": "unclaimed", "reason": "no_handler",
     "detail": "no handler for this file type"},
    {"path": "sample.csv.gz", "outcome": "linked", "reason": "duplicate_by_name",
     "detail": "same logical name as sample.csv; linked by naming convention, content not verified",
     "duplicate_of": "sample.csv"},
    {"path": "100.dat", "outcome": "referenced",
     "detail": "described as part of 100.hea", "part_of": "100.hea"}
  ]
}
```

`described`, `linked`, `referenced` and `undescribed` sum to `total`, so a tool
can check coverage without reading the file list. `by_reason` accounts for the
`undescribed` alone.

`outcome` is one of `described`, `linked`, `referenced`, `unclaimed`, `failed`,
or `would_process` (`--dry-run` only). `reason` is one of `no_handler`,
`archive`, `unsupported_input`, `claim_failed`, `extract_failed`,
`build_failed`, `partition_schema_conflict`, `duplicate_by_name`, or
`probable_duplicate`, and is absent on `described` and `referenced`.

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

Schema is read from Parquet metadata only — the file data is never loaded.

Spark and Arrow write one table as a directory of `part-*.parquet`, while a vendor export directory holds several unrelated tables. Directory membership cannot tell those apart, so two files are shards of one logical table — one `cr:FileSet` and one `cr:RecordSet` — only when both hold:

| Evidence | What counts |
|----------|-------------|
| A shard-shaped name | The names match once digit runs are masked, and at least one of those runs is a name component of its own: the whole stem (`0.parquet`) or introduced by `-`, `_` or `.` (`part-00000.parquet`). Digits fused to letters are part of a word, so `assay1.parquet` and `assay2.parquet` stay two tables |
| An identical Arrow schema | Compared as Arrow types, not as Croissant types, which collapse timestamp units, nullability, decimal precision and nested structure |

Anything else is described on its own, as CSV and JSON files are. Files at the dataset root never pair.

Where files agree on the name but not the schema, that is drift inside one table: the majority schema is described and the rest are reported with reason `partition_schema_conflict` rather than folded in. The `cr:FileSet` then lists its shards individually — a directory-wide glob would re-admit exactly the files that were kept out.

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

## HDF5 (`.h5`, `.h5ad`, `.hdf5`)

HDF5 is the default container for scientific array data, and the handler reads `h5py`. **Nothing but structure is read** — dataset paths, dtypes, shapes, and the named attributes a recognised layout defines — so describing a 5 GB file costs what describing a 5 MB one with the same structure costs. Attributes are asked for by name and never as a set, because an attribute holds a value: a Keras `model_config` or a MATLAB header can be megabytes.

A file is claimed on the HDF5 signature, which does not have to sit at offset 0: HDF5 permits a user block before the superblock, and MATLAB v7.3 puts a text header in a 512-byte one. The first 8 KiB are searched, at offsets 0 and 512·2ⁿ. A legal file behind a larger user block is reported as having no handler.

### What becomes a record set

Every file gets one of two views, never both.

| Layout | Recognised by | Record sets |
|--------|---------------|-------------|
| AnnData (`.h5ad`) | root `encoding-type: anndata`, or an `obs` and a `var` with no root `encoding-type` at all | `<stem>_obs`, `<stem>_var` |
| 10x feature-barcode matrix | a `matrix` group holding a `features` group and an `indptr` that declares a width | `<stem>_features`, `<stem>_barcodes` |
| 10x, Cell Ranger 2 | exactly one group holding `genes`, `gene_names`, `barcodes`, `data` and an `indptr` that declares a width, and no root `filetype` | `<stem>_genes`, `<stem>_barcodes` |
| anything else | — | `<stem>`, one field per leaf dataset |

Nothing is refused. A Keras model, a NetCDF4 file, a MATLAB v7.3 session, an NWB recording and a BigDataViewer volume all fall to the generic view and are described by their datasets — and so does anything that only partly matches a layout: a `matrix` group missing its `features`, either 10x shape whose `indptr` says nothing about how many barcodes there are, a Cell Ranger 2 file that also declares a `filetype`, or a barnyard run whose two genome groups mean neither can be read as *the* one. A partial match is described for what it holds rather than claimed as a layout, since a field standing in for an absent array would name something that is not there.

### How to find a field in the file

**Every field's description states its HDF5 path**, so nothing has to be inferred from a naming convention:

```text
Column 'cell_type' at /obs/cell_type in integrated.h5ad
Array /obsm/X_pca, indexed by the obs axis, in integrated.h5ad
Column 'genes' at /GRCh38/genes in filtered_gene_bc_matrices_h5.h5
HDF5 dataset /model_weights/dense/kernel:0 in model.h5
Member 'age' of the record array /obs in legacy.h5ad
```

A column name is not enough on its own, which is why the path is spelled out. `genes` on a Cell Ranger 2 file lives under a group named for the reference genome, and the record-set identifier is derived from the file name, so nothing else in the document would record `GRCh38`. An array is a second case: `X` is indexed by the observation axis but stored at the root, not under `/obs`. Each table's record set also names the group its columns are in (`columns at /matrix/features`).

In the generic view the field's `name` is itself the path. In a recognised layout the `name` is the column, as it is for a CSV.

**No field carries an `extract`.** mlcroissant validates one and cannot execute it: its reader dispatches on `encodingFormat` over CSV, TSV, JSON, Parquet, text, image, audio, video, DICOM and archive types, and has no HDF5 reader at any media type. A field claiming a readable column would be a promise nobody can keep, and the validator would not catch it. The record sets are descriptive, as this tool's Parquet, JSONL and DICOM record sets already are.

### Types, and where they come from

For a recognised layout the AnnData `encoding-type` decides, not the dtype of the object carrying it. A `categorical` is typed from its `categories` and never from its `codes` — the codes are `int8` below 127 categories and `int32` at 40 000, so their width is the encoding rather than the type. A `nullable-integer` is typed from its `values` and not its `mask`. Pre-spec files are read too: `obs` as a compound-dtype dataset gives its members as columns, and an integer column beside `uns/<column>_categories` is a categorical.

Elsewhere the dtype maps directly, with two cases worth naming. A fixed-length string reports `sc:Text`, which loses the byte width. An object or region reference also reports `sc:Text`, because Croissant has nothing better, but its description says it is a reference — `dtype.kind` alone cannot tell one from a variable-length string, and calling it text without saying so would invite a reader to expect labels where there are only pointers.

### Arrays attach to the axis that indexes them

`X` is *n_obs* × *n_var*, so on `<stem>_obs` it is one field of *n_var* values per row; `layers/*` and `obsm/*` join it, and `varm/*` go to `<stem>_var`. A 10x matrix is features × barcodes, so it attaches to `<stem>_features` with the barcode count as its shape. The generic view has no declared axes, so each dataset carries its full shape.

### What is not described

`uns`, `obsp`, `varp` and `raw` are named in the record-set description and not described: the first is arbitrary, the middle two are graphs over one axis rather than per-row features, and `raw` holds a second copy of `X` and `var`. Every recognised layout names its other top-level entries the same way, so nothing in a file goes unmentioned by both views.

The generic view describes at most 300 leaf datasets. When it runs out of room it says the cap was reached and that at least one further dataset is not described — not how many, because counting them means walking the whole file, which is the cost the cap exists to avoid. A file holding exactly 300 claims no omission.

An **external link is never followed**, including a valid one. It can name any HDF5 file on the machine, so following it would describe structure the dataset does not contain and record a path outside its root. The count of links not followed is in the description; the targets are not.

### No value from inside the file

Column names, dataset paths, dtypes, shapes and row counts only. A categorical's labels in particular do not appear: HDF5 guarantees nothing about what they describe, so nothing here can tell an assay vocabulary from a clinical one. Barcodes, feature ids, cell ids and library ids are record-level identifiers and never appear.

### What a wrapper costs

`.h5.gz` is described identically to `.h5`, but not as cheaply. h5py seeks backwards through the file, and a non-seekable codec pays for that by decompressing and discarding everything it skips.

Measured on a 120 MB `.h5ad` holding incompressible data, reading its structure took 2 ms uncompressed, 0.7 s gzipped, 6 s xz-wrapped and 7 s bz2-wrapped. That cost tracks the file's size rather than its structure — uncompressed does not — so a 5 GB `.h5ad.gz` is of the order of half a minute, and the same file bz2-wrapped is several minutes. Uncompressed is worth it for this format, and gzip is worth it over the other two.

## Hidden files and directories

Files inside hidden directories (any path component starting with `.`) are always skipped, and do not appear in the coverage report. Use `--include` and `--exclude` glob patterns to further control which files are processed.
