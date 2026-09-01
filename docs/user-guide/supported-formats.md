# Supported Formats

croissant-baker detects file types automatically. Handlers are checked in the order listed below — the first match wins.

Nothing is skipped in silence. Each file that goes undescribed is named on the log as the run passes over it — so a long bake says so while it is still running — and counted in the closing coverage summary with the reason. Per-file warnings stop after the first 50, since past that the summary is the better account. The full list is under `--verbose`, or in the machine-readable `--report` file:

```text
Scanned 8 file(s): 5 described, 3 not described.
  no handler: 1
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
     "detail": "no handler for this file type"},
    {"path": "sample.csv.gz", "outcome": "linked", "reason": "duplicate_by_name",
     "detail": "same logical name as sample.csv; linked by naming convention, content not verified",
     "duplicate_of": "sample.csv"}
  ]
}
```

`outcome` is one of `described`, `linked`, `unclaimed`, `failed`, or
`would_process` (`--dry-run` only). `reason` is one of `no_handler`, `archive`,
`unsupported_input`, `claim_failed`, `extract_failed`, `build_failed`,
`partition_schema_conflict`, `duplicate_by_name`, or `probable_duplicate`, and
is absent on `described`.

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

Standard images (`.png`, `.jpg`, `.gif`, `.bmp`, `.webp`, `.ico`) are read with Pillow. Every TIFF — `.tif`, `.tiff` and the BigTIFF spelling `.btf` — is read with `tifffile`. Pillow opens some TIFFs too, but reports one band for a three-channel image, decodes the ImageDescription tag as latin-1 so `µm` comes back mojibake, and opens none of the twelve-band rasters this repository carries.

BigTIFF is not a separate format but classic TIFF's 64-bit offset field, which any writer switches to at 4 GiB — the size whole-slide imaging, electron-microscopy volumes and large rasters all cross. `image_format` stays `TIFF` for a `.btf` and `encodingFormat` stays `image/tiff`: BigTIFF has no registration of its own.

Images are grouped into one `cr:FileSet` and one summary `cr:RecordSet` covering width, height, band count and encoding format. Only the header is read, at any file size.

!!! note
    Reading a wrapped TIFF costs more than reading a wrapped file of any other format. A backward seek on a compressed stream is a decompression from offset 0, and `tifffile` seeks to the end of the file once and rewinds two or three times — so a `.tif.gz` is decompressed two to three times over. `.bz2` and `.xz` are far dearer again. An uncompressed TIFF pays none of this.

### OME-TIFF

OME-TIFF is the interchange format of light microscopy: what Bio-Formats writes, and what OMERO and the Image Data Resource serve. Its TIFF header carries an OME-XML document describing the image, and those files get a collection of their own:

| Node | Content |
|------|---------|
| `cr:FileSet` `ome-image-files` | the OME files, listed individually |
| `cr:RecordSet` `ome_images` | one row per OME **file**, with the fields below |

`ome_version`, `ome_image_count`, `size_c`, `size_z`, `size_t`, `dimension_order`, `pixel_type`, `physical_size_x`, `physical_size_y`, `physical_size_unit`, and `channel_names` as one array field. A field whose OME attribute no file in the batch declares is not emitted.

**The OME files leave the `images` collection.** Ten OME fields on the shared record set would attribute a pixel size to every PNG in the same directory, so the two collections partition the batch instead: every image is in exactly one. A consumer reading only `images` stops seeing the OME files. The `ome-image-files` FileSet lists its files rather than globbing, because a `**/*.tif` pattern would re-admit the plain TIFFs beside them; for the same reason, `images` lists any file whose extension an OME file also uses, and keeps a glob for every other extension.

**Rows are files, not images.** One OME-XML document may declare several `<Image>` elements, and one logical image may be spread over several `.ome.tif` files linked by `TiffData/UUID/@FileName`. Neither is grouped here. So the Pixels fields describe `Image[0]` of each file, `ome_image_count` says how many the file declared, and the record-set description states both. `num_bands` keeps its meaning — TIFF `SamplesPerPixel`, genuinely 1 for a three-channel OME stored as three IFDs — and `size_c` is the channel count.

**No `Field.value` is emitted.** The per-file numbers stay in the files; what each field's description carries is the range or the set the batch was observed to hold. Channel names are emitted this way, because the OME schema defines `Channel/@Name` as the label of an acquisition channel — an antibody, a probe, a fluorophore — which is what makes an imaging dataset findable. `Image/@Name` is not, and neither are `Creator` or the file `UUID`: the schema says nothing about what they contain, and in practice they hold slide labels and operator notes. Where the format guarantees the semantics the vocabulary may be described; where it guarantees nothing it may not.

Only the `image` field carries `extract: {fileProperty: content}`, because its content *is* the file's content. mlcroissant reads `image/tiff` content as the decoded pixels, so the same extract on `size_c` would ask a consumer to cast an image to an integer. Every header field is sourced `{fileSet}` and nothing more.

The OME-XML is parsed with `xml.etree.ElementTree` and no new dependency. A document that declares a DTD or an entity is not parsed, and neither is one over 8 MiB; the file is still described, as a plain TIFF, and the refusal is counted in the record-set description. A `BinaryOnly` file keeps its metadata in a companion `.companion.ome` file, which is named and not read.

Not read: `Plane`, `Objective`, `TimeIncrement`, plate and well metadata, pyramid levels, and the vendor TIFF dialects (`.svs`, `.ndpi`, `.scn`, `.qptiff`).

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
