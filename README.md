# reframe

Face-recognition photo finder for recovering photos of a specific person from large, messy, or recovered photo collections. Built for the data-recovery scenario: a drive gets accidentally formatted, you run a recovery tool, and you end up with 100k+ files with stripped EXIF, broken filenames, and no folder structure — but you specifically want to find every photo of one person.

## What it does

Given a folder of reference photos showing the target person, scans an arbitrary photo collection and produces two output buckets:

- **`face_match/`** — high-confidence matches where `face_recognition` detected the target person directly.
- **`possible_matches/`** — neighbor expansion pile. For each face match, grabs same-directory photos with adjacent sequence numbers (e.g., a hit on `IMG_9669.jpg` pulls in `IMG_9670.jpg`, `IMG_9680.jpg`, etc.). Useful for catching photos from the same shoot where the subject's face wasn't detectable — turned away, blurry, body-only — under the assumption that sequential cameras shots tend to share a subject.

## How it works

1. **Reference loading** — reads reference photos, downscales, and tries face detection with progressively more aggressive strategies (HOG → HOG+upsample → CNN at 800px). For references where automatic detection fails, use `crop_references.py` to manually crop face bounding boxes and save a pickle of encodings (see [Generating better references](#generating-better-references)).
2. **Filename + size pre-filter** (optional) — drops files whose name matches a configurable regex *and* whose size is below a configurable threshold (default 500 KB). Defaults catch 4K Stogram / Instagram-archive captures (`FILE12345.JPG`, always small) without touching EaseUS-recovered camera photos that share the same naming pattern but are megabytes in size. Cost is one `os.stat` per filename match — still essentially free.
3. **EXIF pre-filter** (optional) — if reference photos have EXIF, the scanner skips collection photos whose EXIF says a non-matching camera make/model. Photos with no/corrupted EXIF pass through (so recovered files with stripped metadata aren't lost).
4. **Date range pre-filter** (optional) — skips photos outside a configured `DateTimeOriginal` window. Same pass-through behavior for missing EXIF.
5. **Parallel face scan** — surviving photos are dispatched to a pool of worker processes (defaults to `cpu_count`). Each worker runs `face_recognition.face_encodings`, compares against reference encodings, and copies hits to `face_match/`. Uses a `wait(FIRST_COMPLETED)` pattern with a bounded in-flight queue so a slow file never blocks reporting from the others. **All face encodings + EXIF are cached in a SQLite DB** keyed on `(path, size, mtime)`, so re-runs after threshold or reference changes skip the expensive encoding step and finish in minutes instead of hours.
6. **Neighbor expansion** — after the scan, groups face matches by `(directory, filename_prefix)`, computes numeric range, and copies same-prefix siblings in the expanded window (with an optional file-size sanity check) to `possible_matches/`.

## Requirements

Docker. Everything runs in the container — no Python install on the host needed.

## Quick start

```bash
# Build
docker build -t reframe .

# Put a few clear photos of the target person in ./reference_photos/
# Then run:
docker run --rm \
  -v "$(pwd)/reference_photos:/app/reference_photos" \
  -v "/path/to/your/photo/collection:/app/photo_folder" \
  -v "$(pwd)/similar_photos:/app/similar_photos" \
  reframe
```

When it finishes, check `./similar_photos/face_match/` and `./similar_photos/possible_matches/`.

## Synology NAS (Container Manager)

This is built to work as a one-shot container you launch from the Synology UI.

1. Copy the project to your NAS (or build the image locally and `docker save` / `docker load` it on the NAS).
2. In **Container Manager → Image**, build the image from this directory.
3. **Create a container** from the image with these bind mounts:

   | Container path | NAS path | Notes |
   |---|---|---|
   | `/app/reference_photos` | `/volume1/photo/references` | Reference photos of target person |
   | `/app/photo_folder` | `/volume1/photo/source` | Collection to scan (read-only OK) |
   | `/app/similar_photos` | `/volume1/photo/output` | Where matches get copied |

4. Optionally fill in the **Environment** tab to tune behavior — every setting is env-overridable (see [Configuration](#configuration)).
5. Launch. The container runs once, writes its output, and exits.

## Configuration

Every config knob is set via environment variable, so the same image works for any combination of paths/thresholds without rebuilding.

| Variable | Default | Purpose |
|---|---|---|
| `REFERENCE_FOLDER` | `/app/reference_photos` | Folder with reference images |
| `PHOTO_FOLDER` | `/app/photo_folder` | Photo collection to scan (recursive) |
| `OUTPUT_FOLDER` | `/app/similar_photos` | Where matches and neighbors get copied |
| `SIMILARITY_THRESHOLD` | `0.5` | Face match strictness — lower is stricter (dlib default is `0.6`) |
| `DATE_RANGE_START` | (unset) | EXIF date filter start, `YYYY-MM-DD` |
| `DATE_RANGE_END` | (unset) | EXIF date filter end, `YYYY-MM-DD` |
| `NEIGHBOR_WINDOW` | `20` | Numeric range extends ± this many positions past the matched min/max |
| `NEIGHBOR_SIZE_RATIO` | `0.5` | Neighbor file size must be within this fraction of avg match size (`0` disables) |
| `WORKERS` | `cpu_count()` | Number of parallel face-recognition worker processes. Set `1` to disable parallelism, or tune down if dlib's internal threading oversubscribes your CPU |
| `SKIP_FILENAME_PATTERNS` | `^FILE\d+\.JPG$` | Comma-separated regexes. Filenames matching any pattern are candidates for the skip filter. Default catches 4K Stogram captures; set empty to disable |
| `SKIP_FILENAME_MAX_SIZE` | `500000` | Max file size (bytes) for the skip filter to fire. A file must match the regex *and* be no larger than this to be skipped. Set `0` to skip purely by name (dangerous if your dump contains EaseUS-recovered files with `FILE<num>.JPG` names) |
| `REFERENCE_PICKLE` | (unset) | Path to a pickle of pre-computed face encodings (produced by `crop_references.py`). If set and the file exists, used instead of re-encoding `REFERENCE_FOLDER` each run. Higher recall when used with manual cropping |
| `EXIF_SERIAL_FILTER` | (unset) | Comma-separated camera EXIF `SerialNumber`s to allow. Files with EXIF serial *not* in this list are skipped; files with no EXIF serial pass through. Strong "shot by this camera" fingerprint when EXIF survives |
| `CACHE_PATH` | `$OUTPUT_FOLDER/.cache/encodings.db` | SQLite cache of face encodings + EXIF. Set empty to disable. Survives across runs; re-runs after config tweaks are much faster |

### Tuning

- **Too many false positives in `face_match/`** — lower `SIMILARITY_THRESHOLD` (try `0.45`).
- **Missing obvious face matches** — raise `SIMILARITY_THRESHOLD` toward `0.6`, or add more/better reference photos.
- **Too few neighbors picked up** — raise `NEIGHBOR_WINDOW` (try `50`) or disable the size filter (`NEIGHBOR_SIZE_RATIO=0`).
- **Too much noise in `possible_matches/`** — tighten `NEIGHBOR_WINDOW` (try `5`–`10`) and lower `NEIGHBOR_SIZE_RATIO` to `0.3`.
- **Scan throughput feels low** — check the `[progress]` line. If `in-flight` is consistently at `WORKERS * 4`, workers are saturated and you're at hardware ceiling. If `in-flight` is small and CPU isn't pinned, EXIF reads in the main process may be the bottleneck on slow disks. On a 4-core NAS, `WORKERS=3` often beats `WORKERS=4` because dlib uses some internal threading and 4 workers can oversubscribe.
- **Dump dominated by Instagram-archive junk** — confirm `SKIP_FILENAME_PATTERNS` is biting (the progress line shows the `skipped: N filename` count). Add more patterns to skip other recognized-junk filenames, e.g. `SKIP_FILENAME_PATTERNS='^FILE\d+\.JPG$,^thumb_,^icon_'`.
- **Iterating on threshold / references** — after the first full scan, the SQLite cache (`CACHE_PATH`) has every file's face encodings. A re-run with a tighter `SIMILARITY_THRESHOLD` or improved `REFERENCE_PICKLE` skips the encoding step entirely and finishes in minutes.
- **Very few reference encodings (e.g. 3 of 12)** — the original HOG detector misses angled faces. The pipeline now tries HOG → HOG+upsample → CNN at 800px automatically, but for the hardest references run `crop_references.py --manual` to add manually-cropped encodings.

## Helper utilities

### Generating better references

`face_recognition`'s default HOG detector misses references with angled or partial faces. The bundled `crop_references.py` script tries multiple detection strategies and falls back to manual bounding-box entry for the ones that fail:

```bash
# Inside the container (or with face_recognition installed locally):
python crop_references.py ./reference_photos --manual --output ./reference_encodings.pkl
```

For each reference photo that automatic detection misses, you'll get a prompt. Open the file in any image viewer, eyeball the face bounding box, and type `top right bottom left` (pixel coordinates from the full-resolution image). Type `skip` to skip, `open` to print the absolute path, or Ctrl-D to exit.

Then set `REFERENCE_PICKLE=/app/reference_encodings.pkl` in your `docker run` command (or in Synology Container Manager's Environment tab) and the much-richer encoding set will be used for every subsequent scan.

### Clustering recovered photos by face identity

After `find_photos.py` has scanned a collection, the SQLite cache holds face encodings for every photo it touched. `cluster_faces.py` reuses those encodings to group photos by detected face identity — no labels needed, no model retraining. Useful when you've recovered hundreds of photos containing many different people (other models, friends, photographers) and want them automatically sorted.

Inside the container (or via `docker run reframe python cluster_faces.py ...`):

```bash
python cluster_faces.py \
  --input /app/photo_folder \
  --input /app/similar_photos/face_match \
  --cache /app/similar_photos/.cache/encodings.db \
  --output /app/similar_photos/clusters \
  --eps 0.45 \
  --min-samples 3 \
  --workers 3
```

You can pass `--input` multiple times to combine sources. The script reuses cached encodings (essentially free) and only encodes uncached files in parallel.

Output structure:

```
similar_photos/clusters/
├── cluster_001/    # largest group of one identity (copied/linked photos)
├── cluster_002/
├── ...
├── noise/          # singletons or groups below --min-samples
├── no_face/        # photos where no face was detected
└── manifest.csv    # per-photo: path, face_count, cluster names
```

Photos containing multiple faces (e.g., two people in one shot) appear in **every** matching cluster.

Tuning:
- **Too many clusters / same person split** — raise `--eps` (e.g., `0.50`). Easier to merge identities.
- **People merged together** — lower `--eps` (e.g., `0.40`). Stricter identity match.
- **Cluster folders cluttered with one-offs** — raise `--min-samples` to a higher number; rare-faces go to `noise/`.
- **Save disk space** — add `--link` to use hardlinks instead of copies (requires same filesystem; matters on a NAS).

### Verifying the filename skip filter

The default filter (`SKIP_FILENAME_PATTERNS=^FILE\d+\.JPG$` AND size ≤ 500 KB) is designed to nuke 4K Stogram Instagram archives without touching real camera photos. But if your recovery dump has a different naming convention, the filter could be silently throwing away files you care about. Run the bundled verifier before trusting the filter on a new dump:

```bash
# On the NAS (requires exiftool: opkg install exiftool):
./scripts/verify_skipped.sh "/volume1/_Backups/4TDrive/your-dump" 50
```

The script samples 50 files the filter would skip, dumps their EXIF, and prints a verdict: **SAFE**, **CAUTION**, or **STOP**. If any sampled file has a Make/Model/SerialNumber EXIF tag, the filter is wrong for your dump and needs to be tightened before committing to a multi-hour scan.

## Limitations

- **No visible face = no direct match.** Subjects turned away, occluded, or out-of-frame won't be caught by face matching. Neighbor expansion is the partial workaround, but it only helps when sequential filenames are preserved.
- **Recovery dumps lose metadata.** Most file-recovery tools strip EXIF and assign random filenames. The EXIF filters here pass unmatched-EXIF files through by design (so recovered files aren't lost), but that also means the filters do less work the messier the source.
- **Identity, not visual similarity.** This is a face-recognition tool, not a person-re-identification tool. It can't reliably tell person A's body from person B's body in a photo with no visible face.
- **CPU-bound.** No GPU support. Expect roughly 1–2 images/second per CPU core; a 100k-image collection on a typical multi-core CPU is a multi-hour to multi-day job. Parallelism scales near-linearly with `WORKERS` until you saturate cores.

## Tips

- **Reference photos matter.** A few clear, front-facing photos beat many awkward-angle ones. If face detection fails on a reference, the script reports it — drop or replace those references.
- **Pre-filter the collection** if you can. Even narrowing to "photos from this 2-month window" cuts the work and the noise dramatically.
- **Mount the source folder read-only.** This tool only reads from `PHOTO_FOLDER` and only writes to `OUTPUT_FOLDER`. Use `:ro` on the source mount for peace of mind:
  `-v "/source:/app/photo_folder:ro"`

## Acknowledgements

Stands on `face_recognition` (built on dlib's HOG detector and ResNet face encoder), `Pillow`, and `numpy`.
