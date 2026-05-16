# reframe

Face-recognition photo finder for recovering photos of a specific person from large, messy, or recovered photo collections. Built for the data-recovery scenario: a drive gets accidentally formatted, you run a recovery tool, and you end up with 100k+ files with stripped EXIF, broken filenames, and no folder structure — but you specifically want to find every photo of one person.

## What it does

Given a folder of reference photos showing the target person, scans an arbitrary photo collection and produces two output buckets:

- **`face_match/`** — high-confidence matches where `face_recognition` detected the target person directly.
- **`possible_matches/`** — neighbor expansion pile. For each face match, grabs same-directory photos with adjacent sequence numbers (e.g., a hit on `IMG_9669.jpg` pulls in `IMG_9670.jpg`, `IMG_9680.jpg`, etc.). Useful for catching photos from the same shoot where the subject's face wasn't detectable — turned away, blurry, body-only — under the assumption that sequential cameras shots tend to share a subject.

## How it works

1. **Reference loading** — reads reference photos, downscales for performance, runs HOG face detection with upsampling to catch smaller/angled faces.
2. **Filename pre-filter** (optional) — drops files whose name matches a configurable regex before they ever get opened. Defaults to skipping 4K Stogram / Instagram-archive captures (`FILE12345.JPG`-style) which dominate some recovery dumps and never contain personal photos. Free, runs at directory-walk time.
3. **EXIF pre-filter** (optional) — if reference photos have EXIF, the scanner skips collection photos whose EXIF says a non-matching camera make/model. Photos with no/corrupted EXIF pass through (so recovered files with stripped metadata aren't lost).
4. **Date range pre-filter** (optional) — skips photos outside a configured `DateTimeOriginal` window. Same pass-through behavior for missing EXIF.
5. **Parallel face scan** — surviving photos are dispatched to a pool of worker processes (defaults to `cpu_count`). Each worker runs `face_recognition.face_encodings`, compares against reference encodings, and copies hits to `face_match/`. Uses a `wait(FIRST_COMPLETED)` pattern with a bounded in-flight queue so a slow file never blocks reporting from the others.
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
| `SKIP_FILENAME_PATTERNS` | `^FILE\d+\.JPG$` | Comma-separated regexes. Filenames matching any pattern are skipped at directory-walk time. Default catches 4K Stogram captures; set empty to disable |

### Tuning

- **Too many false positives in `face_match/`** — lower `SIMILARITY_THRESHOLD` (try `0.45`).
- **Missing obvious face matches** — raise `SIMILARITY_THRESHOLD` toward `0.6`, or add more/better reference photos.
- **Too few neighbors picked up** — raise `NEIGHBOR_WINDOW` (try `50`) or disable the size filter (`NEIGHBOR_SIZE_RATIO=0`).
- **Too much noise in `possible_matches/`** — tighten `NEIGHBOR_WINDOW` (try `5`–`10`) and lower `NEIGHBOR_SIZE_RATIO` to `0.3`.
- **Scan throughput feels low** — check the `[progress]` line. If `in-flight` is consistently at `WORKERS * 4`, workers are saturated and you're at hardware ceiling. If `in-flight` is small and CPU isn't pinned, EXIF reads in the main process may be the bottleneck on slow disks. On a 4-core NAS, `WORKERS=3` often beats `WORKERS=4` because dlib uses some internal threading and 4 workers can oversubscribe.
- **Dump dominated by Instagram-archive junk** — confirm `SKIP_FILENAME_PATTERNS` is biting (the progress line shows the `skipped: N filename` count). Add more patterns to skip other recognized-junk filenames, e.g. `SKIP_FILENAME_PATTERNS='^FILE\d+\.JPG$,^thumb_,^icon_'`.

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
