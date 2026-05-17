
"""
Photo Similarity Finder (Face match)

1. Place reference photos of the person you want to find in REFERENCE_FOLDER.
2. Point PHOTO_FOLDER at your photo collection.
3. Optionally set DATE_RANGE_START / DATE_RANGE_END to narrow by EXIF date.
4. Run this script. Matches go to OUTPUT_FOLDER/{face_match,possible_matches}/.

All settings below can be overridden via environment variables — useful when
running on a NAS (Synology Container Manager, Portainer, etc.) where it's
easier to set env vars in the UI than edit code:

    REFERENCE_FOLDER, PHOTO_FOLDER, OUTPUT_FOLDER
    SIMILARITY_THRESHOLD          (float, default 0.5)
    DATE_RANGE_START              ('YYYY-MM-DD' or empty)
    DATE_RANGE_END                ('YYYY-MM-DD' or empty)
    NEIGHBOR_WINDOW               (int, default 20)
    NEIGHBOR_SIZE_RATIO           (float, default 0.5)
    WORKERS                       (int, default = cpu count; set 1 to disable parallelism)
    SKIP_FILENAME_PATTERNS        (comma-separated regexes; default '^FILE\\d+\\.JPG$'
                                   for 4K Stogram Instagram archives)
    SKIP_FILENAME_MAX_SIZE        (int bytes, default 500000; only skip when filename
                                   matches AND size <= this — protects EaseUS-recovered
                                   photos that share the FILE<num>.JPG naming pattern)
    REFERENCE_PICKLE              (path; if set, load encodings from this pickle instead
                                   of re-encoding REFERENCE_FOLDER each run. Generate with
                                   crop_references.py for the best recall.)
    EXIF_SERIAL_FILTER            (comma-separated camera SerialNumbers; if set, files
                                   with EXIF SerialNumber NOT in this set are skipped.
                                   Files with no EXIF serial pass through.)
    CACHE_PATH                    (sqlite db path, default $OUTPUT_FOLDER/.cache/encodings.db.
                                   Set empty to disable caching. Skip-on-cache makes
                                   re-runs after config tweaks take minutes, not hours.)
"""


import os
import re
import sys
import pickle
import face_recognition
from PIL import Image
from PIL.ExifTags import TAGS
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import numpy as np

import cache as cache_mod


# --- CONFIGURATION (env-overridable) ---
REFERENCE_FOLDER = os.environ.get('REFERENCE_FOLDER', './reference_photos')
PHOTO_FOLDER = os.environ.get('PHOTO_FOLDER', './photo_folder')
OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', './similar_photos')
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', '0.5'))

# Optional EXIF date filter (inclusive). Empty/unset disables.
DATE_RANGE_START = os.environ.get('DATE_RANGE_START') or None
DATE_RANGE_END = os.environ.get('DATE_RANGE_END') or None

NEIGHBOR_WINDOW = int(os.environ.get('NEIGHBOR_WINDOW', '20'))
NEIGHBOR_SIZE_RATIO = float(os.environ.get('NEIGHBOR_SIZE_RATIO', '0.5'))

_workers_env = os.environ.get('WORKERS', '').strip()
WORKERS = int(_workers_env) if _workers_env else (os.cpu_count() or 2)

_skip_fn_env = os.environ.get('SKIP_FILENAME_PATTERNS', r'^FILE\d+\.JPG$')
SKIP_FILENAME_PATTERNS = [re.compile(p) for p in _skip_fn_env.split(',') if p.strip()]
SKIP_FILENAME_MAX_SIZE = int(os.environ.get('SKIP_FILENAME_MAX_SIZE', '500000'))

REFERENCE_PICKLE = os.environ.get('REFERENCE_PICKLE') or None

_serial_env = (os.environ.get('EXIF_SERIAL_FILTER') or '').strip()
EXIF_SERIAL_FILTER = {s.strip() for s in _serial_env.split(',') if s.strip()} or None

_cache_default = os.path.join(OUTPUT_FOLDER, '.cache', 'encodings.db')
CACHE_PATH = os.environ.get('CACHE_PATH', _cache_default).strip()
CACHE_ENABLED = bool(CACHE_PATH)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
SKIP_DIRS = {'Spotlight-V100', 'fseventsd', '.Trashes', '.fseventsd', '.Spotlight-V100',
             '$RECYCLE.BIN', 'System Volume Information', '@eaDir'}

SEQ_PATTERN = re.compile(r'^([^\d]*?)(\d+)')


def parse_seq(stem):
    """Return (prefix, number) for filenames like 'IMG_9669' or 'DSC01128', else None."""
    m = SEQ_PATTERN.match(stem)
    return (m.group(1), int(m.group(2))) if m else None


def get_exif_rich(path):
    """Return dict with make, model, serial, software, lens, dt. Any field may be None."""
    out = {'make': None, 'model': None, 'serial': None, 'software': None, 'lens': None, 'dt': None}
    try:
        with Image.open(path) as img:
            exif = img._getexif() or {}
        for tid, v in exif.items():
            tag = TAGS.get(tid)
            if tag == 'Make': out['make'] = str(v).strip()
            elif tag == 'Model': out['model'] = str(v).strip()
            elif tag in ('BodySerialNumber', 'SerialNumber', 'CameraSerialNumber'):
                out['serial'] = str(v).strip()
            elif tag == 'Software': out['software'] = str(v).strip()
            elif tag in ('LensModel', 'Lens'):
                out['lens'] = str(v).strip()
            elif tag == 'DateTimeOriginal':
                try:
                    out['dt'] = datetime.strptime(str(v).strip(), '%Y:%m:%d %H:%M:%S')
                except Exception:
                    pass
    except Exception:
        pass
    return out


def in_date_range(dt):
    if dt is None:
        return True
    if DATE_RANGE_START and dt < datetime.strptime(DATE_RANGE_START, '%Y-%m-%d'):
        return False
    if DATE_RANGE_END and dt > datetime.strptime(DATE_RANGE_END + ' 23:59:59', '%Y-%m-%d %H:%M:%S'):
        return False
    return True


def encode_reference_aggressive(ref_file):
    """Try HOG → HOG+upsample → CNN@800. Returns (encodings, strategy_label)."""
    pil = Image.open(ref_file).convert('RGB')
    pil.thumbnail((1600, 1600), Image.LANCZOS)
    img = np.asarray(pil)
    # Strategy 1: HOG
    locs = face_recognition.face_locations(img, model='hog')
    encs = face_recognition.face_encodings(img, known_face_locations=locs)
    if encs:
        return encs, 'hog@1600'
    # Strategy 2: HOG + upsample=2
    locs = face_recognition.face_locations(img, number_of_times_to_upsample=2, model='hog')
    encs = face_recognition.face_encodings(img, known_face_locations=locs)
    if encs:
        return encs, 'hog_upsample2@1600'
    # Strategy 3: CNN at 800px (memory-safe now that CLIP isn't loaded)
    pil_small = Image.open(ref_file).convert('RGB')
    pil_small.thumbnail((800, 800), Image.LANCZOS)
    img_small = np.asarray(pil_small)
    try:
        locs = face_recognition.face_locations(img_small, model='cnn')
        encs = face_recognition.face_encodings(img_small, known_face_locations=locs)
        if encs:
            return encs, 'cnn@800'
    except Exception as e:
        print(f'  CNN failed on {ref_file.name}: {e}', flush=True)
    return [], 'none'


# --- WORKER FUNCTIONS (ProcessPoolExecutor) ---
WORKER_REFS = None
WORKER_THRESHOLD = None
WORKER_OUTPUT = None
WORKER_CACHE_PATH = None
WORKER_CONN = None


def _worker_init(refs, threshold, output_folder, cache_path):
    global WORKER_REFS, WORKER_THRESHOLD, WORKER_OUTPUT, WORKER_CACHE_PATH, WORKER_CONN
    WORKER_REFS = refs
    WORKER_THRESHOLD = threshold
    WORKER_OUTPUT = output_folder
    WORKER_CACHE_PATH = cache_path
    if cache_path:
        WORKER_CONN = cache_mod.connect(cache_path)


def _process_one_photo(photo_path_str):
    """Worker: face_recognition + copy on match. Returns matched path str or None.
    Uses the cache to skip files already encoded (across runs)."""
    photo_path = Path(photo_path_str)
    try:
        st = photo_path.stat()
        size, mtime = st.st_size, st.st_mtime

        # Cache hit: use stored encodings, skip the expensive work.
        if WORKER_CONN is not None:
            cached_encs = cache_mod.get_encodings(WORKER_CONN, photo_path_str)
            if cached_encs is not None:
                if any(any(face_recognition.compare_faces(WORKER_REFS, e, tolerance=WORKER_THRESHOLD))
                       for e in cached_encs):
                    dest = Path(WORKER_OUTPUT) / 'face_match' / photo_path.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(photo_path, dest)
                    return photo_path_str
                return None
            # If is_cached() is true but get_encodings() returned None, the file
            # was previously processed and either had no face or errored — skip.
            if cache_mod.is_cached(WORKER_CONN, photo_path_str, size, mtime):
                return None

        # Cache miss: do the work.
        img = face_recognition.load_image_file(photo_path)
        encs = face_recognition.face_encodings(img)
        exif = get_exif_rich(photo_path)
        try:
            with Image.open(photo_path) as pil:
                wh = pil.size
        except Exception:
            wh = None

        if WORKER_CONN is not None:
            cache_mod.store(WORKER_CONN, photo_path_str, size, mtime, encs, exif, image_size=wh)

        if any(any(face_recognition.compare_faces(WORKER_REFS, e, tolerance=WORKER_THRESHOLD))
               for e in encs):
            dest = Path(WORKER_OUTPUT) / 'face_match' / photo_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(photo_path, dest)
            return photo_path_str
    except Exception as e:
        if WORKER_CONN is not None:
            try:
                st = photo_path.stat()
                cache_mod.store(WORKER_CONN, photo_path_str, st.st_size, st.st_mtime,
                                [], {}, error=str(e))
            except Exception:
                pass
        print(f'Error processing {photo_path}: {e}', flush=True)
    return None


# --- LOAD REFERENCE ENCODINGS ---
reference_encodings = []
ref_cameras = set()
ref_serials = set()

if REFERENCE_PICKLE and os.path.exists(REFERENCE_PICKLE):
    print(f'Loading reference encodings from pickle: {REFERENCE_PICKLE}')
    with open(REFERENCE_PICKLE, 'rb') as f:
        loaded = pickle.load(f)
    reference_encodings = [np.asarray(e, dtype=np.float64) for e in loaded]
    print(f'Loaded {len(reference_encodings)} encodings from pickle.')
    # Still scan REFERENCE_FOLDER for EXIF to populate camera/serial allowlists
    for ref_file in Path(REFERENCE_FOLDER).glob('*'):
        if ref_file.suffix.lower() not in IMAGE_EXTS:
            continue
        exif = get_exif_rich(ref_file)
        if exif['make'] or exif['model']:
            ref_cameras.add((exif['make'], exif['model']))
        if exif['serial']:
            ref_serials.add(exif['serial'])
else:
    print('Loading reference images...')
    for ref_file in Path(REFERENCE_FOLDER).glob('*'):
        if ref_file.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            exif = get_exif_rich(ref_file)
            if exif['make'] or exif['model']:
                ref_cameras.add((exif['make'], exif['model']))
            if exif['serial']:
                ref_serials.add(exif['serial'])
            encs, strategy = encode_reference_aggressive(ref_file)
            if encs:
                reference_encodings.extend(encs)
                print(f'  [{strategy:>22}] {ref_file.name}: {len(encs)} face(s)')
            else:
                print(f'  [{"MISSED":>22}] {ref_file.name}')
        except Exception as e:
            print(f'  Error loading {ref_file.name}: {e}')

if not reference_encodings:
    print('No valid reference face encodings found. Exiting.')
    sys.exit(1)

print(f'Loaded {len(reference_encodings)} face encodings total.')
print(f'Reference cameras: {sorted(ref_cameras) if ref_cameras else "(none — EXIF camera filter disabled)"}')
if ref_serials:
    print(f'Reference serials: {sorted(ref_serials)}')
if EXIF_SERIAL_FILTER:
    print(f'EXIF serial allowlist: {sorted(EXIF_SERIAL_FILTER)}')
if DATE_RANGE_START or DATE_RANGE_END:
    print(f'Date filter: {DATE_RANGE_START or "..."} to {DATE_RANGE_END or "..."}')
if SKIP_FILENAME_PATTERNS:
    guard = f'<= {SKIP_FILENAME_MAX_SIZE} bytes' if SKIP_FILENAME_MAX_SIZE > 0 else 'no size guard (dangerous on EaseUS dumps)'
    print(f'Filename skip patterns: {[p.pattern for p in SKIP_FILENAME_PATTERNS]} (only when {guard})')
if CACHE_ENABLED:
    print(f'Cache: {CACHE_PATH}')
    main_conn = cache_mod.connect(CACHE_PATH)
    s = cache_mod.stats(main_conn)
    print(f'  existing: {s["total"]} cached ({s["with_faces"]} with faces, {s["errored"]} errored)')
    main_conn.close()


# --- SCAN PHOTOS ---
print('Scanning photos...', flush=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

skipped_by_filename = 0


def iter_photos(root):
    global skipped_by_filename
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                continue
            full_path = os.path.join(dirpath, fn)
            if SKIP_FILENAME_PATTERNS and any(p.match(fn) for p in SKIP_FILENAME_PATTERNS):
                if SKIP_FILENAME_MAX_SIZE <= 0:
                    skipped_by_filename += 1
                    continue
                try:
                    if os.path.getsize(full_path) <= SKIP_FILENAME_MAX_SIZE:
                        skipped_by_filename += 1
                        continue
                except OSError:
                    pass
            yield Path(full_path)


skipped_by_camera = 0
skipped_by_date = 0
skipped_by_serial = 0
face_matches = []


def filtered_photo_paths():
    """Yield photo paths that pass all EXIF-based pre-filters."""
    global skipped_by_camera, skipped_by_date, skipped_by_serial
    for photo_path in iter_photos(PHOTO_FOLDER):
        exif = get_exif_rich(photo_path)
        if ref_cameras and (exif['make'] or exif['model']) and (exif['make'], exif['model']) not in ref_cameras:
            skipped_by_camera += 1
            continue
        if EXIF_SERIAL_FILTER and exif['serial'] and exif['serial'] not in EXIF_SERIAL_FILTER:
            skipped_by_serial += 1
            continue
        if not in_date_range(exif['dt']):
            skipped_by_date += 1
            continue
        yield str(photo_path)


print(f'Using {WORKERS} worker process(es) for face_recognition.', flush=True)
mp_ctx = mp.get_context('fork')
QUEUE_DEPTH = WORKERS * 4
with ProcessPoolExecutor(
    max_workers=WORKERS,
    mp_context=mp_ctx,
    initializer=_worker_init,
    initargs=(reference_encodings, SIMILARITY_THRESHOLD, OUTPUT_FOLDER,
              CACHE_PATH if CACHE_ENABLED else None),
) as executor:
    paths_iter = filtered_photo_paths()
    pending = set()

    def submit_more(n):
        for _ in range(n):
            try:
                p = next(paths_iter)
            except StopIteration:
                return
            pending.add(executor.submit(_process_one_photo, p))

    submit_more(QUEUE_DEPTH)
    completed = 0
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            try:
                result = fut.result()
            except Exception as e:
                print(f'Worker exception: {e}', flush=True)
                result = None
            completed += 1
            if completed % 100 == 0:
                serial_msg = f', {skipped_by_serial} serial' if EXIF_SERIAL_FILTER else ''
                print(f'[progress] processed {completed} files (skipped: '
                      f'{skipped_by_filename} filename, {skipped_by_camera} camera{serial_msg}, '
                      f'{skipped_by_date} date, in-flight: {len(pending)})...', flush=True)
            if result is not None:
                face_matches.append(Path(result))
                print(f'MATCH (face): {result} -> {OUTPUT_FOLDER}/face_match/{Path(result).name}',
                      flush=True)
        submit_more(len(done))


# --- NEIGHBOR EXPANSION ---
print(f'Expanding {len(face_matches)} face matches to neighbors...', flush=True)
groups = defaultdict(list)
for path in face_matches:
    parsed = parse_seq(path.stem)
    if not parsed:
        continue
    prefix, num = parsed
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    groups[(path.parent, prefix)].append((num, size))

neighbor_count = 0
for (parent, prefix), nums_sizes in groups.items():
    nums = [ns[0] for ns in nums_sizes]
    sizes = [ns[1] for ns in nums_sizes if ns[1] > 0]
    lo, hi = min(nums) - NEIGHBOR_WINDOW, max(nums) + NEIGHBOR_WINDOW
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    matched_names = {p.name for p in face_matches if p.parent == parent}

    try:
        candidates = list(parent.iterdir())
    except OSError:
        continue
    for f in candidates:
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        if f.name in matched_names:
            continue
        parsed = parse_seq(f.stem)
        if not parsed or parsed[0] != prefix:
            continue
        if not (lo <= parsed[1] <= hi):
            continue
        if NEIGHBOR_SIZE_RATIO > 0 and avg_size > 0:
            try:
                fsize = f.stat().st_size
            except OSError:
                continue
            if abs(fsize - avg_size) / avg_size > NEIGHBOR_SIZE_RATIO:
                continue
        dest = Path(OUTPUT_FOLDER) / 'possible_matches' / f.name
        os.makedirs(dest.parent, exist_ok=True)
        if dest.exists():
            continue
        shutil.copy2(f, dest)
        neighbor_count += 1
        print(f'NEIGHBOR: {f} -> {dest}')

print(f'Done! {len(face_matches)} face matches, {neighbor_count} neighbors in {OUTPUT_FOLDER}')
if CACHE_ENABLED:
    final_conn = cache_mod.connect(CACHE_PATH)
    s = cache_mod.stats(final_conn)
    print(f'Cache: {s["total"]} files indexed ({s["with_faces"]} with faces, {s["errored"]} errored)')
    final_conn.close()
