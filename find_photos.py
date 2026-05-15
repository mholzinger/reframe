
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
"""


import os
import re
import sys
import face_recognition
from PIL import Image
from PIL.ExifTags import TAGS
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np


# --- CONFIGURATION (env-overridable) ---
REFERENCE_FOLDER = os.environ.get('REFERENCE_FOLDER', './reference_photos')
PHOTO_FOLDER = os.environ.get('PHOTO_FOLDER', './photo_folder')
OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', './similar_photos')
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', '0.5'))

# Optional EXIF date filter (inclusive). Empty/unset disables.
# Format: 'YYYY-MM-DD'. Photos with no EXIF date pass through.
DATE_RANGE_START = os.environ.get('DATE_RANGE_START') or None
DATE_RANGE_END = os.environ.get('DATE_RANGE_END') or None

# Neighbor expansion: after face scan, grab same-directory siblings with
# adjacent sequence numbers (DSC01244, DSC01245, ...) as possible_matches.
NEIGHBOR_WINDOW = int(os.environ.get('NEIGHBOR_WINDOW', '20'))
NEIGHBOR_SIZE_RATIO = float(os.environ.get('NEIGHBOR_SIZE_RATIO', '0.5'))

SEQ_PATTERN = re.compile(r'^([^\d]*?)(\d+)')

def parse_seq(stem):
    """Return (prefix, number) for filenames like 'IMG_9669' or 'DSC01128', else None."""
    m = SEQ_PATTERN.match(stem)
    return (m.group(1), int(m.group(2))) if m else None


def get_exif(path):
    """Return (make, model, datetime_taken) from EXIF. Any field may be None."""
    try:
        with Image.open(path) as img:
            exif = img._getexif() or {}
        make = model = dt = None
        for tid, v in exif.items():
            tag = TAGS.get(tid)
            if tag == 'Make': make = str(v).strip()
            elif tag == 'Model': model = str(v).strip()
            elif tag == 'DateTimeOriginal':
                try:
                    dt = datetime.strptime(str(v).strip(), '%Y:%m:%d %H:%M:%S')
                except Exception:
                    pass
        return (make, model, dt)
    except Exception:
        return (None, None, None)


def in_date_range(dt):
    """Return True if dt is within configured range (or no date at all)."""
    if dt is None:
        return True  # photos with no EXIF date pass through
    if DATE_RANGE_START and dt < datetime.strptime(DATE_RANGE_START, '%Y-%m-%d'):
        return False
    if DATE_RANGE_END and dt > datetime.strptime(DATE_RANGE_END + ' 23:59:59', '%Y-%m-%d %H:%M:%S'):
        return False
    return True


# --- LOAD REFERENCE ENCODINGS ---
print('Loading reference images...')
reference_encodings = []
ref_cameras = set()
for ref_file in Path(REFERENCE_FOLDER).glob('*'):
    try:
        make, model, _ = get_exif(ref_file)
        if make or model:
            ref_cameras.add((make, model))
        # Downscale large refs before HOG; faces stay detectable and detection runs ~15x faster
        pil_ref = Image.open(ref_file).convert('RGB')
        pil_ref.thumbnail((1600, 1600), Image.LANCZOS)
        img = np.asarray(pil_ref)
        face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=2)
        encs = face_recognition.face_encodings(img, known_face_locations=face_locations)
        if encs:
            reference_encodings.extend(encs)
            print(f'Loaded reference (face x{len(encs)}): {ref_file}')
        else:
            print(f'No face found in reference: {ref_file}')
    except Exception as e:
        print(f'Error loading {ref_file}: {e}')

if not reference_encodings:
    print('No valid reference face encodings found. Exiting.')
    sys.exit(1)

print(f'Loaded {len(reference_encodings)} face encodings from references.')
print(f'Reference cameras: {sorted(ref_cameras) if ref_cameras else "(none — EXIF camera filter disabled)"}')
if DATE_RANGE_START or DATE_RANGE_END:
    print(f'Date filter: {DATE_RANGE_START or "..."} to {DATE_RANGE_END or "..."}')


# --- SCAN PHOTOS ---
print('Scanning photos...', flush=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
SKIP_DIRS = {'Spotlight-V100', 'fseventsd', '.Trashes', '.fseventsd', '.Spotlight-V100', '$RECYCLE.BIN', 'System Volume Information'}

def iter_photos(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                yield Path(dirpath) / fn

skipped_by_camera = 0
skipped_by_date = 0
face_matches = []  # list of Path; used for neighbor expansion below
for idx, photo_path in enumerate(iter_photos(PHOTO_FOLDER)):
    if idx % 100 == 0:
        print(f'[progress] scanned {idx} files (skipped: {skipped_by_camera} camera, {skipped_by_date} date)...', flush=True)
    # EXIF pre-filters: photos with no/corrupted EXIF pass through both filters.
    make, model, dt = get_exif(photo_path)
    if ref_cameras and (make or model) and (make, model) not in ref_cameras:
        skipped_by_camera += 1
        continue
    if not in_date_range(dt):
        skipped_by_date += 1
        continue
    try:
        img = face_recognition.load_image_file(photo_path)
        encs = face_recognition.face_encodings(img)
        found = any(any(face_recognition.compare_faces(reference_encodings, enc, tolerance=SIMILARITY_THRESHOLD)) for enc in encs)
        if found:
            dest = Path(OUTPUT_FOLDER) / 'face_match' / photo_path.name
            os.makedirs(dest.parent, exist_ok=True)
            shutil.copy2(photo_path, dest)
            face_matches.append(photo_path)
            print(f'MATCH (face): {photo_path} -> {dest}')
    except Exception as e:
        print(f'Error processing {photo_path}: {e}')


# --- NEIGHBOR EXPANSION ---
# For each (directory, filename-prefix) group of face matches, find sibling
# files in the same directory whose sequence number falls within the matched
# range expanded by NEIGHBOR_WINDOW. These are likely from the same shoot
# session (e.g., wife turned away in some frames so face_recognition missed them).
print(f'Expanding {len(face_matches)} face matches to neighbors...', flush=True)
groups = defaultdict(list)  # (parent_dir, prefix) -> list of (number, size)
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