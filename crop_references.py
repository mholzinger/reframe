"""
Reference encoder + manual-crop helper.

Tries every reference photo with progressively more aggressive face detection
(HOG, HOG+upsample, CNN at 800px). For any reference where automatic detection
fails, prints a clear instruction so the user can manually crop a tight face
shot and re-run.

Output: a pickle file containing all face encodings, ready to be loaded by
find_photos.py via the `REFERENCE_PICKLE` env var. Bypasses the need to
re-run automatic detection on every script invocation.

Council recommendation: this addresses the "3 of 12 references" recall
ceiling, which several advisors flagged as the single biggest leverage point.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import face_recognition
from PIL import Image


IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}


def encode_with_strategy(img_array, strategy: str):
    """Run face_recognition with one of three strategies. Returns list of encodings."""
    if strategy == 'hog':
        locs = face_recognition.face_locations(img_array, model='hog')
    elif strategy == 'hog_upsample2':
        locs = face_recognition.face_locations(img_array, number_of_times_to_upsample=2, model='hog')
    elif strategy == 'cnn':
        locs = face_recognition.face_locations(img_array, model='cnn')
    else:
        raise ValueError(f'unknown strategy: {strategy}')
    return face_recognition.face_encodings(img_array, known_face_locations=locs)


def load_and_downscale(path: Path, max_dim: int):
    """Return (numpy_array, (w, h)) downscaled to max_dim on the longer side."""
    pil = Image.open(path).convert('RGB')
    pil.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return np.asarray(pil), pil.size


def encode_reference(path: Path):
    """Try increasingly aggressive strategies. Returns (encodings, strategy_used)."""
    # Strategy 1: HOG at 1600px (fastest, lowest recall)
    img, _ = load_and_downscale(path, 1600)
    encs = encode_with_strategy(img, 'hog')
    if encs:
        return encs, 'hog@1600'

    # Strategy 2: HOG + upsample=2 at 1600px (catches smaller faces)
    encs = encode_with_strategy(img, 'hog_upsample2')
    if encs:
        return encs, 'hog_upsample2@1600'

    # Strategy 3: CNN at 800px (memory-safe; catches angled/partial faces)
    img_small, _ = load_and_downscale(path, 800)
    try:
        encs = encode_with_strategy(img_small, 'cnn')
        if encs:
            return encs, 'cnn@800'
    except Exception as e:
        print(f'  CNN failed on {path.name}: {e}', file=sys.stderr)

    return [], 'none'


def encode_from_bbox(path: Path, bbox: tuple):
    """Encode a manually-specified (top, right, bottom, left) bbox from the full-res image."""
    pil = Image.open(path).convert('RGB')
    img = np.asarray(pil)
    return face_recognition.face_encodings(img, known_face_locations=[bbox])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference_folder', help='Folder of reference photos')
    ap.add_argument('--output', '-o', default='reference_encodings.pkl',
                    help='Pickle output path (default: reference_encodings.pkl)')
    ap.add_argument('--manual', action='store_true',
                    help='Prompt for manual bbox coords for any photo automatic detection misses')
    args = ap.parse_args()

    ref_folder = Path(args.reference_folder)
    if not ref_folder.is_dir():
        print(f'Not a directory: {ref_folder}', file=sys.stderr)
        sys.exit(1)

    all_encodings = []
    missed = []
    for ref_path in sorted(ref_folder.iterdir()):
        if ref_path.suffix.lower() not in IMAGE_EXTS:
            continue
        encs, strategy = encode_reference(ref_path)
        if encs:
            all_encodings.extend(encs)
            print(f'  [{strategy:>22}] {ref_path.name}: {len(encs)} face(s)')
        else:
            missed.append(ref_path)
            print(f'  [{"MISSED":>22}] {ref_path.name}: no face detected')

    if missed and args.manual:
        print('\nManual mode: for each missed reference, open the file in any')
        print('image viewer, eyeball the face bounding box, and enter:')
        print('  top right bottom left  (pixel coords from the full-res image)')
        print('Type "skip" to skip a photo, "open" to print the path, or Ctrl-D to exit.\n')
        for path in missed:
            while True:
                try:
                    line = input(f'{path.name} > ').strip()
                except EOFError:
                    print()
                    break
                if not line or line == 'skip':
                    break
                if line == 'open':
                    print(f'  path: {path.absolute()}')
                    continue
                parts = line.split()
                if len(parts) != 4 or not all(p.isdigit() for p in parts):
                    print('  expected "top right bottom left" as 4 integers')
                    continue
                bbox = tuple(int(p) for p in parts)
                try:
                    encs = encode_from_bbox(path, bbox)
                    if encs:
                        all_encodings.extend(encs)
                        print(f'  encoded ok ({len(encs)} face(s))')
                        break
                    print('  bbox produced no encoding — likely outside the face')
                except Exception as e:
                    print(f'  error: {e}')

    if not all_encodings:
        print('\nNo encodings produced. Aborting.', file=sys.stderr)
        sys.exit(1)

    arrs = [np.asarray(e, dtype=np.float64) for e in all_encodings]
    with open(args.output, 'wb') as f:
        pickle.dump(arrs, f)
    print(f'\nWrote {len(arrs)} face encodings to {args.output}')


if __name__ == '__main__':
    main()
