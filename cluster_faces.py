"""
Cluster photos by detected face identity.

Walks one or more input directories, extracts face encodings (reusing the
SQLite cache where possible), and groups faces by identity using a union-find
clustering at a configurable distance threshold. Photos with multiple faces
appear in every matching cluster; photos with no detected face go to no_face/.

Designed to run inside the reframe Docker image — no new dependencies, just
face_recognition + numpy that are already present.

Council Tier 1-ish: this gets you "who is everyone in my recovered archive"
without any new model. Won't help on body-only/face-occluded photos
(face_recognition limitation), but for the face-visible subset it's exactly
what unsupervised face clustering is supposed to do.

Usage:
    python cluster_faces.py \\
        --input /app/photo_folder \\
        --input /app/similar_photos/face_match \\
        --cache /app/similar_photos/.cache/encodings.db \\
        --output /app/similar_photos/clusters \\
        --eps 0.45 \\
        --min-samples 3 \\
        --workers 3
"""

import argparse
import csv
import os
import sys
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
from pathlib import Path

import numpy as np
import face_recognition

import cache as cache_mod


IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
SKIP_DIRS = {'Spotlight-V100', 'fseventsd', '.Trashes', '.fseventsd', '.Spotlight-V100',
             '$RECYCLE.BIN', 'System Volume Information', '@eaDir', '.cache'}


# ---- Worker for parallel encoding of cache-miss files ----

WORKER_CACHE_PATH = None
WORKER_CONN = None


def _worker_init(cache_path):
    global WORKER_CONN
    if cache_path:
        WORKER_CONN = cache_mod.connect(cache_path)


def _encode_one(path_str):
    """Worker: encode faces in one file, store in cache, return (path, encodings_list)."""
    try:
        img = face_recognition.load_image_file(path_str)
        encs = face_recognition.face_encodings(img)
        if WORKER_CONN is not None:
            try:
                st = os.stat(path_str)
                # Empty exif since we don't need it for clustering — cache
                # is keyed on path and we only need face encodings here.
                cache_mod.store(WORKER_CONN, path_str, st.st_size, st.st_mtime,
                                encs, {})
            except Exception:
                pass
        return path_str, [np.asarray(e, dtype=np.float64) for e in encs]
    except Exception as e:
        return path_str, []  # treat as no faces


# ---- Clustering: union-find on faces within eps distance ----


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def cluster_by_distance(encodings, eps):
    """Cluster encodings by union-find: any two faces within eps distance join.
    Returns: cluster_ids list (one per encoding), cluster_size dict.
    """
    n = len(encodings)
    if n == 0:
        return [], {}
    uf = UnionFind(n)
    arr = np.asarray(encodings)
    for i in range(n):
        # Batched distance from face i to all later faces.
        if i + 1 >= n:
            break
        diffs = arr[i+1:] - arr[i]
        dists = np.linalg.norm(diffs, axis=1)
        close = np.where(dists < eps)[0]
        for j in close:
            uf.union(i, int(i + 1 + j))

    # Map roots to compact cluster IDs.
    roots = {}
    cluster_ids = []
    for i in range(n):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        cluster_ids.append(roots[r])

    sizes = defaultdict(int)
    for cid in cluster_ids:
        sizes[cid] += 1
    return cluster_ids, dict(sizes)


# ---- Walk + collect ----


def walk_images(roots, skip_filename_patterns=None):
    """Yield image paths from one or more input roots."""
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            print(f'Skipping non-directory input: {root}', file=sys.stderr)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                    continue
                if fn.startswith('.'):
                    continue
                yield os.path.join(dirpath, fn)


def gather_encodings(image_paths, cache_path, workers):
    """For each path, return its face encodings. Use cache; encode misses in parallel.
    Returns: list of (path, [encoding, ...])
    """
    results = {}
    missing = []

    if cache_path:
        conn = cache_mod.connect(cache_path)
        for p in image_paths:
            cached = cache_mod.get_encodings(conn, p)
            if cached is not None:
                results[p] = cached
            elif cache_mod.is_cached(conn, p, *_safe_stat(p)):
                results[p] = []  # previously encoded, no faces
            else:
                missing.append(p)
        conn.close()
    else:
        missing = list(image_paths)

    print(f'  cache hits: {len(image_paths) - len(missing)}/{len(image_paths)}, '
          f'misses to encode: {len(missing)}', flush=True)

    if missing:
        mp_ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=workers,
                                  mp_context=mp_ctx,
                                  initializer=_worker_init,
                                  initargs=(cache_path,)) as ex:
            pending = set()
            paths_iter = iter(missing)
            QUEUE_DEPTH = workers * 4

            def submit_more(n):
                for _ in range(n):
                    try:
                        p = next(paths_iter)
                    except StopIteration:
                        return
                    pending.add(ex.submit(_encode_one, p))

            submit_more(QUEUE_DEPTH)
            done_count = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    p, encs = fut.result()
                    results[p] = encs
                    done_count += 1
                    if done_count % 100 == 0:
                        print(f'  encoded {done_count}/{len(missing)}...', flush=True)
                submit_more(len(done))

    return [(p, results.get(p, [])) for p in image_paths]


def _safe_stat(path):
    try:
        st = os.stat(path)
        return st.st_size, st.st_mtime
    except OSError:
        return -1, -1.0


# ---- Output ----


def output_clusters(per_photo, cluster_ids, sizes, output_dir, min_samples, mode='copy'):
    """Materialize clusters as folders of files. per_photo is list of
    (path, faces) where faces is a list of (encoding, cluster_id) tuples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'no_face').mkdir(exist_ok=True)
    (output_dir / 'noise').mkdir(exist_ok=True)

    def place(src, subdir, name):
        dest_dir = output_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        # Avoid clobbering: if a name collision happens, append numeric suffix.
        if dest.exists() and os.path.realpath(dest) != os.path.realpath(src):
            stem = dest.stem
            suffix = dest.suffix
            i = 1
            while (dest_dir / f'{stem}__{i}{suffix}').exists():
                i += 1
            dest = dest_dir / f'{stem}__{i}{suffix}'
        try:
            if mode == 'link':
                os.link(src, dest)
            else:
                shutil.copy2(src, dest)
        except OSError as e:
            print(f'  failed to place {src} → {dest}: {e}', file=sys.stderr)

    # Renumber clusters by size (largest = cluster_001) for readability.
    valid_clusters = sorted(
        [cid for cid, sz in sizes.items() if sz >= min_samples],
        key=lambda c: -sizes[c]
    )
    cluster_name = {cid: f'cluster_{i+1:03d}' for i, cid in enumerate(valid_clusters)}

    manifest_rows = []
    placed_count = defaultdict(int)
    for path, faces in per_photo:
        basename = os.path.basename(path)
        if not faces:
            place(path, 'no_face', basename)
            manifest_rows.append((path, 0, ''))
            placed_count['no_face'] += 1
            continue
        photo_clusters = set()
        for _enc, cid in faces:
            if cid in cluster_name:
                photo_clusters.add(cluster_name[cid])
        if photo_clusters:
            for name in photo_clusters:
                place(path, name, basename)
                placed_count[name] += 1
            manifest_rows.append((path, len(faces), ','.join(sorted(photo_clusters))))
        else:
            place(path, 'noise', basename)
            manifest_rows.append((path, len(faces), 'noise'))
            placed_count['noise'] += 1

    # CSV manifest for review.
    manifest_path = output_dir / 'manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source_path', 'face_count', 'clusters'])
        for row in manifest_rows:
            w.writerow(row)

    print(f'\nOutput summary ({mode} mode):')
    for name in sorted(placed_count.keys()):
        print(f'  {name}: {placed_count[name]}')
    print(f'  manifest: {manifest_path}')


# ---- Main ----


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', action='append', required=True,
                    help='Input directory to walk (repeatable). One or more.')
    ap.add_argument('--cache', default=None,
                    help='Optional SQLite cache from find_photos.py for re-using encodings.')
    ap.add_argument('--output', required=True,
                    help='Output directory for clusters/, noise/, no_face/, manifest.csv.')
    ap.add_argument('--eps', type=float, default=0.45,
                    help='Distance threshold for joining faces into a cluster. '
                         'Default 0.45 (tighter than face match tolerance 0.5).')
    ap.add_argument('--min-samples', type=int, default=3,
                    help='Minimum faces per cluster; smaller groups go to noise/. Default 3.')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 2,
                    help='Parallel workers for encoding cache misses.')
    ap.add_argument('--link', action='store_true',
                    help='Use hardlinks instead of copies (must be same filesystem).')
    ap.add_argument('--limit', type=int, default=0,
                    help='Stop after walking N input files (0 = no limit). Useful for testing.')
    args = ap.parse_args()

    print(f'Inputs:  {args.input}')
    print(f'Output:  {args.output}')
    print(f'Cache:   {args.cache or "(none)"}')
    print(f'eps:     {args.eps}')
    print(f'min:     {args.min_samples}')
    print(f'workers: {args.workers}')
    print(f'mode:    {"link" if args.link else "copy"}')
    print()

    print('Walking input directories...')
    paths = []
    for p in walk_images(args.input):
        paths.append(p)
        if args.limit and len(paths) >= args.limit:
            print(f'  reached --limit {args.limit}, stopping walk')
            break
    print(f'  found {len(paths)} images')

    print('\nGathering encodings (cache + parallel encode for misses)...')
    per_path = gather_encodings(paths, args.cache, args.workers)

    # Flatten to (encoding_idx → (path_idx, face_idx_in_photo)) for clustering.
    flat_encodings = []
    photo_face_index = []  # (path_idx, face_idx_in_photo)
    for pi, (path, encs) in enumerate(per_path):
        for fi, enc in enumerate(encs):
            flat_encodings.append(enc)
            photo_face_index.append((pi, fi))

    print(f'\nClustering {len(flat_encodings)} face encodings...')
    cluster_ids, sizes = cluster_by_distance(flat_encodings, args.eps)
    valid = sum(1 for s in sizes.values() if s >= args.min_samples)
    print(f'  found {len(sizes)} raw clusters, {valid} with >= {args.min_samples} faces')

    # Rebuild per-photo view: each photo gets list of (encoding, cluster_id).
    per_photo = [(p, []) for p, _ in per_path]
    for global_idx, (pi, _fi) in enumerate(photo_face_index):
        per_photo[pi][1].append((flat_encodings[global_idx], cluster_ids[global_idx]))

    print('\nWriting cluster folders...')
    output_clusters(per_photo, cluster_ids, sizes, args.output, args.min_samples,
                    mode='link' if args.link else 'copy')

    print('\nDone.')


if __name__ == '__main__':
    main()
