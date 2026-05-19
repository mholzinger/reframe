#!/usr/bin/env python3
"""
Delete source files in the EaseUS recovery dump that have already been
successfully recovered into face_match/ (and its subfolders).

Safety design:
- DRY-RUN by default. Nothing is deleted without --commit or --trash.
- Verifies face_match copy exists AND matches source size before flagging.
- Skips files where verification fails (logged as warnings).
- --trash uses macOS Finder trash (recoverable from Trash app) instead of rm.
- Writes a per-run transcript so you have a permanent record.
- Refuses to act on more than --limit files per invocation (default 5000).

Typical workflow:
  # 1. Preview what would happen
  python delete_recovered.py \\
      --log /Volumes/_Backups/4TDrive/restore/reframe.log \\
      --face-match /Volumes/_Backups/4TDrive/restore/similar_photos/face_match \\
      --source "/Volumes/_Backups/4TDrive/EaseUS 03-24 0934" \\
      --exclude not_matched

  # 2. After verifying the plan looks right, run it for real (sent to Trash):
  python delete_recovered.py ... --trash --exclude not_matched
"""

import argparse
import csv
import os
import re
import sys
import subprocess
from datetime import datetime
from pathlib import Path


MATCH_LINE = re.compile(r'^MATCH \(face\): (.+?) -> (.+?)$')


def parse_log_for_sources(log_path, container_source_prefix, host_source_prefix):
    """Yield (basename, host_source_path) tuples from MATCH lines in the log."""
    seen = {}
    with open(log_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = MATCH_LINE.match(line.strip())
            if not m:
                continue
            container_path, _ = m.groups()
            if not container_path.startswith(container_source_prefix):
                continue
            # Translate container path back to host path.
            relative = container_path[len(container_source_prefix):].lstrip('/')
            host_path = os.path.join(host_source_prefix, relative)
            basename = os.path.basename(container_path)
            # Keep the most recent log entry for any basename (later runs win).
            seen[basename] = host_path
    return seen


def parse_manifest_for_sources(manifest_path, container_source_prefix, host_source_prefix,
                                exclude_categories=None):
    """Parse a cluster_faces.py manifest.csv. Returns dict basename → host_path
    for source paths starting with container_source_prefix, after filtering out
    rows in any excluded category.

    A row's category is the set of cluster names in column 3 (e.g. 'cluster_001',
    or 'noise', or '' which is treated as 'no_face').
    """
    exclude = set(exclude_categories or [])
    seen = {}
    category_counts = {}
    with open(manifest_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 3:
                continue
            source_path, _face_count, clusters_col = row[0], row[1], row[2]
            if not source_path.startswith(container_source_prefix):
                continue
            row_cats = set(clusters_col.split(',')) if clusters_col else {'no_face'}
            for cat in row_cats:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            # Skip only if EVERY category the row belongs to is excluded.
            if exclude and not (row_cats - exclude):
                continue
            relative = source_path[len(container_source_prefix):].lstrip('/')
            host_path = os.path.join(host_source_prefix, relative)
            basename = os.path.basename(source_path)
            seen[basename] = host_path
    return seen, category_counts


def walk_face_match(face_match_root, exclude_subdirs):
    """Return dict of basename → relative path under face_match."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(face_match_root):
        # In-place filter so os.walk doesn't descend into excluded subdirs.
        dirnames[:] = [d for d in dirnames if d not in exclude_subdirs]
        for fn in filenames:
            if fn.startswith('.'):
                continue
            full = os.path.join(dirpath, fn)
            out[fn] = full
    return out


def send_to_trash(path):
    """macOS: move to ~/.Trash via Finder (recoverable). Returns True on success."""
    # Using osascript is the only reliable way to put files in Trash (not just ~/.Trash)
    # so they show up in Finder's Trash UI and can be restored with Cmd-Z.
    escaped = path.replace('\\', '\\\\').replace('"', '\\"')
    script = f'tell application "Finder" to delete (POSIX file "{escaped}" as alias)'
    try:
        subprocess.run(['osascript', '-e', script], check=True, capture_output=True, text=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', help='Path to reframe.log with MATCH lines (mutually exclusive with --manifest)')
    ap.add_argument('--manifest', help='Path to cluster_faces.py manifest.csv (mutually exclusive with --log)')
    ap.add_argument('--face-match', help='face_match/ directory root (kept files). Required for --log unless --log-only.')
    ap.add_argument('--source', required=True, help='Host path to source dump root')
    ap.add_argument('--container-source', default='/app/photo_folder',
                    help='Container-side prefix in log/manifest paths (default: /app/photo_folder)')
    ap.add_argument('--exclude', action='append', default=[],
                    help='Subfolder name(s) of face_match to skip (e.g. not_matched). Log mode only. Repeatable.')
    ap.add_argument('--log-only', action='store_true',
                    help='Log mode: trust the log as the sole keepers list. Skips face_match walk. '
                         'Deletes source for every MATCH line.')
    ap.add_argument('--exclude-cluster', action='append', default=[],
                    help='Manifest mode: skip rows in this cluster category (repeatable). '
                         'Common values: no_face, noise.')
    ap.add_argument('--commit', action='store_true', help='Actually delete (rm). Implies real deletion.')
    ap.add_argument('--trash', action='store_true', help='Move to macOS Trash instead of rm.')
    ap.add_argument('--limit', type=int, default=5000, help='Refuse to act on more than this many files (default 5000).')
    ap.add_argument('--transcript', default=None, help='Path to write transcript log (default auto in cwd).')
    args = ap.parse_args()

    if args.commit and args.trash:
        print('Use either --commit (rm) or --trash, not both.', file=sys.stderr)
        sys.exit(2)
    if bool(args.log) == bool(args.manifest):
        print('Exactly one of --log or --manifest must be specified.', file=sys.stderr)
        sys.exit(2)
    if args.log and not args.log_only and not args.face_match:
        print('--face-match is required with --log unless --log-only is set.', file=sys.stderr)
        sys.exit(2)

    dry_run = not (args.commit or args.trash)
    use_manifest = bool(args.manifest)

    source_root = Path(args.source).resolve()
    face_match = Path(args.face_match).resolve() if args.face_match else None
    input_path = Path(args.manifest if use_manifest else args.log).resolve()

    if not input_path.is_file():
        sys.exit(f'Input not found: {input_path}')
    if face_match is not None and not face_match.is_dir():
        sys.exit(f'face_match folder not found: {face_match}')
    if not source_root.is_dir():
        sys.exit(f'source root not found: {source_root}')

    print(f'Input:      {input_path} ({"manifest" if use_manifest else "log"})')
    print(f'face_match: {face_match or "(n/a)"}')
    print(f'source:     {source_root}')
    if use_manifest:
        print(f'Excluding cluster categories: {args.exclude_cluster or "(none)"}')
    else:
        print(f'Excluding face_match subdirs: {args.exclude or "(none)"}')
    print(f'Mode:       {"dry-run" if dry_run else ("trash" if args.trash else "rm -f")}')
    print()

    plan = []
    missing_in_log = []
    missing_source = []
    size_mismatch = []

    if use_manifest:
        print('Parsing manifest.csv...')
        basename_to_source, cat_counts = parse_manifest_for_sources(
            input_path, args.container_source, str(source_root),
            exclude_categories=args.exclude_cluster)
        print(f'  total rows in source: {sum(cat_counts.values())}')
        for cat in sorted(cat_counts.keys()):
            marker = ' (excluded)' if cat in (args.exclude_cluster or []) else ''
            print(f'    {cat}: {cat_counts[cat]}{marker}')
        print(f'  candidates after filtering: {len(basename_to_source)}')
        print()
        for basename, source_path in basename_to_source.items():
            if not os.path.isfile(source_path):
                missing_source.append((basename, source_path))
                continue
            plan.append((source_path, None))

    elif args.log_only:
        print('Parsing log for basename → source mapping...')
        basename_to_source = parse_log_for_sources(input_path, args.container_source, str(source_root))
        print(f'  found {len(basename_to_source)} unique basenames in MATCH lines')
        print('Mode: --log-only — using log entries as the keepers list.')
        print()
        for basename, source_path in basename_to_source.items():
            if not os.path.isfile(source_path):
                missing_source.append((basename, source_path))
                continue
            plan.append((source_path, None))

    else:
        print('Parsing log for basename → source mapping...')
        basename_to_source = parse_log_for_sources(input_path, args.container_source, str(source_root))
        print(f'  found {len(basename_to_source)} unique basenames in MATCH lines')
        print('Walking face_match for currently-kept files...')
        kept_basenames = walk_face_match(face_match, set(args.exclude))
        print(f'  found {len(kept_basenames)} kept files (after exclude)')
        print()
        for basename, face_match_copy_path in kept_basenames.items():
            source_path = basename_to_source.get(basename)
            if source_path is None:
                missing_in_log.append(basename)
                continue
            if not os.path.isfile(source_path):
                missing_source.append((basename, source_path))
                continue
            # Sanity: source and face_match copy should be same size.
            try:
                src_size = os.path.getsize(source_path)
                copy_size = os.path.getsize(face_match_copy_path)
            except OSError as e:
                size_mismatch.append((basename, source_path, f'stat error: {e}'))
                continue
            if src_size != copy_size:
                size_mismatch.append((basename, source_path, f'src={src_size} copy={copy_size}'))
                continue
            plan.append((source_path, face_match_copy_path))

    print('Plan summary:')
    print(f'  will delete : {len(plan)}')
    print(f'  not in log  : {len(missing_in_log)} (kept but no MATCH line — investigate)')
    print(f'  source gone : {len(missing_source)} (already deleted in a prior run, probably)')
    print(f'  size mismatch: {len(size_mismatch)} (refused for safety)')
    print()

    if missing_in_log:
        print('First 5 kept-but-not-in-log:')
        for b in missing_in_log[:5]:
            print(f'  {b}')
        print()
    if size_mismatch:
        print('First 5 size mismatches:')
        for b, p, note in size_mismatch[:5]:
            print(f'  {b}: {note}  ({p})')
        print()

    if len(plan) > args.limit:
        print(f'ABORT: plan size {len(plan)} exceeds --limit {args.limit}.')
        print('       Re-run with --limit set higher if this is intended.')
        sys.exit(3)

    if dry_run:
        print('=== DRY RUN — nothing was deleted ===')
        print('Would delete (first 20):')
        for src, _ in plan[:20]:
            print(f'  {src}')
        if len(plan) > 20:
            print(f'  ... and {len(plan) - 20} more')
        print()
        print('To execute: re-run with --trash (recoverable) or --commit (permanent rm).')
        return

    # Real action.
    transcript_path = args.transcript or f'delete_recovered_transcript_{datetime.now().strftime("%Y%m%dT%H%M%S")}.log'
    print(f'Acting on {len(plan)} files. Transcript → {transcript_path}')

    deleted = 0
    failed = 0
    with open(transcript_path, 'w') as tx:
        tx.write(f'# delete_recovered run at {datetime.now().isoformat()}\n')
        tx.write(f'# mode: {"trash" if args.trash else "rm"}\n')
        tx.write(f'# total plan: {len(plan)}\n\n')
        for src, copy in plan:
            try:
                if args.trash:
                    ok = send_to_trash(src)
                    if not ok:
                        raise RuntimeError('osascript trash failed')
                else:
                    os.remove(src)
                deleted += 1
                tx.write(f'OK\t{src}\n')
                if deleted % 100 == 0:
                    print(f'  ... {deleted} deleted')
            except Exception as e:
                failed += 1
                tx.write(f'FAIL\t{src}\t{e}\n')

    print()
    print(f'Done. Deleted: {deleted}. Failed: {failed}.')
    print(f'Transcript: {transcript_path}')


if __name__ == '__main__':
    main()
