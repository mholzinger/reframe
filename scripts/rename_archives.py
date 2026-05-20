#!/usr/bin/env python3
"""
Infer the 'intended name' of each recovered archive from its entry paths
and (optionally) rename the archives accordingly.

Recovered archives have sector-range filenames (e.g., 6961889528-6969368562_4536.ZIP).
The actual content often has a meaningful top-level folder inside the archive,
like '2001 - Sex Shrines - Tera Patrick/' — that folder name is what the
archive was originally called.

This script reads the verify_archives.sh TSV (which already contains the
first-10 entry paths for each archive), finds the common top-level directory
among them, and either prints a rename plan or applies it.

Confidence levels:
  high   — every sampled entry shares the same top-level directory
  medium — majority (>= 50%) of sampled entries share it; rest are stragglers
  none   — files are in root, or no consistent top-level

Usage:
  python rename_archives.py --tsv FILE              # dry-run preview
  python rename_archives.py --tsv FILE --commit     # apply
  python rename_archives.py --tsv FILE --confidence high   # only rename high-confidence
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter


# Replace filesystem-unsafe characters in inferred names
SAFE_NAME = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


def infer_name(entries):
    """Return (name, confidence) where confidence is 'high', 'medium', or 'none'."""
    if not entries:
        return None, 'none'

    top_levels = []
    for e in entries:
        e = e.strip()
        if not e:
            continue
        slash = e.find('/')
        top_levels.append(e[:slash] if slash > 0 else None)

    if not top_levels:
        return None, 'none'

    counts = Counter(top_levels)
    most_common, freq = counts.most_common(1)[0]

    if most_common is None or not most_common.strip():
        return None, 'none'

    if freq == len(top_levels):
        return most_common, 'high'
    if freq * 2 >= len(top_levels):
        return most_common, 'medium'
    return None, 'none'


def sanitize_basename(name, ext):
    cleaned = SAFE_NAME.sub('_', name).strip(' _.')
    # Limit length — most filesystems are fine up to 255, leave headroom
    if len(cleaned) > 200:
        cleaned = cleaned[:200].rstrip(' _.')
    if not cleaned:
        return None
    return f'{cleaned}{ext}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tsv', required=True, help='verify_archives.sh output TSV')
    ap.add_argument('--commit', action='store_true', help='Actually rename files (default dry-run)')
    ap.add_argument('--confidence', choices=['high', 'medium', 'any'], default='high',
                    help='Minimum confidence to act on. Default: high (safest)')
    ap.add_argument('--limit', type=int, default=10000, help='Refuse to rename more than this. Default 10000.')
    ap.add_argument('--transcript', default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.tsv):
        sys.exit(f'TSV not found: {args.tsv}')

    plan = []  # list of (src_path, target_path, confidence)
    no_signal = 0
    seen = {}  # track collisions in inferred names

    with open(args.tsv, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('status') != 'VALID':
                continue
            entries = (row.get('first_10_entries') or '').split('|')
            entries = [e for e in entries if e]
            name, confidence = infer_name(entries)
            if confidence == 'none':
                no_signal += 1
                continue
            if args.confidence == 'high' and confidence != 'high':
                continue
            if args.confidence == 'medium' and confidence not in ('high', 'medium'):
                continue

            src = row['path']
            ext = os.path.splitext(src)[1]
            sanitized = sanitize_basename(name, ext)
            if not sanitized:
                no_signal += 1
                continue

            target = os.path.join(os.path.dirname(src), sanitized)
            # Handle collisions: append _N suffix
            base_stem, base_ext = os.path.splitext(sanitized)
            i = 1
            while target == src or os.path.exists(target) or target in seen:
                target = os.path.join(os.path.dirname(src), f'{base_stem}_{i}{base_ext}')
                i += 1
            seen[target] = src

            plan.append((src, target, confidence))

    # Summary
    high = sum(1 for _, _, c in plan if c == 'high')
    medium = sum(1 for _, _, c in plan if c == 'medium')
    print(f'Total VALID archives processed: {len(plan) + no_signal}')
    print(f'  rename candidates: {len(plan)}')
    print(f'    high confidence: {high}')
    print(f'    medium confidence: {medium}')
    print(f'  no signal (skipped): {no_signal}')
    print()

    if not plan:
        print('Nothing to rename. Try lowering --confidence to medium or any.')
        return

    if len(plan) > args.limit:
        print(f'ABORT: plan size {len(plan)} exceeds --limit {args.limit}.')
        sys.exit(3)

    if not args.commit:
        print('=== DRY RUN — no files renamed ===')
        print('Sample (first 20):')
        for src, tgt, c in plan[:20]:
            print(f'  [{c:>6}] {os.path.basename(src)} -> {os.path.basename(tgt)}')
        if len(plan) > 20:
            print(f'  ... and {len(plan) - 20} more')
        print()
        print('To execute: re-run with --commit')
        return

    # Commit.
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    transcript_path = args.transcript or f'rename_archives_transcript_{ts}.log'
    print(f'Renaming {len(plan)} archives. Transcript → {transcript_path}')

    renamed = 0
    failed = 0
    with open(transcript_path, 'w') as tx:
        tx.write(f'# rename_archives run at {datetime.now().isoformat()}\n')
        tx.write(f'# TSV: {args.tsv}\n')
        tx.write(f'# confidence filter: {args.confidence}\n\n')
        for src, tgt, c in plan:
            try:
                os.rename(src, tgt)
                renamed += 1
                tx.write(f'OK\t{c}\t{src}\t->\t{tgt}\n')
                if renamed % 100 == 0:
                    print(f'  ... {renamed} renamed')
            except Exception as e:
                failed += 1
                tx.write(f'FAIL\t{c}\t{src}\t{e}\n')

    print()
    print(f'Done. Renamed: {renamed}. Failed: {failed}.')
    print(f'Transcript: {transcript_path}')


if __name__ == '__main__':
    main()
