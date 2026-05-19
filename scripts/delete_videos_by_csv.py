#!/usr/bin/env python3
"""
Delete video files based on a validate_videos.sh CSV.

The CSV has tier1_status, tier2_status, path, duration_sec, codec for every
file ffprobe touched. This script filters by status + duration and feeds
the resulting list into the same safety harness as the other delete scripts
(dry-run default, --commit for real, --limit guardrail, transcript log).

Typical 4K Stogram cleanup:
  python delete_videos_by_csv.py \\
      --csv video_validation_*.csv \\
      --max-duration 60 \\
      --include-broken
  # then re-run with --commit when the plan looks right.

Triage the long-tail later by inverting the filter:
  python delete_videos_by_csv.py \\
      --csv video_validation_*.csv \\
      --min-duration 60 --max-duration 300
"""

import argparse
import csv
import os
import sys
from datetime import datetime


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True,
                    help='video_validation_*.csv from validate_videos.sh')
    ap.add_argument('--min-duration', type=float, default=0,
                    help='Only include files >= this duration (seconds). Default 0.')
    ap.add_argument('--max-duration', type=float, default=None,
                    help='Only include files <= this duration (seconds). Default no upper bound.')
    ap.add_argument('--include-broken', action='store_true',
                    help='Include files with tier1_status=BROKEN (no duration; recovery garbage).')
    ap.add_argument('--include-suspicious', action='store_true',
                    help='Include files with tier1_status=SUSPICIOUS (valid header, no duration).')
    ap.add_argument('--codec', action='append', default=[],
                    help='Only include files with this codec (repeatable). Empty = any.')
    ap.add_argument('--limit', type=int, default=5000,
                    help='Refuse to act on more than N files. Default 5000.')
    ap.add_argument('--commit', action='store_true',
                    help='Actually delete (rm). Dry-run if omitted.')
    ap.add_argument('--transcript', default=None,
                    help='Path to write transcript log (default auto in cwd).')
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f'CSV not found: {args.csv}')

    print(f'CSV:          {args.csv}')
    print(f'Min duration: {args.min_duration}s')
    print(f'Max duration: {args.max_duration if args.max_duration is not None else "(no limit)"}s')
    print(f'Include BROKEN:    {args.include_broken}')
    print(f'Include SUSPICIOUS: {args.include_suspicious}')
    print(f'Codecs filter:     {args.codec or "(any)"}')
    print(f'Mode:         {"COMMIT (will delete)" if args.commit else "DRY RUN"}')
    print()

    plan = []
    skipped_status = {'OK': 0, 'BROKEN': 0, 'SUSPICIOUS': 0, 'OTHER': 0}
    skipped_duration = 0
    skipped_codec = 0
    missing_source = 0

    with open(args.csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t1 = row.get('tier1_status', '').strip()
            path = row.get('path', '').strip()
            dur_str = row.get('duration_sec', '').strip()
            codec = row.get('codec', '').strip()

            # Status filter
            if t1 == 'BROKEN':
                if not args.include_broken:
                    skipped_status['BROKEN'] += 1
                    continue
            elif t1 == 'SUSPICIOUS':
                if not args.include_suspicious:
                    skipped_status['SUSPICIOUS'] += 1
                    continue
            elif t1 != 'OK':
                skipped_status['OTHER'] += 1
                continue

            # Codec filter (only meaningful for OK files; BROKEN has no codec)
            if args.codec and t1 == 'OK' and codec not in args.codec:
                skipped_codec += 1
                continue

            # Duration filter (only meaningful when ffprobe gave us a number)
            if dur_str and dur_str.lower() not in ('', 'n/a'):
                try:
                    dur = float(dur_str)
                except ValueError:
                    dur = None
            else:
                dur = None

            if dur is not None:
                if dur < args.min_duration:
                    skipped_duration += 1
                    continue
                if args.max_duration is not None and dur > args.max_duration:
                    skipped_duration += 1
                    continue
            elif args.min_duration > 0 or args.max_duration is not None:
                # User specified duration filter but this file has no duration.
                # Only include if --include-broken (which already passed status check above).
                if t1 != 'BROKEN':
                    skipped_duration += 1
                    continue

            if not os.path.isfile(path):
                missing_source += 1
                continue

            plan.append(path)

    total_seen = sum(skipped_status.values()) + skipped_duration + skipped_codec + missing_source + len(plan)
    print(f'Plan summary:')
    print(f'  total CSV rows : {total_seen}')
    print(f'  will delete    : {len(plan)}')
    print(f'  skipped status : {dict(skipped_status)} (filter)')
    print(f'  skipped duration: {skipped_duration} (out of range)')
    print(f'  skipped codec   : {skipped_codec}')
    print(f'  source missing  : {missing_source} (deleted by prior run, probably)')
    print()

    if not plan:
        print('Nothing to delete. Adjust filters and re-run.')
        return

    if len(plan) > args.limit:
        print(f'ABORT: plan size {len(plan)} exceeds --limit {args.limit}.')
        print(f'       Re-run with --limit {len(plan)} (or higher) if intended.')
        sys.exit(3)

    # Total bytes about to be freed (approximate from current stat).
    total_bytes = 0
    for p in plan[:5000]:  # cap stat calls for big plans
        try:
            total_bytes += os.path.getsize(p)
        except OSError:
            pass
    sampled = min(len(plan), 5000)
    extrapolated = int(total_bytes / sampled * len(plan)) if sampled > 0 else 0
    print(f'Approx total size to free: {extrapolated / (1024**3):.2f} GiB '
          f'(extrapolated from {sampled} samples)')
    print()

    if not args.commit:
        print('=== DRY RUN — nothing was deleted ===')
        print('Sample (first 10):')
        for p in plan[:10]:
            print(f'  {p}')
        if len(plan) > 10:
            print(f'  ... and {len(plan) - 10} more')
        print()
        print('To execute: re-run with --commit')
        return

    # Real deletion.
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    transcript_path = args.transcript or f'delete_videos_transcript_{ts}.log'
    print(f'Acting on {len(plan)} files. Transcript → {transcript_path}')

    deleted = 0
    failed = 0
    with open(transcript_path, 'w') as tx:
        tx.write(f'# delete_videos_by_csv run at {datetime.now().isoformat()}\n')
        tx.write(f'# CSV: {args.csv}\n')
        tx.write(f'# plan size: {len(plan)}\n\n')
        for path in plan:
            try:
                os.remove(path)
                deleted += 1
                tx.write(f'OK\t{path}\n')
                if deleted % 1000 == 0:
                    print(f'  ... {deleted} deleted')
            except Exception as e:
                failed += 1
                tx.write(f'FAIL\t{path}\t{e}\n')

    print()
    print(f'Done. Deleted: {deleted}. Failed: {failed}.')
    print(f'Transcript: {transcript_path}')


if __name__ == '__main__':
    main()
