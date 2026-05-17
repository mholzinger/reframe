"""
SQLite cache for face encodings + EXIF metadata.

Keyed on (path, size, mtime) so re-runs over the same drive skip work that's
already been done. Encodings stored as pickled numpy arrays in BLOB columns.

The cache is the foundation that makes the council's recommended iterative
workflow possible: reference changes, threshold tuning, and re-queries cost
seconds instead of hours.
"""

import sqlite3
import pickle
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS encodings (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    face_count INTEGER NOT NULL,
    face_encodings BLOB,          -- pickled list[np.ndarray] (128-d each)
    exif_make TEXT,
    exif_model TEXT,
    exif_serial TEXT,
    exif_software TEXT,
    exif_lens TEXT,
    exif_dt TEXT,                 -- ISO format if parseable
    image_width INTEGER,
    image_height INTEGER,
    processed_at REAL NOT NULL,
    error TEXT                    -- non-null if the file couldn't be processed
);

CREATE INDEX IF NOT EXISTS idx_face_count ON encodings(face_count);
CREATE INDEX IF NOT EXISTS idx_exif_serial ON encodings(exif_serial);
CREATE INDEX IF NOT EXISTS idx_error ON encodings(error);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open or create the cache database. Caller manages lifecycle."""
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.executescript(SCHEMA)
    # Hot-path tuning for many small writes:
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn


def is_cached(conn: sqlite3.Connection, path: str, size: int, mtime: float) -> bool:
    """True iff this exact file (path + size + mtime) has a complete row."""
    row = conn.execute(
        'SELECT 1 FROM encodings WHERE path = ? AND size = ? AND ABS(mtime - ?) < 0.001',
        (path, size, mtime),
    ).fetchone()
    return row is not None


def store(
    conn: sqlite3.Connection,
    path: str,
    size: int,
    mtime: float,
    face_encodings: list,
    exif: dict,
    image_size: Optional[tuple] = None,
    error: Optional[str] = None,
) -> None:
    """Insert or replace a cache row."""
    import time
    blob = pickle.dumps([np.asarray(e, dtype=np.float64) for e in face_encodings]) if face_encodings else None
    w, h = (image_size or (None, None))
    conn.execute(
        '''INSERT OR REPLACE INTO encodings
           (path, size, mtime, face_count, face_encodings,
            exif_make, exif_model, exif_serial, exif_software, exif_lens, exif_dt,
            image_width, image_height, processed_at, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            path, size, mtime,
            len(face_encodings or []),
            blob,
            exif.get('make'), exif.get('model'), exif.get('serial'),
            exif.get('software'), exif.get('lens'),
            exif.get('dt').isoformat() if exif.get('dt') else None,
            w, h,
            time.time(),
            error,
        ),
    )


def get_encodings(conn: sqlite3.Connection, path: str) -> Optional[list]:
    """Return the list of face encodings for a path, or None if not cached or no faces."""
    row = conn.execute(
        'SELECT face_encodings FROM encodings WHERE path = ? AND error IS NULL',
        (path,),
    ).fetchone()
    if not row or not row[0]:
        return None
    return pickle.loads(row[0])


def known_serials(conn: sqlite3.Connection, paths: Iterable[str]) -> set:
    """Return the set of EXIF SerialNumbers across the given paths."""
    serials = set()
    for p in paths:
        row = conn.execute(
            'SELECT exif_serial FROM encodings WHERE path = ?', (str(p),)
        ).fetchone()
        if row and row[0]:
            serials.add(row[0])
    return serials


def stats(conn: sqlite3.Connection) -> dict:
    """Quick numeric summary for status reporting."""
    total = conn.execute('SELECT COUNT(*) FROM encodings').fetchone()[0]
    with_faces = conn.execute('SELECT COUNT(*) FROM encodings WHERE face_count > 0').fetchone()[0]
    errored = conn.execute('SELECT COUNT(*) FROM encodings WHERE error IS NOT NULL').fetchone()[0]
    return {'total': total, 'with_faces': with_faces, 'errored': errored}
