"""
In-place reading and writing of results JSONL files, shared by the metric scripts.

Taken from scripts/extract_gender.py: rows are read and written with the json module, so every field
that a script does not set round-trips exactly (a pandas round-trip would, e.g., cut floats to 10
digits), and a file is only ever replaced atomically (temp file + os.replace), so a crash cannot leave
it truncated.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    """Read a results file; every row needs a `response`."""
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path} (has the generation job finished?)")
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = sum("response" not in row for row in rows)
    if missing:
        raise ValueError(f"{path.name}: {missing} rows have no 'response' field")
    return rows


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    """Rewrite `path` through a temp file in the same directory, so it is never left half-written."""
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def finite_or_none(value) -> float | None:
    """A metric that could not be computed is stored as null rather than a bare NaN token (invalid JSON)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None
