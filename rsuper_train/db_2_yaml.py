#!/usr/bin/env python3
# -------------------------------------------------------------
# dump_kv_to_yaml.py
# -------------------------------------------------------------
"""
Read a SQLite key/value store and export one YAML file per row.

Usage
-----
python dump_kv_to_yaml.py --db_folder /path/to/db_or_folder \
                          --dest      /where/to/write/yaml
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import yaml


def find_db(path: Path) -> Path:
    """Return a concrete *.db file from *path* (file or directory)."""
    if path.is_file():
        return path
    db_files = list(path.glob("*.db"))
    if not db_files:
        sys.exit(f"[error] no .db file found in {path}")
    if len(db_files) > 1:
        print(f"[warning] multiple .db files in {path}, using {db_files[0].name}")
    return db_files[0]


def export_rows(db_file: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_file) as conn:
        try:
            rows = conn.execute("SELECT key, value FROM kv;").fetchall()
        except sqlite3.DatabaseError as e:
            sys.exit(f"[error] cannot read {db_file}: {e}")

    if not rows:
        print(f"[info] {db_file} contained zero rows.")
        return

    for key, value in rows:
        fname = dest / f"{key}_crop.yaml"
        with open(fname, "w", encoding="utf‑8") as f:
            # YAML dump of a *one‑element sequence*  →   - value
            yaml.safe_dump([value], f, default_flow_style=False, sort_keys=False)
    print(f"[done] wrote {len(rows)} files into {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export kv rows to YAML files.")
    parser.add_argument("--db_folder", required=True,
                        help="Folder *or* full path to the .db file.")
    parser.add_argument("--dest",      required=True,
                        help="Destination directory for the YAML files.")
    args = parser.parse_args()

    db_path  = find_db(Path(args.db_folder).expanduser().resolve())
    dest_dir = Path(args.dest).expanduser().resolve()

    export_rows(db_path, dest_dir)


if __name__ == "__main__":
    main()