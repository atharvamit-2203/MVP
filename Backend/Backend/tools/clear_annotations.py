#!/usr/bin/env python3
"""Backup and clear the Backend/annotations directory.

This script creates a timestamped backup of the entire annotations folder
and then deletes saved image files and clears annotations.jsonl.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
import sys


def main() -> int:
    repo_backend = Path(__file__).resolve().parents[2]
    ann_dir = repo_backend / "annotations"
    if not ann_dir.exists():
        print(f"No annotations directory found at: {ann_dir}")
        return 0

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = ann_dir.parent / f"annotations_backup_{ts}"
    print(f"Creating backup: {backup_dir}")
    try:
        shutil.copytree(ann_dir, backup_dir)
    except Exception as exc:
        print("Failed to create backup:", exc)
        return 2

    # Remove image files but keep README or other metadata files
    removed = 0
    for p in ann_dir.iterdir():
        if p.is_file() and p.name != "annotations.jsonl" and not p.name.lower().endswith(".md"):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass

    ann_file = ann_dir / "annotations.jsonl"
    if ann_file.exists():
        try:
            ann_file.write_text("", encoding="utf-8")
            print(f"Cleared file: {ann_file}")
        except Exception as exc:
            print("Failed to clear annotations.jsonl:", exc)
            return 3
    else:
        ann_file.write_text("", encoding="utf-8")
        print(f"Created empty annotations file: {ann_file}")

    print(f"Done. Removed {removed} files. Backup at: {backup_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())