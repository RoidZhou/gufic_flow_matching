#!/usr/bin/env python
import argparse
import os
from pathlib import Path


def check_parquet_file(path: Path) -> str | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            head = f.read(4)
            if size >= 4:
                f.seek(-4, os.SEEK_END)
                tail = f.read(4)
            else:
                tail = b""
    except OSError as exc:
        return f"read failed: {exc}"

    if size < 8:
        return f"too small ({size} bytes)"
    if head != b"PAR1" or tail != b"PAR1":
        return "missing parquet PAR1 magic bytes"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find corrupted/non-parquet episode files in a LeRobot dataset."
    )
    parser.add_argument("dataset_root", help="LeRobot dataset root containing data/ and meta/.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser()
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise SystemExit(f"Missing data directory: {data_dir}")

    parquet_paths = sorted(data_dir.rglob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No .parquet files found under: {data_dir}")

    bad = []
    for path in parquet_paths:
        reason = check_parquet_file(path)
        if reason is not None:
            bad.append((path, reason))

    print(f"checked {len(parquet_paths)} parquet files under {data_dir}")
    if not bad:
        print("all parquet files have valid PAR1 header/footer")
        return

    print(f"found {len(bad)} bad parquet files:")
    for path, reason in bad:
        print(f"  - {path}: {reason}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
