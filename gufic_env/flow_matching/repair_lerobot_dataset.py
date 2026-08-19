#!/usr/bin/env python
import argparse
import json
import math
import shutil
from pathlib import Path


PARQUET_MAGIC = b"PAR1"


def parquet_error(path: Path) -> str | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            head = f.read(4)
            if size >= 4:
                f.seek(-4, 2)
                tail = f.read(4)
            else:
                tail = b""
    except OSError as exc:
        return f"read failed: {exc}"

    if size < 8:
        return f"too small ({size} bytes)"
    if head != PARQUET_MAGIC or tail != PARQUET_MAGIC:
        return "missing parquet PAR1 magic bytes"
    return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_info(root: Path) -> dict:
    with (root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def save_info(root: Path, info: dict) -> None:
    path = root / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
        f.write("\n")


def episode_path(root: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return root / data_path.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def replace_column(table, name: str, values):
    import pyarrow as pa

    if name not in table.column_names:
        return table
    column_index = table.schema.get_field_index(name)
    array = pa.array(values, type=table.schema.field(column_index).type)
    return table.set_column(column_index, name, array)


def rewrite_episode_parquet(src: Path, dst: Path, new_episode_index: int, global_start: int) -> int:
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(src)
    length = table.num_rows
    frame_index = np.arange(length, dtype=np.int64)
    global_index = frame_index + int(global_start)

    table = replace_column(table, "episode_index", np.full(length, new_episode_index, dtype=np.int64))
    table = replace_column(table, "frame_index", frame_index)
    table = replace_column(table, "index", global_index)

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return length


def update_index_stats(stats: dict, key: str, min_value: int, max_value: int, count: int) -> None:
    if key not in stats:
        return
    mean = (min_value + max_value) / 2.0 if count else 0.0
    # Std of consecutive integers [min_value, max_value].
    std = math.sqrt((count * count - 1) / 12.0) if count > 1 else 0.0
    stats[key] = {
        "min": [int(min_value)],
        "max": [int(max_value)],
        "mean": [float(mean)],
        "std": [float(std)],
        "count": [int(count)],
    }


def update_episode_stats(row: dict, new_episode_index: int, global_start: int, length: int) -> dict:
    row = json.loads(json.dumps(row))
    row["episode_index"] = int(new_episode_index)
    stats = row.get("stats", {})
    if length > 0:
        update_index_stats(stats, "episode_index", new_episode_index, new_episode_index, length)
        update_index_stats(stats, "frame_index", 0, length - 1, length)
        update_index_stats(stats, "index", global_start, global_start + length - 1, length)
    return row


def copy_tasks_and_extra_meta(src_root: Path, dst_root: Path) -> None:
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    for src in src_meta.iterdir():
        if src.name in {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}:
            continue
        dst = dst_meta / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def copy_episode_videos(src_root: Path, dst_root: Path, old_index: int, new_index: int, chunks_size: int) -> None:
    videos_root = src_root / "videos"
    if not videos_root.exists():
        return

    old_name = f"episode_{old_index:06d}.mp4"
    new_name = f"episode_{new_index:06d}.mp4"
    new_chunk = f"chunk-{new_index // chunks_size:03d}"
    for src in videos_root.rglob(old_name):
        rel = src.relative_to(videos_root)
        rel_parts = list(rel.parts)
        rel_parts[0] = new_chunk
        rel_parts[-1] = new_name
        dst = dst_root / "videos" / Path(*rel_parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a clean LeRobot dataset copy by dropping corrupted parquet "
            "episodes and reindexing the remaining episodes."
        )
    )
    parser.add_argument("src_root", help="Original LeRobot dataset root.")
    parser.add_argument("dst_root", help="New clean dataset root to create.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()
    if not (src_root / "meta" / "info.json").exists():
        raise SystemExit(f"Missing LeRobot meta/info.json under {src_root}")
    if dst_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Destination exists: {dst_root}. Pass --overwrite to replace it.")
        shutil.rmtree(dst_root)

    info = load_info(src_root)
    chunks_size = int(info.get("chunks_size", 1000))
    episodes = read_jsonl(src_root / "meta" / "episodes.jsonl")
    episode_stats = {
        int(row["episode_index"]): row
        for row in read_jsonl(src_root / "meta" / "episodes_stats.jsonl")
        if "episode_index" in row
    }

    good = []
    bad = []
    for row in episodes:
        old_index = int(row["episode_index"])
        path = episode_path(src_root, info, old_index)
        reason = parquet_error(path)
        if reason is None:
            good.append((old_index, row, path))
        else:
            bad.append((old_index, path, reason))

    if not good:
        raise SystemExit("No valid parquet episodes were found; nothing to repair.")

    copy_tasks_and_extra_meta(src_root, dst_root)

    new_episodes = []
    new_episode_stats = []
    total_frames = 0
    for new_index, (old_index, row, src_path) in enumerate(good):
        dst_path = episode_path(dst_root, info, new_index)
        length = rewrite_episode_parquet(
            src=src_path,
            dst=dst_path,
            new_episode_index=new_index,
            global_start=total_frames,
        )
        copy_episode_videos(src_root, dst_root, old_index, new_index, chunks_size)

        new_row = dict(row)
        new_row["episode_index"] = int(new_index)
        new_row["length"] = int(length)
        new_episodes.append(new_row)

        if old_index in episode_stats:
            new_episode_stats.append(
                update_episode_stats(
                    episode_stats[old_index],
                    new_episode_index=new_index,
                    global_start=total_frames,
                    length=length,
                )
            )
        total_frames += length

    new_info = dict(info)
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_frames"] = int(total_frames)
    new_info["total_chunks"] = max(1, math.ceil(len(new_episodes) / chunks_size))
    new_info["splits"] = {"train": f"0:{len(new_episodes)}"}

    save_info(dst_root, new_info)
    write_jsonl(dst_root / "meta" / "episodes.jsonl", new_episodes)
    if new_episode_stats:
        write_jsonl(dst_root / "meta" / "episodes_stats.jsonl", new_episode_stats)

    print(f"source: {src_root}")
    print(f"clean:  {dst_root}")
    print(f"kept {len(good)} episodes, dropped {len(bad)} corrupted episodes")
    print(f"total_frames={total_frames}")
    if bad:
        print("dropped episodes:")
        for old_index, path, reason in bad:
            print(f"  - episode {old_index:06d}: {path} ({reason})")


if __name__ == "__main__":
    main()
