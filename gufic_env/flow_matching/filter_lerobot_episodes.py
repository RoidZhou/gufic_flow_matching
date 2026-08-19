#!/usr/bin/env python3
"""Delete selected episodes from one LeRobot v2.1 dataset in place."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_ROOT = Path("/root/autodl-tmp/ur5_rg2_real_smolvla_dataset_force_boltnut_speed_627")
DEFAULT_DROP_EPISODES = "6, 12, 16, 18, 22, 23, 25, 27, 42, 49, 58, 67"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete selected episodes and compact a LeRobot dataset in place."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--drop-episodes",
        default=DEFAULT_DROP_EPISODES,
        help="Comma-separated episode indexes/ranges to remove, e.g. 6,7,18,52,66.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Required acknowledgement because this permanently modifies --root.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=4)
        file.write("\n")


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record))
            file.write("\n")


def parse_episode_indexes(value: str) -> set[int]:
    indexes: set[int] = set()
    try:
        for item in value.split(","):
            start_end = [part.strip() for part in item.split("-", maxsplit=1)]
            if len(start_end) == 1:
                indexes.add(int(start_end[0]))
            else:
                start, end = (int(part) for part in start_end)
                if end < start:
                    raise ValueError("range end precedes range start")
                indexes.update(range(start, end + 1))
    except ValueError as exc:
        raise ValueError("--drop-episodes must look like 6,7,18,52,66 or 0-19.") from exc
    if any(index < 0 for index in indexes):
        raise ValueError("Episode indexes must be non-negative.")
    return indexes


def episode_parquet_path(root: Path, info: dict, episode_index: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
    )


def replace_int_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Parquet table is missing required column: {name}")
    return table.set_column(
        column_index,
        name,
        pa.array(values, type=table.schema.field(column_index).type),
    )


def range_std(length: int) -> float:
    return math.sqrt((length**2 - 1) / 12) if length > 1 else 0.0


def rewrite_index_stats(stats: dict, episode_index: int, first_index: int, task_index: int, length: int) -> None:
    count = [length]
    stats["episode_index"] = {
        "min": [episode_index], "max": [episode_index], "mean": [float(episode_index)], "std": [0.0], "count": count
    }
    stats["task_index"] = {
        "min": [task_index], "max": [task_index], "mean": [float(task_index)], "std": [0.0], "count": count
    }
    stats["frame_index"] = {
        "min": [0], "max": [length - 1], "mean": [(length - 1) / 2], "std": [range_std(length)], "count": count
    }
    stats["index"] = {
        "min": [first_index],
        "max": [first_index + length - 1],
        "mean": [first_index + (length - 1) / 2],
        "std": [range_std(length)],
        "count": count,
    }


def main() -> None:
    args = parse_args()
    if not args.in_place:
        raise ValueError("This operation is destructive. Pass --in-place to continue.")

    root = args.root.resolve()
    info_path = root / "meta/info.json"
    episodes_path = root / "meta/episodes.jsonl"
    stats_path = root / "meta/episodes_stats.jsonl"
    if not all(path.is_file() for path in (info_path, episodes_path, stats_path)):
        raise FileNotFoundError(f"Not a complete LeRobot dataset root: {root}")

    info = read_json(info_path)
    episodes = read_jsonl(episodes_path)
    stats_by_index = {
        int(record["episode_index"]): record for record in read_jsonl(stats_path)
    }
    drop_episodes = parse_episode_indexes(args.drop_episodes)
    known_indexes = {int(episode["episode_index"]) for episode in episodes}
    unknown_indexes = drop_episodes - known_indexes
    if unknown_indexes:
        raise ValueError(f"Cannot drop missing episode indexes: {sorted(unknown_indexes)}")
    missing_stats = known_indexes - set(stats_by_index)
    if missing_stats:
        raise ValueError(f"Missing episode stats for indexes: {sorted(missing_stats)[:10]}")

    # Delete old targets first so every compacted parquet filename is free.
    for old_episode_index in sorted(drop_episodes):
        path = episode_parquet_path(root, info, old_episode_index)
        if not path.is_file():
            raise FileNotFoundError(f"Episode parquet does not exist: {path}")
        path.unlink()

    kept_episodes = [episode for episode in episodes if int(episode["episode_index"]) not in drop_episodes]
    output_episodes: list[dict] = []
    output_stats: list[dict] = []
    total_frames = 0

    for new_episode_index, episode in enumerate(kept_episodes):
        old_episode_index = int(episode["episode_index"])
        source_path = episode_parquet_path(root, info, old_episode_index)
        table = pq.read_table(source_path)
        length = table.num_rows
        if length != int(episode["length"]):
            raise ValueError(f"Length mismatch for {source_path}: metadata={episode['length']}, parquet={length}")

        table = replace_int_column(table, "episode_index", np.full(length, new_episode_index, dtype=np.int64))
        table = replace_int_column(table, "frame_index", np.arange(length, dtype=np.int64))
        table = replace_int_column(table, "index", np.arange(total_frames, total_frames + length, dtype=np.int64))
        output_path = episode_parquet_path(root, info, new_episode_index)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".tmp.parquet")
        pq.write_table(table, temporary_path)
        temporary_path.replace(output_path)
        if source_path != output_path:
            source_path.unlink()

        output_episode = dict(episode)
        output_episode["episode_index"] = new_episode_index
        output_episode["length"] = length
        output_episodes.append(output_episode)

        stats_record = dict(stats_by_index[old_episode_index])
        stats_record["episode_index"] = new_episode_index
        stats_record["stats"] = dict(stats_record["stats"])
        task_values = table.column("task_index").to_numpy(zero_copy_only=False)
        task_index = int(task_values[0]) if length else 0
        rewrite_index_stats(stats_record["stats"], new_episode_index, total_frames, task_index, length)
        output_stats.append(stats_record)
        total_frames += length
        print(f"Kept {old_episode_index:03d} as {new_episode_index:03d} ({length} frames)")

    info["total_episodes"] = len(output_episodes)
    info["total_frames"] = total_frames
    info["total_chunks"] = math.ceil(len(output_episodes) / int(info["chunks_size"])) if output_episodes else 0
    info["splits"] = {"train": f"0:{len(output_episodes)}"}
    write_json(info_path, info)
    write_jsonl(episodes_path, output_episodes)
    write_jsonl(stats_path, output_stats)
    print(f"Deleted {sorted(drop_episodes)}. {len(output_episodes)} episodes remain with {total_frames} frames.")


if __name__ == "__main__":
    main()

    '''
    python filter_lerobot_episodes.py --in-place
    '''