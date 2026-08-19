#!/usr/bin/env python3
"""Merge two LeRobot v2.1 datasets and remove selected episodes."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_FIRST_ROOT = Path("/root/autodl-tmp/ur5_rg2_real_smolvla_dataset_force_boltnut_speedup_total_627")
DEFAULT_SECOND_ROOT = Path("/root/autodl-tmp/ur5_rg2_real_smolvla_dataset_force_boltnut_speedup_total")
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/ur5_rg2_real_smolvla_dataset_force_boltnut_speedup_200")


@dataclass(frozen=True)
class EpisodeSource:
    root: Path
    episode: dict
    episode_stats: dict
    task_index_map: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two LeRobot datasets and reindex their metadata."
    )
    parser.add_argument("--first-root", type=Path, default=DEFAULT_FIRST_ROOT)
    parser.add_argument("--second-root", type=Path, default=DEFAULT_SECOND_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--drop-second-episodes",
        default="",
        help="Episode indexes/ranges removed from the second source before merging, e.g. 31.",
    )
    parser.add_argument(
        "--drop-merged-episodes",
        default="",
        help="Episode indexes/ranges removed after the two sources are merged, e.g. 0-19.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing the merged dataset.",
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
    if not value.strip():
        return indexes
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
        raise ValueError("Episode indexes must look like 31 or 0-19,31.") from exc
    if any(index < 0 for index in indexes):
        raise ValueError("Episode indexes must be non-negative.")
    return indexes


def load_task_index_map(root: Path, output_tasks: dict[str, int]) -> dict[int, int]:
    source_to_output: dict[int, int] = {}
    for record in read_jsonl(root / "meta/tasks.jsonl"):
        task = record["task"]
        output_tasks.setdefault(task, len(output_tasks))
        source_to_output[int(record["task_index"])] = output_tasks[task]
    return source_to_output


def load_episode_sources(root: Path, task_index_map: dict[int, int]) -> list[EpisodeSource]:
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    stats_by_index = {
        int(record["episode_index"]): record
        for record in read_jsonl(root / "meta/episodes_stats.jsonl")
    }
    missing_stats = [episode["episode_index"] for episode in episodes if episode["episode_index"] not in stats_by_index]
    if missing_stats:
        raise ValueError(f"{root} is missing stats for episodes: {missing_stats[:10]}")
    return [
        EpisodeSource(root, episode, stats_by_index[int(episode["episode_index"])], task_index_map)
        for episode in episodes
    ]


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


def validate_compatible(first_info: dict, second_info: dict) -> None:
    keys = ("codebase_version", "robot_type", "fps", "chunks_size", "features", "data_path", "video_path")
    for key in keys:
        if first_info.get(key) != second_info.get(key):
            raise ValueError(f"Source dataset metadata differs for {key!r}; refusing to merge.")


def main() -> None:
    args = parse_args()
    first_root = args.first_root.resolve()
    second_root = args.second_root.resolve()
    output_root = args.output_root.resolve()
    drop_second = parse_episode_indexes(args.drop_second_episodes)
    drop_merged = parse_episode_indexes(args.drop_merged_episodes)

    for root in (first_root, second_root):
        if not (root / "meta/info.json").is_file():
            raise FileNotFoundError(f"Not a LeRobot dataset root: {root}")
    if output_root in (first_root, second_root):
        raise ValueError("--output-root must be different from both source roots.")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)

    first_info = read_json(first_root / "meta/info.json")
    second_info = read_json(second_root / "meta/info.json")
    validate_compatible(first_info, second_info)

    output_tasks: dict[str, int] = {}
    first_task_map = load_task_index_map(first_root, output_tasks)
    second_task_map = load_task_index_map(second_root, output_tasks)
    first_sources = load_episode_sources(first_root, first_task_map)
    second_sources = [
        source
        for source in load_episode_sources(second_root, second_task_map)
        if int(source.episode["episode_index"]) not in drop_second
    ]
    merged_sources = first_sources + second_sources
    unknown_merged_indexes = drop_merged - set(range(len(merged_sources)))
    if unknown_merged_indexes:
        raise ValueError(f"Cannot drop missing merged episode indexes: {sorted(unknown_merged_indexes)}")
    final_sources = [
        source for merged_index, source in enumerate(merged_sources) if merged_index not in drop_merged
    ]

    output_root.mkdir(parents=True)
    (output_root / "meta").mkdir()
    output_episodes: list[dict] = []
    output_episode_stats: list[dict] = []
    total_frames = 0

    for new_episode_index, source in enumerate(final_sources):
        old_episode_index = int(source.episode["episode_index"])
        source_path = episode_parquet_path(source.root, first_info, old_episode_index)
        if not source_path.is_file():
            raise FileNotFoundError(f"Episode parquet does not exist: {source_path}")
        table = pq.read_table(source_path)
        length = table.num_rows
        if length != int(source.episode["length"]):
            raise ValueError(f"Length mismatch for {source_path}: metadata={source.episode['length']}, parquet={length}")

        old_task_indexes = table.column("task_index").to_numpy(zero_copy_only=False)
        new_task_indexes = np.asarray(
            [source.task_index_map[int(index)] for index in old_task_indexes], dtype=np.int64
        )
        table = replace_int_column(table, "episode_index", np.full(length, new_episode_index, dtype=np.int64))
        table = replace_int_column(table, "frame_index", np.arange(length, dtype=np.int64))
        table = replace_int_column(table, "index", np.arange(total_frames, total_frames + length, dtype=np.int64))
        table = replace_int_column(table, "task_index", new_task_indexes)

        output_path = episode_parquet_path(output_root, first_info, new_episode_index)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)

        output_episode = dict(source.episode)
        output_episode["episode_index"] = new_episode_index
        output_episode["length"] = length
        output_episodes.append(output_episode)

        output_stats = dict(source.episode_stats)
        output_stats["episode_index"] = new_episode_index
        output_stats["stats"] = dict(source.episode_stats["stats"])
        task_index = int(new_task_indexes[0]) if length else 0
        rewrite_index_stats(output_stats["stats"], new_episode_index, total_frames, task_index, length)
        output_episode_stats.append(output_stats)
        total_frames += length
        print(f"Wrote episode {new_episode_index:03d} from {source.root.name}/{old_episode_index:03d} ({length} frames)")

    output_info = dict(first_info)
    output_info["total_episodes"] = len(output_episodes)
    output_info["total_frames"] = total_frames
    output_info["total_tasks"] = len(output_tasks)
    output_info["total_chunks"] = math.ceil(len(output_episodes) / int(first_info["chunks_size"])) if output_episodes else 0
    output_info["splits"] = {"train": f"0:{len(output_episodes)}"}
    write_json(output_root / "meta/info.json", output_info)
    write_jsonl(output_root / "meta/episodes.jsonl", output_episodes)
    write_jsonl(output_root / "meta/episodes_stats.jsonl", output_episode_stats)
    write_jsonl(
        output_root / "meta/tasks.jsonl",
        [{"task_index": task_index, "task": task} for task, task_index in output_tasks.items()],
    )
    print(
        f"Done: {len(output_episodes)} episodes, {total_frames} frames. "
        f"Skipped second source: {sorted(drop_second)}; dropped after merge: {sorted(drop_merged)}."
    )


if __name__ == "__main__":
    main()
    '''
    python merge_lerobot_datasets.py --overwrite
    '''