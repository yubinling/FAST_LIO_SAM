#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count top-N object ids by appeared frames in object tracking results."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="Log/object_tracking.txt",
        help="Input tracking result file or directory. Default: Log/object_tracking.txt",
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=10,
        help="Show top N object ids ranked by number of appeared frames.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Optional output report path.",
    )
    return parser.parse_args()


def resolve_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.rglob("*.txt"))
        if files:
            return files
        raise FileNotFoundError(f"No .txt files found in directory: {input_path}")

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def parse_tracking_files(files: Iterable[Path]) -> Dict[int, Set[int]]:
    id_to_frames: Dict[int, Set[int]] = defaultdict(set)

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid line in {file_path} at line {line_no}: {raw_line.rstrip()}"
                    )

                try:
                    frame_id = int(float(parts[0]))
                    object_id = int(float(parts[1]))
                except ValueError as exc:
                    raise ValueError(
                        f"Failed to parse frame/object id in {file_path} at line {line_no}: "
                        f"{raw_line.rstrip()}"
                    ) from exc

                id_to_frames[object_id].add(frame_id)

    return id_to_frames


def rank_ids(id_to_frames: Dict[int, Set[int]], top_n: int) -> List[Tuple[int, List[int]]]:
    ranked = sorted(
        ((object_id, sorted(frames)) for object_id, frames in id_to_frames.items()),
        key=lambda item: (-len(item[1]), item[0]),
    )
    return ranked[:top_n]


def format_frame_ranges(frames: List[int]) -> str:
    if not frames:
        return ""

    ranges: List[str] = []
    start = frames[0]
    end = frames[0]

    for frame in frames[1:]:
        if frame == end + 1:
            end = frame
            continue

        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        start = frame
        end = frame

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ", ".join(ranges)


def build_report(files: List[Path], ranked: List[Tuple[int, List[int]]], total_ids: int) -> str:
    lines: List[str] = []
    lines.append(f"Input files: {', '.join(str(path) for path in files)}")
    lines.append(f"Total unique object ids: {total_ids}")
    lines.append(f"Top ids shown: {len(ranked)}")
    lines.append("")

    for index, (object_id, frames) in enumerate(ranked, start=1):
        frame_ranges = format_frame_ranges(frames)
        lines.append(
            f"{index}. object_id={object_id}, frame_count={len(frames)}, "
            f"first_frame={frames[0]}, last_frame={frames[-1]}"
        )
        lines.append(f"   frames: {frame_ranges}")

    if not ranked:
        lines.append("No valid object ids found.")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    files = resolve_input_files(input_path)
    id_to_frames = parse_tracking_files(files)
    ranked = rank_ids(id_to_frames, args.top_n)
    report = build_report(files, ranked, len(id_to_frames))

    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
