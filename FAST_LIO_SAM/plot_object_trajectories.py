#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TrajectoryPoint = Tuple[int, float, float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot object trajectories from object_tracking.txt and label each track with its id."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="Log/object_tracking.txt",
        help="Input tracking result file. Default: Log/object_tracking.txt",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="Log/object_trajectories_xy.png",
        help="Output image path. Default: Log/object_trajectories_xy.png",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Only plot top N ids ranked by frame count. 0 means plot all ids.",
    )
    parser.add_argument(
        "--min-frame-count",
        type=int,
        default=1,
        help="Only plot ids whose frame count is at least this value.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        default=None,
        help="Only plot the specified object ids, for example: --ids 0 234 522",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        default=(12.0, 10.0),
        help="Figure size in inches. Default: 12 10",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive window after saving the image.",
    )
    return parser.parse_args()


def parse_tracking_file(file_path: Path) -> Dict[int, List[TrajectoryPoint]]:
    trajectories: DefaultDict[int, List[TrajectoryPoint]] = defaultdict(list)

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                raise ValueError(
                    f"Invalid line in {file_path} at line {line_no}: {raw_line.rstrip()}"
                )

            try:
                frame_id = int(float(parts[0]))
                object_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                yaw = float(parts[5])
                score = float(parts[6])
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse line in {file_path} at line {line_no}: {raw_line.rstrip()}"
                ) from exc

            trajectories[object_id].append((frame_id, x, y, z, yaw, score))

    for object_id in trajectories:
        trajectories[object_id].sort(key=lambda item: item[0])

    return dict(trajectories)


def filter_trajectories(
    trajectories: Dict[int, List[TrajectoryPoint]],
    top_n: int,
    min_frame_count: int,
    selected_ids: List[int] | None,
) -> Dict[int, List[TrajectoryPoint]]:
    selected_id_set = set(selected_ids) if selected_ids else None

    filtered_items = [
        (object_id, points)
        for object_id, points in trajectories.items()
        if selected_id_set is None or object_id in selected_id_set
        if len(points) >= min_frame_count
    ]

    filtered_items.sort(key=lambda item: (-len(item[1]), item[0]))

    if top_n > 0 and selected_id_set is None:
        filtered_items = filtered_items[:top_n]

    return dict(filtered_items)


def plot_trajectories(
    trajectories: Dict[int, List[TrajectoryPoint]], output_path: Path, figsize: Tuple[float, float]
) -> None:
    if not trajectories:
        raise ValueError("No trajectories satisfy the plotting conditions.")

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab20")

    sorted_ids = sorted(trajectories.keys())
    for index, object_id in enumerate(sorted_ids):
        points = trajectories[object_id]
        xs = [point[1] for point in points]
        ys = [point[2] for point in points]
        color = cmap(index % 20)

        ax.plot(xs, ys, linewidth=1.8, color=color, alpha=0.9)
        ax.scatter(xs[0], ys[0], s=18, color=color, marker="o")
        ax.scatter(xs[-1], ys[-1], s=28, color=color, marker="x")

        label_x = xs[-1]
        label_y = ys[-1]
        ax.annotate(
            str(object_id),
            xy=(label_x, label_y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.75},
        )

    ax.set_title("Object Trajectories (XY)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.axis("equal")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    trajectories = parse_tracking_file(input_path)
    trajectories = filter_trajectories(
        trajectories,
        top_n=args.top_n,
        min_frame_count=args.min_frame_count,
        selected_ids=args.ids,
    )
    plot_trajectories(trajectories, output_path, figsize=tuple(args.figsize))

    print(f"Saved trajectory plot to: {output_path}")
    print(f"Plotted object count: {len(trajectories)}")

    if args.show:
        img = plt.imread(output_path)
        plt.figure(figsize=tuple(args.figsize))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
