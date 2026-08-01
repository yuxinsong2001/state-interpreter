"""Replay ordered latent CSV data through the online State Interpreter."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

repository_src = Path(__file__).resolve().parents[1] / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from state_interpreter import RelativeTemporalStateInterpreter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-steps", type=int, default=10)
    parser.add_argument("--temporal-window", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    with Path(args.input).open("r", newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("input CSV contains no rows")

    z_fields = sorted(
        (name for name in source_rows[0] if name.startswith("z_")),
        key=lambda name: int(name.split("_")[1]),
    )
    bearings = list(dict.fromkeys(row["bearing_id"] for row in source_rows))
    output_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for bearing in bearings:
        rows = [row for row in source_rows if row["bearing_id"] == bearing]
        interpreter = RelativeTemporalStateInterpreter(
            embedding_dim=len(z_fields),
            calibration_steps=args.calibration_steps,
            temporal_window=args.temporal_window,
        )
        ready_levels: list[float] = []
        ready_times: list[float] = []
        for row in rows:
            z = torch.tensor([float(row[field]) for field in z_fields])
            output = interpreter.update(z)
            output_rows.append(
                {
                    "bearing_id": bearing,
                    "split": row["split"],
                    "step_id": int(row["step_id"]),
                    "normalized_lifetime": float(row["normalized_lifetime"]),
                    "phase": "calibrating" if output is None else "ready",
                    "level": "" if output is None else float(output.level),
                    "trend": "" if output is None else float(output.trend),
                    "movement": "" if output is None else float(output.movement),
                }
            )
            if output is not None:
                ready_levels.append(float(output.level))
                ready_times.append(float(row["normalized_lifetime"]))
        rho = float(spearmanr(ready_times, ready_levels).statistic)
        summaries.append(
            {
                "bearing_id": bearing,
                "split": rows[0]["split"],
                "total_samples": len(rows),
                "calibration_samples": args.calibration_steps,
                "ready_samples": len(ready_levels),
                "level_spearman_rho": rho,
                "level_first": ready_levels[0],
                "level_last": ready_levels[-1],
            }
        )

    with (output_dir / "online_states.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    with (output_dir / "bearing_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    report = {
        "status": "completed",
        "input": str(Path(args.input).resolve()),
        "sample_count": len(source_rows),
        "bearing_count": len(bearings),
        "calibration_steps": args.calibration_steps,
        "temporal_window": args.temporal_window,
        "state_fields": ["level", "trend", "movement"],
        "bearing_summaries": summaries,
        "future_information_used_after_calibration": False,
        "calibration_note": (
            "A fixed number of early observations is required before READY; "
            "no formal state is emitted during calibration."
        ),
    }
    with (output_dir / "replay_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)
    for bearing in bearings:
        ready = [
            row
            for row in output_rows
            if row["bearing_id"] == bearing and row["phase"] == "ready"
        ]
        for axis, field in zip(axes, ("level", "trend", "movement")):
            axis.plot(
                [int(row["step_id"]) for row in ready],
                [float(row[field]) for row in ready],
                label=bearing,
            )
    for axis, label in zip(axes, ("Level", "Trend", "Movement")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Measurement step")
    fig.suptitle(
        f"Online State Interpreter replay ({args.calibration_steps}-step calibration)"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "online_state_replay.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
