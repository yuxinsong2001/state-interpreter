"""Read a small number of real XJTU-SY measurements and report their contract."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

# Allow this repository-owned script to run from a fresh checkout without an
# editable install.  Installed package imports remain unchanged.
repository_src = Path(__file__).resolve().parents[1] / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import torch

from state_interpreter.adapters import XJTUSYDatasetAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--condition", default="35Hz12kN")
    parser.add_argument("--bearing", default="Bearing1_1")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    adapter = XJTUSYDatasetAdapter(
        args.root,
        conditions=[args.condition],
        bearings=[args.bearing],
    )
    measurements = list(islice(adapter.iter_measurements(), args.limit))
    if len(measurements) != args.limit:
        raise RuntimeError(
            f"expected {args.limit} measurements, got {len(measurements)}"
        )

    result = {
        "episode_id": measurements[0].episode_id,
        "count": len(measurements),
        "step_ids": [item.step_id for item in measurements],
        "time_indices_minutes": [item.time_index for item in measurements],
        "shapes": [list(item.vibration.shape) for item in measurements],
        "dtypes": [str(item.vibration.dtype) for item in measurements],
        "all_finite": [bool(torch.isfinite(item.vibration).all()) for item in measurements],
        "operating_conditions": measurements[0].operating_conditions,
        "source_files": [item.metadata["source_path"] for item in measurements],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
