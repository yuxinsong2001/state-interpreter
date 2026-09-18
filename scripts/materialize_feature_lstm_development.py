"""Materialize preregistered Condition 3 development features exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.feature_lstm_protocol import FeatureLSTMDevelopmentPolicy
from state_interpreter.feature_materialization import materialize_development_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()

    policy = FeatureLSTMDevelopmentPolicy.load(args.config, ROOT)
    output_dir = ROOT / policy.config["materialization"]["cache_directory"]
    manifest, quality = materialize_development_cache(
        policy=policy,
        dataset_root=Path(args.dataset_root),
        token=args.confirm,
        output_dir=output_dir,
        progress=lambda message: print(message, flush=True),
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest": str(manifest),
                "quality_report": str(quality),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
