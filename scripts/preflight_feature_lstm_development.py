"""Validate Feature LSTM preregistration without accepting a dataset path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.feature_lstm_protocol import FeatureLSTMDevelopmentPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    policy = FeatureLSTMDevelopmentPolicy.load(args.config, ROOT)
    verified = policy.verify_frozen_artifacts()
    report = {
        "status": "preflight_passed_no_condition3_data_read",
        "experiment_id": policy.config["experiment_id"],
        "permitted_materialization": policy.config["split"]["development"],
        "protected_validation": policy.config["split"]["validation"],
        "protected_blind": policy.config["split"]["blind"],
        "outer_lobo_folds": policy.config["outer_lobo_folds"],
        "fixed_training": policy.config["training"],
        "validation_gate": policy.config["evaluation"]["validation_gate"],
        "cache_schema": policy.config["materialization"]["cache_schema"],
        "verified_frozen_artifacts": verified,
        "next_step_requires_exact_token": policy.config["tokens"][
            "materialize_development"
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
