"""Validate the Source-DANN development contract without reading a cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(config: dict) -> dict:
    if config["status"] != "preregistered_preflight_no_cache_read":
        raise ValueError("invalid Source-DANN preregistration status")
    expected = ["Bearing3_1", "Bearing3_2", "Bearing3_3"]
    if config["development_bearings"] != expected:
        raise ValueError("development split changed")
    if config["protected_validation"] != ["Bearing3_4"]:
        raise ValueError("validation protection changed")
    if config["protected_blind"] != ["Bearing3_5"]:
        raise ValueError("blind protection changed")
    if config["causal_inference"]["future_context_allowed"] is not False:
        raise ValueError("future context must remain prohibited")
    arms = config["paired_arms"]
    if arms["source_only"]["maximum_grl_coefficient"] != 0.0:
        raise ValueError("source-only control changed")
    if arms["source_dann"]["maximum_grl_coefficient"] != 0.1:
        raise ValueError("Source-DANN coefficient changed")
    if not arms["all_other_model_training_and_evaluation_settings_identical"]:
        raise ValueError("paired-arm equality must remain enforced")
    if config["evaluation_gate"]["combined_rule"] != (
        "source_dann_health_gate_and_identity_gate_must_both_pass"
    ):
        raise ValueError("combined gate changed")
    verified = {}
    for artifact in config["frozen_artifacts"]:
        path = ROOT / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen artifact mismatch: {artifact['path']}")
        verified[artifact["path"]] = artifact["sha256"]
    return {
        "status": "source_dann_preflight_passed_no_cache_read",
        "experiment_id": config["experiment_id"],
        "development_bearings": expected,
        "protected_bearings": ["Bearing3_4", "Bearing3_5"],
        "paired_arms": arms,
        "model": config["model"],
        "training": config["training"],
        "evaluation_gate": config["evaluation_gate"],
        "verified_frozen_artifacts": verified,
        "next_step_requires_exact_token": config["execution_token"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(validate(config), indent=2))


if __name__ == "__main__":
    main()

