"""Reload residual checkpoints and verify archived primary metrics exactly."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(directory))

import torch

from state_interpreter.forecast_evaluation import aligned_squared_errors, forecast_metrics
from state_interpreter.residual_gru import ResidualGRUStateInterpreter
from run_gru_interpreter_exploratory import center_sequence, load_latents, sha256, to_latent


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / cfg["output_directory"]
    verification_path = output / "verification_record.json"
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite verification record")
    report_path = output / "experiment_report.json"
    metrics_path = output / "per_seed_bearing_metrics.csv"
    protected = [report_path, metrics_path, output / "training_history.csv", output / "training_summary.csv"]
    checkpoints = {seed: output / f"residual_seed_{seed}_checkpoint.pt" for seed in cfg["random_seeds"]}
    protected.extend(checkpoints.values())
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for path in checkpoints.values():
        if sha256(path) != report["checkpoint_sha256"][path.name]:
            raise ValueError("checkpoint hash mismatch")

    by_bearing, columns = load_latents(ROOT / cfg["input_latents"])
    raw = {bearing: to_latent(rows, columns) for bearing, rows in by_bearing.items()}
    centered = {bearing: center_sequence(sequence, cfg["calibration_steps"]) for bearing, sequence in raw.items()}
    fit = torch.cat([centered[bearing] for bearing in cfg["train_bearings"]])
    mean, std = fit.mean(0), fit.std(0, unbiased=False).clamp_min(1e-6)
    standardized = {bearing: (sequence - mean) / std for bearing, sequence in centered.items()}
    archived = {
        (int(row["seed"]), row["bearing_id"]): row
        for row in read_csv(metrics_path) if row["arm"] == "residual"
    }
    metric_fields = ("gru_mse", "persistence_mse", "mse_ratio", "skill", "gru_win_fraction", "gru_median_step_mse", "persistence_median_step_mse")
    differences = []
    prefix_differences = []
    for seed, path in checkpoints.items():
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        if not torch.equal(mean, checkpoint["feature_mean"]) or not torch.equal(std, checkpoint["feature_std"]):
            raise ValueError("saved normalization mismatch")
        model = ResidualGRUStateInterpreter(embedding_dim=len(columns), hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for bearing, sequence in standardized.items():
            with torch.no_grad():
                forecast = model(sequence[None]).next_embedding_prediction[0]
                k = min(cfg["prefix_check_length"], len(sequence) - 1)
                prefix = model(sequence[:k][None]).next_embedding_prediction[0]
            prefix_differences.append(float((prefix - forecast[:k]).abs().max()))
            model_errors, persistence_errors = aligned_squared_errors(sequence, forecast)
            actual = forecast_metrics(model_errors, persistence_errors, target_start_step=cfg["primary_target_start_step"])
            reference = archived[(seed, bearing)]
            differences.extend(abs(float(actual[field]) - float(reference[field])) for field in metric_fields)
    if max(differences, default=0) != 0 or max(prefix_differences, default=0) > cfg["prefix_tolerance"]:
        raise RuntimeError("saved checkpoint verification failed")
    if not all(sha256(ROOT / relative) == digest for relative, digest in before.items()):
        raise RuntimeError("protected experiment artifact changed")
    verification_path.write_text(json.dumps({
        "status": "verified", "checkpoint_count": len(checkpoints),
        "metric_comparison_count": len(differences),
        "maximum_metric_absolute_difference": max(differences, default=0),
        "maximum_prefix_absolute_difference": max(prefix_differences, default=0),
        "protected_sha256": before, "protected_files_unchanged": True,
    }, indent=2), encoding="utf-8")
    print(verification_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
