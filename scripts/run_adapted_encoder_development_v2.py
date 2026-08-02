"""Run arm B development inference and compare it with arm A."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

repository_root = Path(__file__).resolve().parents[1]
repository_src = repository_root / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import torch
import matplotlib.pyplot as plt

from state_interpreter.experiment_config import file_sha256, load_cross_condition_config


def _load_arm_a_module():
    script = repository_root / "scripts" / "run_direct_generalization_v2.py"
    spec = importlib.util.spec_from_file_location("arm_a_shared_analysis", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared analysis helpers: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARM_A = _load_arm_a_module()


def save_plot(path: Path, rows: list[dict[str, object]], bearings: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for bearing in bearings:
        ready = [
            row
            for row in rows
            if row["bearing_id"] == bearing and row["phase"] == "ready"
        ]
        for axis, field in zip(axes, ("level", "trend", "movement")):
            axis.plot(
                [int(row["step_id"]) for row in ready],
                [float(row[field]) for row in ready],
                linewidth=1.3,
                label=bearing,
            )
    for axis, label in zip(axes, ("Level", "Trend", "Movement")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Measurement step")
    fig.suptitle("Arm B development: target-adapted encoder on 37.5Hz11kN")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def validate_arm_b_artifacts(
    *, config, config_path: Path, checkpoint_path: Path, training_record_path: Path
) -> tuple[dict[str, object], str]:
    config_digest = file_sha256(config_path)
    checkpoint_digest = file_sha256(checkpoint_path)
    record = load_json(training_record_path)
    if record.get("status") != "completed":
        raise ValueError("arm B training record is not completed")
    if record.get("config_id") != config.config_id:
        raise ValueError("arm B training record uses a different config")
    if record.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("arm B checkpoint hash differs from training record")
    blind = record.get("blind_holdout")
    if not isinstance(blind, dict) or blind.get("raw_data_read") is not False:
        raise ValueError("training record does not preserve the blind holdout")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config_id") != config.config_id:
        raise ValueError("arm B checkpoint uses a different config")
    if checkpoint.get("config_sha256") != config_digest:
        raise ValueError("arm B checkpoint config hash mismatch")
    if checkpoint.get("blind_holdout_read") is not False:
        raise ValueError("arm B checkpoint does not preserve the blind holdout")
    if checkpoint.get("condition") != config.target_condition:
        raise ValueError("arm B checkpoint condition mismatch")
    if tuple(checkpoint["split"]["train"]) != config.arm_b_train_bearings:
        raise ValueError("arm B checkpoint training split mismatch")
    if tuple(checkpoint["split"]["validation"]) != (config.arm_b_validation_bearing,):
        raise ValueError("arm B checkpoint validation split mismatch")
    if int(checkpoint["model"]["latent_dim"]) != config.embedding_dim:
        raise ValueError("arm B checkpoint latent dimension mismatch")
    return checkpoint, checkpoint_digest


def load_arm_a_summaries(
    path: Path, *, config_hash: str, bearings: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    report = load_json(path)
    if report.get("status") != "completed":
        raise ValueError("arm A development report is not completed")
    if report.get("config_sha256") != config_hash:
        raise ValueError("arm A report config hash mismatch")
    if report.get("blind_holdout_read") is not False:
        raise ValueError("arm A report does not preserve the blind holdout")
    if tuple(report.get("evaluated_bearings", ())) != bearings:
        raise ValueError("arm A and arm B development bearings differ")
    summaries = report.get("bearing_summaries")
    if not isinstance(summaries, list):
        raise ValueError("arm A bearing summaries are missing")
    return {str(row["bearing_id"]): row for row in summaries}


def compare_summaries(
    arm_a: dict[str, dict[str, object]], arm_b: list[dict[str, object]]
) -> list[dict[str, object]]:
    comparison = []
    for row in arm_b:
        bearing = str(row["bearing_id"])
        if bearing not in arm_a:
            raise ValueError(f"arm A summary missing bearing: {bearing}")
        rho_a = float(arm_a[bearing]["level_spearman_rho"])
        rho_b = float(row["level_spearman_rho"])
        comparison.append(
            {
                "bearing_id": bearing,
                "arm_a_level_spearman_rho": rho_a,
                "arm_b_level_spearman_rho": rho_b,
                "arm_b_minus_arm_a_rho": rho_b - rho_a,
                "arm_a_reconstruction_mse_mean": float(
                    arm_a[bearing]["reconstruction_mse_mean"]
                ),
                "arm_b_reconstruction_mse_mean": float(
                    row["reconstruction_mse_mean"]
                ),
            }
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-record", required=True)
    parser.add_argument("--arm-a-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    started = time.monotonic()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    training_record_path = Path(args.training_record).resolve()
    arm_a_report_path = Path(args.arm_a_report).resolve()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    config = load_cross_condition_config(config_path)
    bearings = config.bearings_for("arm_b_development")
    config.split.assert_excludes_holdout(bearings, purpose="arm B development")
    checkpoint, checkpoint_digest = validate_arm_b_artifacts(
        config=config,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        training_record_path=training_record_path,
    )
    config_digest = file_sha256(config_path)
    arm_a = load_arm_a_summaries(
        arm_a_report_path, config_hash=config_digest, bearings=bearings
    )
    _, model, preprocessor, standardizer = ARM_A.load_frozen_model(checkpoint_path)

    output_dir.mkdir(parents=True)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for bearing in bearings:
        print(f"PROCESSING {bearing}", flush=True)
        rows = ARM_A.process_bearing(
            data_root=Path(args.root),
            condition=config.target_condition,
            bearing=bearing,
            model=model,
            preprocessor=preprocessor,
            standardizer=standardizer,
            calibration_steps=config.calibration_steps,
            temporal_window=config.temporal_window,
            batch_size=args.batch_size,
        )
        all_rows.extend(rows)
        summary = {"bearing_id": bearing, **ARM_A.summarize_ready_states(rows)}
        summaries.append(summary)
        print(
            f"COMPLETED {bearing} samples={len(rows)} "
            f"rho={summary['level_spearman_rho']:.6f}",
            flush=True,
        )

    comparison = compare_summaries(arm_a, summaries)
    with (output_dir / "development_states.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "bearing_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "arm_a_vs_b_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    save_plot(output_dir / "development_state_plot.png", all_rows, bearings)

    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-08-02",
            "verification_status": "UNVERIFIED",
            "version_label": "exp_result_v1",
        },
        "experiment_id": "arm_b_adapted_encoder_dev_20260802",
        "type": "analysis",
        "status": "completed",
        "config": str(config_path),
        "config_sha256": config_digest,
        "arm_b_checkpoint": str(checkpoint_path),
        "arm_b_checkpoint_sha256": checkpoint_digest,
        "arm_b_training_record": str(training_record_path),
        "arm_a_development_report": str(arm_a_report_path),
        "target_condition": config.target_condition,
        "evaluated_bearings": list(bearings),
        "blind_holdout": config.split.blind_holdout,
        "blind_holdout_read": False,
        "autoencoder_trained_or_updated_during_inference": False,
        "normalization_refitted_during_inference": False,
        "interpreter_parameters": {
            "calibration_steps": config.calibration_steps,
            "temporal_window": config.temporal_window,
            "state_fields": list(config.state_fields),
        },
        "sample_count": len(all_rows),
        "bearing_summaries": summaries,
        "arm_a_vs_b_comparison": comparison,
        "mean_arm_b_minus_arm_a_rho": sum(
            float(row["arm_b_minus_arm_a_rho"]) for row in comparison
        ) / len(comparison),
        "duration_seconds": time.monotonic() - started,
        "outputs": [
            "development_states.csv",
            "bearing_summary.csv",
            "arm_a_vs_b_comparison.csv",
            "development_state_plot.png",
            "development_report.json",
        ],
        "interpretation_gate": {
            "passed": False,
            "reason": "Development comparison generated; blind evaluation remains locked.",
        },
    }
    with (output_dir / "development_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
