"""Joint two-arm evaluation with a safe non-blind rehearsal mode."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from state_interpreter import RelativeTemporalStateInterpreter
from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.data import materialize_stft_data
from state_interpreter.experiment_config import (
    file_sha256,
    load_cross_condition_config,
    verify_frozen_source_artifacts,
)


def _load_script(name: str, filename: str):
    path = repository_root / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARM_A = _load_script("joint_arm_a_helpers", "run_direct_generalization_v2.py")
ARM_B = _load_script(
    "joint_arm_b_helpers", "run_adapted_encoder_development_v2.py"
)
BLIND_CONFIRMATION = "EXECUTE_BEARING2_5_ONCE"


def resolve_evaluation_bearing(config, mode: str, confirmation: str | None) -> str:
    if mode == "rehearsal":
        return config.arm_b_validation_bearing
    if mode != "blind":
        raise ValueError(f"unsupported evaluation mode: {mode}")
    if confirmation != BLIND_CONFIRMATION:
        raise ValueError("blind mode requires the exact one-time confirmation token")
    return config.split.blind_holdout


def verify_preprocessing(checkpoint: dict[str, object], config, arm: str) -> None:
    stft = checkpoint["stft"]
    actual = (
        int(stft["n_fft"]),
        int(stft["win_length"]),
        int(stft["hop_length"]),
        tuple(int(value) for value in stft["output_size"]),
    )
    expected = (
        config.n_fft,
        config.win_length,
        config.hop_length,
        config.output_size,
    )
    if actual != expected:
        raise ValueError(f"{arm} preprocessing differs from locked config")


def infer_arm(
    *,
    arm: str,
    condition: str,
    bearing: str,
    materialized,
    model,
    standardizer,
    calibration_steps: int,
    temporal_window: int,
    batch_size: int,
) -> list[dict[str, object]]:
    inputs = standardizer.transform(materialized.tensors)
    latent_batches: list[torch.Tensor] = []
    mse_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            output = model(batch)
            latent_batches.append(output.z.cpu())
            mse_batches.append(
                (output.reconstruction - batch).square().mean(dim=(1, 2, 3)).cpu()
            )
    latent = torch.cat(latent_batches)
    reconstruction_mse = torch.cat(mse_batches)
    if not bool(torch.isfinite(latent).all()) or not bool(
        torch.isfinite(reconstruction_mse).all()
    ):
        raise RuntimeError(f"non-finite output from {arm}")

    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=latent.shape[1],
        calibration_steps=calibration_steps,
        temporal_window=temporal_window,
    )
    denominator = max(1, max(materialized.step_ids))
    rows: list[dict[str, object]] = []
    for index, z in enumerate(latent):
        state = interpreter.update(z)
        row: dict[str, object] = {
            "arm": arm,
            "condition": condition,
            "bearing_id": bearing,
            "episode_id": materialized.episode_ids[index],
            "step_id": materialized.step_ids[index],
            "normalized_lifetime": materialized.step_ids[index] / denominator,
            "phase": "calibrating" if state is None else "ready",
            "level": "" if state is None else float(state.level),
            "trend": "" if state is None else float(state.trend),
            "movement": "" if state is None else float(state.movement),
            "reconstruction_mse": float(reconstruction_mse[index]),
            "source_path": materialized.source_paths[index],
        }
        row.update({f"z_{i}": float(value) for i, value in enumerate(z)})
        rows.append(row)
    return rows


def compare(arm_a: dict[str, object], arm_b: dict[str, object]) -> dict[str, object]:
    rho_a = float(arm_a["level_spearman_rho"])
    rho_b = float(arm_b["level_spearman_rho"])
    return {
        "arm_a_level_spearman_rho": rho_a,
        "arm_b_level_spearman_rho": rho_b,
        "arm_b_minus_arm_a_rho": rho_b - rho_a,
        "arm_a_reconstruction_mse_mean": float(
            arm_a["reconstruction_mse_mean"]
        ),
        "arm_b_reconstruction_mse_mean": float(
            arm_b["reconstruction_mse_mean"]
        ),
    }


def save_plot(path: Path, rows_a, rows_b, bearing: str, *, mode: str) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for rows, label in ((rows_a, "Arm A"), (rows_b, "Arm B")):
        ready = [row for row in rows if row["phase"] == "ready"]
        for axis, field in zip(axes, ("level", "trend", "movement")):
            axis.plot(
                [int(row["step_id"]) for row in ready],
                [float(row[field]) for row in ready],
                label=label,
                linewidth=1.5,
            )
    for axis, label in zip(axes, ("Level", "Trend", "Movement")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Measurement step")
    title_mode = "blind evaluation" if mode == "blind" else "evaluation rehearsal"
    figure.suptitle(f"Joint {title_mode}: {bearing}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm-b-checkpoint", required=True)
    parser.add_argument("--arm-b-training-record", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("rehearsal", "blind"), required=True)
    parser.add_argument("--confirm-blind")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    started = time.monotonic()
    config_path = Path(args.config).resolve()
    arm_b_checkpoint_path = Path(args.arm_b_checkpoint).resolve()
    arm_b_training_record = Path(args.arm_b_training_record).resolve()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    config = load_cross_condition_config(config_path)
    record_dir = repository_root / "records" / config_path.stem
    existing_blind_records = tuple(record_dir.glob("*_joint_blind_evaluation.json"))
    if args.mode == "blind" and existing_blind_records:
        names = ", ".join(path.name for path in existing_blind_records)
        raise RuntimeError(f"blind evaluation is already recorded: {names}")
    bearing = resolve_evaluation_bearing(config, args.mode, args.confirm_blind)
    blind_read = args.mode == "blind"
    source_hashes = verify_frozen_source_artifacts(config, repository_root)
    source_checkpoint_path = (repository_root / config.source_checkpoint.path).resolve()
    source_checkpoint, model_a, preprocessor_a, standardizer_a = (
        ARM_A.load_frozen_model(source_checkpoint_path)
    )
    arm_b_checkpoint, arm_b_hash = ARM_B.validate_arm_b_artifacts(
        config=config,
        config_path=config_path,
        checkpoint_path=arm_b_checkpoint_path,
        training_record_path=arm_b_training_record,
    )
    _, model_b, preprocessor_b, standardizer_b = ARM_A.load_frozen_model(
        arm_b_checkpoint_path
    )
    verify_preprocessing(source_checkpoint, config, "arm A")
    verify_preprocessing(arm_b_checkpoint, config, "arm B")
    preprocessing_a = (
        preprocessor_a.n_fft,
        preprocessor_a.win_length,
        preprocessor_a.hop_length,
        preprocessor_a.output_size,
    )
    preprocessing_b = (
        preprocessor_b.n_fft,
        preprocessor_b.win_length,
        preprocessor_b.hop_length,
        preprocessor_b.output_size,
    )
    if preprocessing_a != preprocessing_b:
        raise ValueError("arms do not share identical preprocessing")

    # One materialization is deliberately shared by both arms.
    adapter = XJTUSYDatasetAdapter(
        args.root, conditions=[config.target_condition], bearings=[bearing]
    )
    print(f"MATERIALIZING_ONCE mode={args.mode} bearing={bearing}", flush=True)
    materialized = materialize_stft_data(adapter, preprocessor_a)
    rows_a = infer_arm(
        arm="arm_a",
        condition=config.target_condition,
        bearing=bearing,
        materialized=materialized,
        model=model_a,
        standardizer=standardizer_a,
        calibration_steps=config.calibration_steps,
        temporal_window=config.temporal_window,
        batch_size=args.batch_size,
    )
    rows_b = infer_arm(
        arm="arm_b",
        condition=config.target_condition,
        bearing=bearing,
        materialized=materialized,
        model=model_b,
        standardizer=standardizer_b,
        calibration_steps=config.calibration_steps,
        temporal_window=config.temporal_window,
        batch_size=args.batch_size,
    )
    summary_a = ARM_A.summarize_ready_states(rows_a)
    summary_b = ARM_A.summarize_ready_states(rows_b)
    comparison = compare(summary_a, summary_b)

    output_dir.mkdir(parents=True)
    all_rows = rows_a + rows_b
    with (output_dir / "joint_states.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    save_plot(
        output_dir / "joint_state_plot.png",
        rows_a,
        rows_b,
        bearing,
        mode=args.mode,
    )

    run_date = time.strftime("%Y-%m-%d")
    run_date_tag = time.strftime("%Y%m%d")

    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": run_date,
            "verification_status": "UNVERIFIED",
            "version_label": "exp_result_v1",
        },
        "experiment_id": f"joint_evaluation_{args.mode}_{run_date_tag}",
        "type": "analysis",
        "status": "completed",
        "mode": args.mode,
        "bearing": bearing,
        "target_condition": config.target_condition,
        "single_shared_materialization": True,
        "raw_sample_count": len(materialized.tensors),
        "joint_state_row_count": len(all_rows),
        "blind_holdout": config.split.blind_holdout,
        "blind_holdout_read": blind_read,
        "config_sha256": file_sha256(config_path),
        "arm_a_checkpoint_sha256": source_hashes["source_checkpoint"],
        "arm_b_checkpoint_sha256": arm_b_hash,
        "interpreter_parameters": {
            "calibration_steps": config.calibration_steps,
            "temporal_window": config.temporal_window,
            "state_fields": list(config.state_fields),
        },
        "arm_a_summary": summary_a,
        "arm_b_summary": summary_b,
        "comparison": comparison,
        "duration_seconds": time.monotonic() - started,
        "outputs": ["joint_states.csv", "joint_state_plot.png", "joint_report.json"],
        "interpretation_gate": {
            "passed": False,
            "reason": (
                "Workflow rehearsal only; no new scientific conclusion."
                if args.mode == "rehearsal"
                else "Blind result generated; scientific interpretation required."
            ),
        },
    }
    with (output_dir / "joint_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
