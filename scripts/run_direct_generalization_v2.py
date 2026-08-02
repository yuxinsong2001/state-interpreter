"""Run arm A development inference with the frozen source AutoEncoder."""

from __future__ import annotations

import argparse
import csv
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
import numpy as np
import torch
from scipy.stats import spearmanr

from state_interpreter import RelativeTemporalStateInterpreter
from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.data import ChannelStandardizer, materialize_stft_data
from state_interpreter.encoders import SmallConvAutoEncoder
from state_interpreter.experiment_config import (
    file_sha256,
    load_cross_condition_config,
    verify_frozen_source_artifacts,
)
from state_interpreter.preprocessing import LogSTFTPreprocessor


def directional_monotonicity(values: np.ndarray) -> float:
    differences = np.diff(values)
    nonzero = np.sign(differences[np.abs(differences) > 1e-12])
    return 0.0 if len(nonzero) == 0 else float(nonzero.mean())


def roughness_smoothness(values: np.ndarray) -> float:
    first = np.diff(values)
    if len(first) < 2:
        return 1.0
    roughness = np.sum(np.abs(np.diff(first))) / (
        np.sum(np.abs(first)) + 1e-12
    )
    return float(1.0 / (1.0 + roughness))


def summarize_ready_states(rows: list[dict[str, object]]) -> dict[str, object]:
    ready = [row for row in rows if row["phase"] == "ready"]
    if len(ready) < 2:
        raise ValueError("at least two READY states are required")
    lifetime = np.asarray([float(row["normalized_lifetime"]) for row in ready])
    level = np.asarray([float(row["level"]) for row in ready])
    trend = np.asarray([float(row["trend"]) for row in ready])
    movement = np.asarray([float(row["movement"]) for row in ready])
    return {
        "total_samples": len(rows),
        "calibration_samples": len(rows) - len(ready),
        "ready_samples": len(ready),
        "ready_fraction": len(ready) / len(rows),
        "level_spearman_rho": float(spearmanr(lifetime, level).statistic),
        "level_directional_monotonicity": directional_monotonicity(level),
        "level_smoothness": roughness_smoothness(level),
        "trend_noise": float(np.std(np.diff(trend))),
        "level_first": float(level[0]),
        "level_last": float(level[-1]),
        "movement_max": float(movement.max()),
        "movement_max_step": int(
            ready[int(np.argmax(movement))]["step_id"]
        ),
        "movement_last": float(movement[-1]),
        "reconstruction_mse_mean": float(
            np.mean([float(row["reconstruction_mse"]) for row in rows])
        ),
    }


def load_frozen_model(checkpoint_path: Path) -> tuple[
    dict[str, object], SmallConvAutoEncoder, LogSTFTPreprocessor, ChannelStandardizer
]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint["model"]
    model = SmallConvAutoEncoder(
        input_channels=int(model_config["input_channels"]),
        latent_dim=int(model_config["latent_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stft = checkpoint["stft"]
    preprocessor = LogSTFTPreprocessor(
        n_fft=int(stft["n_fft"]),
        win_length=int(stft["win_length"]),
        hop_length=int(stft["hop_length"]),
        output_size=tuple(stft["output_size"]),
    )
    standardizer = ChannelStandardizer(
        mean=checkpoint["standardizer"]["mean"],
        std=checkpoint["standardizer"]["std"],
    )
    return checkpoint, model, preprocessor, standardizer


def process_bearing(
    *,
    data_root: Path,
    condition: str,
    bearing: str,
    model: SmallConvAutoEncoder,
    preprocessor: LogSTFTPreprocessor,
    standardizer: ChannelStandardizer,
    calibration_steps: int,
    temporal_window: int,
    batch_size: int,
) -> list[dict[str, object]]:
    adapter = XJTUSYDatasetAdapter(
        data_root, conditions=[condition], bearings=[bearing]
    )
    materialized = materialize_stft_data(adapter, preprocessor)
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
        raise RuntimeError(f"non-finite model output for {bearing}")

    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=latent.shape[1],
        calibration_steps=calibration_steps,
        temporal_window=temporal_window,
    )
    maximum_step = max(materialized.step_ids)
    denominator = max(1, maximum_step)
    rows: list[dict[str, object]] = []
    for index, z in enumerate(latent):
        state = interpreter.update(z)
        row: dict[str, object] = {
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
    fig.suptitle("Arm A development: frozen source system on 37.5Hz11kN")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    started = time.monotonic()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    config = load_cross_condition_config(config_path)
    verified = verify_frozen_source_artifacts(config, repository_root)
    bearings = config.bearings_for("arm_a_development")
    config.split.assert_excludes_holdout(bearings, purpose="arm A development")
    checkpoint_path = (repository_root / config.source_checkpoint.path).resolve()
    checkpoint, model, preprocessor, standardizer = load_frozen_model(
        checkpoint_path
    )
    if int(checkpoint["model"]["latent_dim"]) != config.embedding_dim:
        raise ValueError("checkpoint latent dimension differs from v2.1 config")
    if str(checkpoint["condition"]) != config.source_condition:
        raise ValueError("checkpoint source condition differs from v2.1 config")

    output_dir.mkdir(parents=True)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for bearing in bearings:
        print(f"PROCESSING {bearing}", flush=True)
        rows = process_bearing(
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
        summaries.append(
            {
                "bearing_id": bearing,
                **summarize_ready_states(rows),
            }
        )
        print(
            f"COMPLETED {bearing} samples={len(rows)} "
            f"rho={summaries[-1]['level_spearman_rho']:.6f}",
            flush=True,
        )

    state_fields = list(all_rows[0])
    with (output_dir / "development_states.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=state_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "bearing_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    save_plot(output_dir / "development_state_plot.png", all_rows, bearings)

    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-08-02",
            "verification_status": "UNVERIFIED",
            "version_label": "exp_result_v1",
        },
        "experiment_id": "arm_a_direct_generalization_dev_20260802",
        "type": "analysis",
        "status": "completed",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": verified["source_checkpoint"],
        "source_config_sha256": verified["source_config"],
        "source_condition": config.source_condition,
        "target_condition": config.target_condition,
        "evaluated_bearings": list(bearings),
        "blind_holdout": config.split.blind_holdout,
        "blind_holdout_read": False,
        "autoencoder_trained_or_updated": False,
        "normalization_refitted": False,
        "interpreter_parameters": {
            "calibration_steps": config.calibration_steps,
            "temporal_window": config.temporal_window,
            "state_fields": list(config.state_fields),
        },
        "sample_count": len(all_rows),
        "bearing_summaries": summaries,
        "duration_seconds": time.monotonic() - started,
        "outputs": [
            "development_states.csv",
            "bearing_summary.csv",
            "development_state_plot.png",
            "development_report.json",
        ],
        "interpretation_gate": {
            "passed": False,
            "reason": "Development metrics generated; scientific interpretation remains required.",
        },
    }
    with (output_dir / "development_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

