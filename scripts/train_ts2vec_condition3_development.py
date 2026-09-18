"""Execute the preregistered compact causal TS2Vec LOBO development study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.stats import spearmanr
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.data.stable_feature_scaling import StableCausalFeatureScaler
from state_interpreter.encoders.ts2vec import (
    TS2VecEncoder,
    hierarchical_contrastive_loss,
    sample_context_views,
)
from state_interpreter.feature_lstm_experiment import backward_step_fraction
from state_interpreter.representation_diagnostics import (
    evenly_spaced_indices,
    nearest_centroid_predict,
    ridge_fit_predict,
)
from state_interpreter.ts2vec_experiment import (
    causal_context_windows,
    evenly_spaced_chunks,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def encode_causal(
    encoder: torch.nn.Module,
    sequence: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
) -> np.ndarray:
    was_training = encoder.training
    encoder.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, sequence.shape[0], batch_size):
            endpoints = np.arange(start, min(start + batch_size, sequence.shape[0]))
            contexts = causal_context_windows(
                sequence, endpoints, context_length=context_length
            )
            outputs.append(encoder(contexts)[:, -1].cpu())
    encoder.train(was_training)
    return torch.cat(outputs).numpy()


def identity_probe(
    representations: dict[str, np.ndarray],
    bearings: list[str],
    *,
    sample_count: int,
    folds: int,
) -> tuple[float, list[dict[str, object]]]:
    samples = []
    labels = []
    blocks = []
    for label, bearing in enumerate(bearings):
        indices = evenly_spaced_indices(representations[bearing].shape[0], sample_count)
        lifetime = indices / float(representations[bearing].shape[0] - 1)
        block = np.minimum((lifetime * folds).astype(int), folds - 1)
        samples.append(representations[bearing][indices])
        labels.append(np.full(indices.size, label, dtype=int))
        blocks.append(block)
    values = np.concatenate(samples)
    targets = np.concatenate(labels)
    block_ids = np.concatenate(blocks)
    predictions = np.empty_like(targets)
    rows = []
    for block in range(folds):
        test = block_ids == block
        predictions[test] = nearest_centroid_predict(
            values[~test], targets[~test], values[test]
        )
        rows.append(
            {
                "held_lifetime_block": block,
                "accuracy": float(np.mean(predictions[test] == targets[test])),
                "test_samples": int(np.count_nonzero(test)),
            }
        )
    return float(np.mean(predictions == targets)), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "preregistered_preflight_no_cache_read":
        raise ValueError("invalid TS2Vec protocol status")
    if args.confirm != config["execution_token"]:
        raise PermissionError("exact TS2Vec execution token required")
    bearings = ["Bearing3_1", "Bearing3_2", "Bearing3_3"]
    if config["development_bearings"] != bearings:
        raise ValueError("development split changed")
    if config["protected_validation"] != ["Bearing3_4"]:
        raise ValueError("validation protection changed")
    if config["protected_blind"] != ["Bearing3_5"]:
        raise ValueError("blind protection changed")
    for artifact in config["frozen_artifacts"]:
        if sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"frozen artifact mismatch: {artifact['path']}")

    final_output = (ROOT / config["output_directory"]).resolve()
    staging = final_output.with_name(final_output.name + ".staging")
    if final_output.exists() or staging.exists():
        raise FileExistsError("refusing to overwrite TS2Vec output or staging data")

    cache = (ROOT / config["input"]["cache_directory"]).resolve()
    sequences: dict[str, np.ndarray] = {}
    for entry in config["input"]["cache_entries"]:
        bearing = entry["bearing_id"]
        if bearing not in bearings:
            raise ValueError("cache entry outside development scope")
        path = cache / f"{bearing}.npz"
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"cache hash mismatch: {bearing}")
        with np.load(path, allow_pickle=False) as payload:
            features = torch.from_numpy(
                np.asarray(payload["features"], dtype=np.float32)
            )
        scaler = StableCausalFeatureScaler.fit(
            features,
            calibration_steps=config["input"]["calibration_steps"],
            mode=config["input"]["scaling_mode"],
        )
        sequences[bearing] = scaler.transform(features).numpy()

    staging.mkdir(parents=True)
    started = time.time()
    training_rows: list[dict[str, object]] = []
    health_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    encoder_config = config["encoder"]
    chunk_config = config["training_chunks"]
    training_config = config["training"]
    inference_config = config["causal_inference"]
    for seed in training_config["seeds"]:
        for fold in config["outer_lobo_folds"]:
            seed_everything(seed)
            rng = np.random.default_rng(seed)
            test_bearing = fold["test"]
            train_bearings = fold["train"]
            chunks = torch.cat(
                [
                    evenly_spaced_chunks(
                        sequences[bearing],
                        chunk_length=chunk_config["chunk_length"],
                        chunk_count=chunk_config["chunks_per_training_bearing"],
                    )
                    for bearing in train_bearings
                ]
            )
            encoder = TS2VecEncoder(
                input_dims=encoder_config["input_dims"],
                output_dims=encoder_config["output_dims"],
                hidden_dims=encoder_config["hidden_dims"],
                depth=encoder_config["depth"],
                mask_probability=encoder_config["mask_probability"],
            )
            averaged = torch.optim.swa_utils.AveragedModel(encoder)
            averaged.update_parameters(encoder)
            optimizer = torch.optim.AdamW(
                encoder.parameters(),
                lr=training_config["learning_rate"],
                weight_decay=training_config["weight_decay"],
            )
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(chunks.shape[0], generator=generator)
            cursor = 0
            for iteration in range(1, training_config["fixed_iterations"] + 1):
                if cursor + chunk_config["batch_size"] > chunks.shape[0]:
                    order = torch.randperm(chunks.shape[0], generator=generator)
                    cursor = 0
                indices = order[cursor : cursor + chunk_config["batch_size"]]
                cursor += chunk_config["batch_size"]
                batch = chunks.index_select(0, indices)
                views = sample_context_views(
                    batch,
                    temporal_unit=encoder_config["temporal_unit"],
                    random_state=rng,
                )
                optimizer.zero_grad(set_to_none=True)
                first = encoder(views.first)[:, -views.overlap_length :]
                second = encoder(views.second)[:, : views.overlap_length]
                loss = hierarchical_contrastive_loss(
                    first,
                    second,
                    alpha=encoder_config["contrastive_alpha"],
                    temporal_unit=encoder_config["temporal_unit"],
                )
                loss.backward()
                optimizer.step()
                averaged.update_parameters(encoder)
                training_rows.append(
                    {
                        "seed": seed,
                        "test_bearing": test_bearing,
                        "iteration": iteration,
                        "contrastive_loss": float(loss.detach()),
                    }
                )
                if iteration in (1, 50, 100, 150, 200):
                    print(
                        f"seed={seed} test={test_bearing} "
                        f"iteration={iteration}/200 "
                        f"loss={float(loss.detach()):.6f}",
                        flush=True,
                    )

            checkpoint = staging / f"seed_{seed}_{test_bearing}_final.pt"
            torch.save(
                {
                    "state_dict": averaged.state_dict(),
                    "seed": seed,
                    "test_bearing": test_bearing,
                    "train_bearings": train_bearings,
                    "iteration": training_config["fixed_iterations"],
                },
                checkpoint,
            )
            representations = {
                bearing: encode_causal(
                    averaged,
                    sequences[bearing],
                    context_length=inference_config["context_length"],
                    batch_size=64,
                )
                for bearing in bearings
            }

            readout = config["health_readout"]
            train_x = []
            train_y = []
            for bearing in train_bearings:
                sample = evenly_spaced_indices(
                    representations[bearing].shape[0],
                    readout["equal_samples_per_training_bearing"],
                )
                train_x.append(representations[bearing][sample])
                train_y.append(sample / float(representations[bearing].shape[0] - 1))
            test_lifetime = np.linspace(
                0.0, 1.0, representations[test_bearing].shape[0]
            )
            predicted_lifetime = ridge_fit_predict(
                np.concatenate(train_x),
                np.concatenate(train_y),
                representations[test_bearing],
                alpha=readout["alpha"],
            )
            progress = np.clip(predicted_lifetime, 0.0, 1.0)
            rho = float(spearmanr(test_lifetime, progress).statistic)
            health_rows.append(
                {
                    "seed": seed,
                    "test_bearing": test_bearing,
                    "train_bearings": "+".join(train_bearings),
                    "spearman_progress": rho,
                    "backward_step_fraction": backward_step_fraction(progress),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
            for step, (life, raw_prediction, state) in enumerate(
                zip(test_lifetime, predicted_lifetime, progress)
            ):
                trajectory_rows.append(
                    {
                        "seed": seed,
                        "bearing_id": test_bearing,
                        "step_id": step,
                        "normalized_lifetime": float(life),
                        "raw_predicted_lifetime": float(raw_prediction),
                        "health_progress": float(state),
                    }
                )

            probe = config["identity_probe"]
            identity_accuracy, blocks = identity_probe(
                representations,
                bearings,
                sample_count=probe["equal_samples_per_bearing"],
                folds=probe["contiguous_lifetime_folds"],
            )
            for row in blocks:
                identity_rows.append(
                    {
                        "seed": seed,
                        "outer_test_bearing": test_bearing,
                        **row,
                    }
                )
            identity_rows.append(
                {
                    "seed": seed,
                    "outer_test_bearing": test_bearing,
                    "held_lifetime_block": "all",
                    "accuracy": identity_accuracy,
                    "test_samples": probe["equal_samples_per_bearing"] * len(bearings),
                }
            )

    summary_rows = []
    for bearing in bearings:
        rows = [row for row in health_rows if row["test_bearing"] == bearing]
        values = np.asarray([row["spearman_progress"] for row in rows], dtype=float)
        summary_rows.append(
            {
                "bearing_id": bearing,
                "mean_spearman": float(values.mean()),
                "std_spearman": float(values.std(ddof=0)),
                "positive_seed_count": int(np.count_nonzero(values > 0.0)),
            }
        )
    identity_overall = [
        float(row["accuracy"])
        for row in identity_rows
        if row["held_lifetime_block"] == "all"
    ]
    gate_config = config["evaluation_gate"]
    mean_rhos = [float(row["mean_spearman"]) for row in summary_rows]
    health_gate = {
        "all_three_mean_spearman_positive": all(value > 0.0 for value in mean_rhos),
        "at_least_two_mean_spearman_at_least_0_5": sum(
            value >= gate_config["health"]["at_least_two_mean_spearman_at_least"]
            for value in mean_rhos
        )
        >= 2,
        "minimum_positive_seeds_per_bearing": all(
            int(row["positive_seed_count"])
            >= gate_config["health"]["minimum_positive_seeds_per_bearing"]
            for row in summary_rows
        ),
    }
    health_gate["passed"] = all(health_gate.values())
    mean_identity = float(np.mean(identity_overall))
    identity_gate = {
        "mean_accuracy": mean_identity,
        "maximum_allowed": gate_config["identity"]["mean_accuracy_at_most"],
        "passed": mean_identity <= gate_config["identity"]["mean_accuracy_at_most"],
    }
    combined_passed = bool(health_gate["passed"] and identity_gate["passed"])
    write_csv(staging / "training_history.csv", training_rows)
    write_csv(staging / "per_seed_health_metrics.csv", health_rows)
    write_csv(staging / "health_summary_by_bearing.csv", summary_rows)
    write_csv(staging / "identity_probe.csv", identity_rows)
    write_csv(staging / "health_trajectories.csv", trajectory_rows)
    report = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "config_sha256": sha256(config_path),
        "health_summary": summary_rows,
        "health_gate": health_gate,
        "identity_gate": identity_gate,
        "combined_gate_passed": combined_passed,
        "validation_authorized": combined_passed,
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
        "outer_test_used_for_checkpoint_selection": False,
        "future_context_used": False,
    }
    (staging / "experiment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    staging.rename(final_output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
