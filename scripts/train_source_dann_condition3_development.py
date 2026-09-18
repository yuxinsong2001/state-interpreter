"""Run the preregistered paired Source-only versus Source-DANN LOBO study."""

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
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.data.stable_feature_scaling import StableCausalFeatureScaler
from state_interpreter.encoders.domain_adversarial import (
    DomainAdversarialHealthEncoder,
    dann_coefficient,
)
from state_interpreter.feature_lstm_experiment import backward_step_fraction
from state_interpreter.representation_diagnostics import (
    evenly_spaced_indices,
    nearest_centroid_predict,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def right_padded_causal_windows(
    sequence: np.ndarray,
    endpoints: np.ndarray,
    *,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(sequence, dtype=torch.float32)
    indices = np.asarray(endpoints, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("sequence must have shape [timestamps, features]")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("endpoints must be a non-empty vector")
    if context_length < 2:
        raise ValueError("context_length must be at least two")
    if indices.min() < 0 or indices.max() >= values.shape[0]:
        raise ValueError("endpoint outside sequence")
    output = values.new_zeros((indices.size, context_length, values.shape[1]))
    lengths = torch.empty(indices.size, dtype=torch.long)
    for row, endpoint in enumerate(indices.tolist()):
        start = max(0, endpoint - context_length + 1)
        context = values[start : endpoint + 1]
        output[row, : context.shape[0]] = context
        lengths[row] = context.shape[0]
    return output, lengths


def validate_execution_contract(config: dict, confirmation: str) -> None:
    if config["status"] != "preregistered_preflight_no_cache_read":
        raise ValueError("invalid Source-DANN protocol status")
    if confirmation != config["execution_token"]:
        raise PermissionError("exact Source-DANN execution token required")
    if config["development_bearings"] != [
        "Bearing3_1",
        "Bearing3_2",
        "Bearing3_3",
    ]:
        raise ValueError("development split changed")
    if config["protected_validation"] != ["Bearing3_4"]:
        raise ValueError("validation protection changed")
    if config["protected_blind"] != ["Bearing3_5"]:
        raise ValueError("blind protection changed")
    if config["causal_inference"]["future_context_allowed"] is not False:
        raise ValueError("future context must remain prohibited")
    if config["paired_arms"]["source_only"]["maximum_grl_coefficient"] != 0.0:
        raise ValueError("source-only control changed")
    if config["paired_arms"]["source_dann"]["maximum_grl_coefficient"] != 0.1:
        raise ValueError("Source-DANN coefficient changed")
    for artifact in config["frozen_artifacts"]:
        if sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"frozen artifact mismatch: {artifact['path']}")


def encode_sequence(
    model: DomainAdversarialHealthEncoder,
    sequence: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    was_training = model.training
    model.eval()
    health_parts = []
    embedding_parts = []
    with torch.inference_mode():
        for start in range(0, sequence.shape[0], batch_size):
            endpoints = np.arange(start, min(start + batch_size, sequence.shape[0]))
            windows, lengths = right_padded_causal_windows(
                sequence,
                endpoints,
                context_length=context_length,
            )
            output = model(windows, lengths=lengths, grl_coefficient=0.0)
            health_parts.append(output.health.squeeze(-1).cpu())
            embedding_parts.append(output.embedding.cpu())
    model.train(was_training)
    return torch.cat(health_parts).numpy(), torch.cat(embedding_parts).numpy()


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
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_execution_contract(config, args.confirm)

    final_output = (ROOT / config["output_directory"]).resolve()
    staging = final_output.with_name(final_output.name + ".staging")
    if final_output.exists() or staging.exists():
        raise FileExistsError("refusing to overwrite Source-DANN output or staging data")

    bearings = config["development_bearings"]
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
            features = torch.from_numpy(np.asarray(payload["features"], dtype=np.float32))
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
    training = config["training"]
    model_config = config["model"]
    input_config = config["input"]
    arms = ["source_only", "source_dann"]

    for seed in training["seeds"]:
        for fold in config["outer_lobo_folds"]:
            test_bearing = fold["test"]
            train_bearings = fold["train"]
            training_windows = []
            training_lengths = []
            training_health = []
            training_domains = []
            for domain_label, bearing in enumerate(train_bearings):
                endpoints = evenly_spaced_indices(
                    sequences[bearing].shape[0],
                    input_config["equal_training_endpoints_per_bearing"],
                )
                windows, lengths = right_padded_causal_windows(
                    sequences[bearing],
                    endpoints,
                    context_length=input_config["context_length"],
                )
                training_windows.append(windows)
                training_lengths.append(lengths)
                training_health.append(
                    torch.as_tensor(
                        endpoints / float(sequences[bearing].shape[0] - 1),
                        dtype=torch.float32,
                    )
                )
                training_domains.append(
                    torch.full((endpoints.size,), domain_label, dtype=torch.long)
                )
            windows = torch.cat(training_windows)
            lengths = torch.cat(training_lengths)
            health_targets = torch.cat(training_health)
            domain_targets = torch.cat(training_domains)

            for arm in arms:
                seed_everything(seed)
                model = DomainAdversarialHealthEncoder(
                    n_features=input_config["feature_count"],
                    hidden_size=model_config["hidden_size"],
                    embedding_size=model_config["embedding_size"],
                    domain_hidden_size=model_config["domain_hidden_size"],
                    domain_count=model_config["domain_count_per_fold"],
                )
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=training["learning_rate"],
                    weight_decay=training["weight_decay"],
                )
                generator = torch.Generator().manual_seed(seed)
                maximum_grl = config["paired_arms"][arm]["maximum_grl_coefficient"]
                for epoch in range(1, training["fixed_epochs"] + 1):
                    order = torch.randperm(windows.shape[0], generator=generator)
                    epoch_health = 0.0
                    epoch_domain = 0.0
                    sample_total = 0
                    progress = (epoch - 1) / max(training["fixed_epochs"] - 1, 1)
                    coefficient = (
                        0.0
                        if maximum_grl == 0.0
                        else dann_coefficient(progress, maximum=maximum_grl)
                    )
                    for start in range(0, windows.shape[0], training["batch_size"]):
                        batch_indices = order[start : start + training["batch_size"]]
                        output = model(
                            windows.index_select(0, batch_indices),
                            lengths=lengths.index_select(0, batch_indices),
                            grl_coefficient=coefficient,
                        )
                        health_loss = F.mse_loss(
                            output.health.squeeze(-1),
                            health_targets.index_select(0, batch_indices),
                        )
                        domain_loss = F.cross_entropy(
                            output.domain_logits,
                            domain_targets.index_select(0, batch_indices),
                        )
                        loss = health_loss + domain_loss
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        count = batch_indices.numel()
                        epoch_health += float(health_loss.detach()) * count
                        epoch_domain += float(domain_loss.detach()) * count
                        sample_total += count
                    training_rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "test_bearing": test_bearing,
                            "epoch": epoch,
                            "grl_coefficient": coefficient,
                            "health_mse": epoch_health / sample_total,
                            "domain_cross_entropy": epoch_domain / sample_total,
                        }
                    )
                    if epoch in (1, 10, 20, 30):
                        print(
                            f"arm={arm} seed={seed} test={test_bearing} "
                            f"epoch={epoch}/30 health={epoch_health / sample_total:.6f} "
                            f"domain={epoch_domain / sample_total:.6f} grl={coefficient:.6f}",
                            flush=True,
                        )

                checkpoint = staging / f"{arm}_seed_{seed}_{test_bearing}_final.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "arm": arm,
                        "seed": seed,
                        "test_bearing": test_bearing,
                        "train_bearings": train_bearings,
                        "epoch": training["fixed_epochs"],
                    },
                    checkpoint,
                )
                encoded = {
                    bearing: encode_sequence(
                        model,
                        sequences[bearing],
                        context_length=input_config["context_length"],
                        batch_size=64,
                    )
                    for bearing in bearings
                }
                test_health, _ = encoded[test_bearing]
                lifetime = np.linspace(0.0, 1.0, test_health.size)
                rho = float(spearmanr(lifetime, test_health).statistic)
                health_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "test_bearing": test_bearing,
                        "train_bearings": "+".join(train_bearings),
                        "spearman_health": rho,
                        "backward_step_fraction": backward_step_fraction(test_health),
                        "checkpoint_sha256": sha256(checkpoint),
                    }
                )
                for step, (life, prediction) in enumerate(zip(lifetime, test_health)):
                    trajectory_rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "bearing_id": test_bearing,
                            "step_id": step,
                            "normalized_lifetime": float(life),
                            "predicted_health": float(prediction),
                        }
                    )
                representations = {
                    bearing: encoded[bearing][1] for bearing in bearings
                }
                accuracy, block_rows = identity_probe(
                    representations,
                    bearings,
                    sample_count=300,
                    folds=5,
                )
                for row in block_rows:
                    identity_rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "outer_test_bearing": test_bearing,
                            **row,
                        }
                    )
                identity_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "outer_test_bearing": test_bearing,
                        "held_lifetime_block": "all",
                        "accuracy": accuracy,
                        "test_samples": 900,
                    }
                )

    summary_rows = []
    for arm in arms:
        for bearing in bearings:
            values = np.asarray(
                [
                    row["spearman_health"]
                    for row in health_rows
                    if row["arm"] == arm and row["test_bearing"] == bearing
                ],
                dtype=float,
            )
            summary_rows.append(
                {
                    "arm": arm,
                    "bearing_id": bearing,
                    "mean_spearman": float(values.mean()),
                    "std_spearman": float(values.std(ddof=0)),
                    "positive_seed_count": int(np.count_nonzero(values > 0.0)),
                }
            )

    identity_summary = {}
    for arm in arms:
        values = [
            float(row["accuracy"])
            for row in identity_rows
            if row["arm"] == arm and row["held_lifetime_block"] == "all"
        ]
        identity_summary[arm] = float(np.mean(values))
    paired_rows = []
    for bearing in bearings:
        source = next(
            row
            for row in summary_rows
            if row["arm"] == "source_only" and row["bearing_id"] == bearing
        )
        dann = next(
            row
            for row in summary_rows
            if row["arm"] == "source_dann" and row["bearing_id"] == bearing
        )
        paired_rows.append(
            {
                "bearing_id": bearing,
                "source_only_mean_spearman": source["mean_spearman"],
                "source_dann_mean_spearman": dann["mean_spearman"],
                "dann_minus_source_only": (
                    dann["mean_spearman"] - source["mean_spearman"]
                ),
            }
        )

    dann_summary = [row for row in summary_rows if row["arm"] == "source_dann"]
    gate = config["evaluation_gate"]
    health_gate = {
        "all_three_mean_spearman_positive": all(
            row["mean_spearman"] > 0.0 for row in dann_summary
        ),
        "at_least_two_mean_spearman_at_least_0_5": sum(
            row["mean_spearman"]
            >= gate["health"]["at_least_two_source_dann_mean_spearman_at_least"]
            for row in dann_summary
        )
        >= 2,
        "minimum_positive_seeds_per_bearing": all(
            row["positive_seed_count"]
            >= gate["health"]["minimum_positive_seeds_per_bearing"]
            for row in dann_summary
        ),
    }
    health_gate["passed"] = all(health_gate.values())
    identity_gate = {
        "mean_accuracy": identity_summary["source_dann"],
        "maximum_allowed": gate["identity"]["source_dann_mean_accuracy_at_most"],
        "passed": (
            identity_summary["source_dann"]
            <= gate["identity"]["source_dann_mean_accuracy_at_most"]
        ),
    }
    combined_passed = bool(health_gate["passed"] and identity_gate["passed"])
    write_csv(staging / "training_history.csv", training_rows)
    write_csv(staging / "per_seed_health_metrics.csv", health_rows)
    write_csv(staging / "health_summary_by_arm_and_bearing.csv", summary_rows)
    write_csv(staging / "paired_arm_comparison.csv", paired_rows)
    write_csv(staging / "identity_probe.csv", identity_rows)
    write_csv(staging / "health_trajectories.csv", trajectory_rows)
    report = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "config_sha256": sha256(config_path),
        "health_summary": summary_rows,
        "paired_comparison": paired_rows,
        "identity_mean_accuracy": identity_summary,
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
