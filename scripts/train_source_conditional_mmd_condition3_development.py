"""Paired source-only/global-MMD/conditional-MMD Condition 3 LOBO experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.conditional_mmd_preflight import (  # noqa: E402
    median_squared_distance, paired_epoch_indices, paired_mmd2,
)
from state_interpreter.encoders.domain_adversarial import DomainAdversarialHealthEncoder  # noqa: E402
from state_interpreter.representation_diagnostics import evenly_spaced_indices  # noqa: E402
from state_interpreter.feature_lstm_experiment import backward_step_fraction  # noqa: E402
from preflight_source_conditional_mmd_numerics import source_sequences  # noqa: E402
from train_source_dann_condition3_development import (  # noqa: E402
    encode_sequence, identity_probe, right_padded_causal_windows, seed_everything,
)

ARMS = ("source_only", "global_mmd", "conditional_mmd")
TOKEN = "TRAIN_SOURCE_CONDITIONAL_MMD_CONDITION3_DEVELOPMENT_ONCE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict) -> dict:
    if config["status"] != "preregistered_no_development_run":
        raise ValueError("invalid conditional MMD preregistration status")
    if config["development_bearings"] != ["Bearing3_1", "Bearing3_2", "Bearing3_3"]:
        raise ValueError("development split changed")
    if config["protected_validation"] != ["Bearing3_4"] or config["protected_blind"] != ["Bearing3_5"]:
        raise ValueError("protected split changed")
    if len(config["outer_lobo_folds"]) != 3 or any(
        set(fold["train"] + [fold["test"]]) != set(config["development_bearings"])
        or len(fold["train"]) != 2 or fold["test"] in fold["train"]
        for fold in config["outer_lobo_folds"]
    ):
        raise ValueError("LOBO folds changed")
    if tuple(config["arms"]) != ARMS or config["regularization"]["weight"] != 0.1:
        raise ValueError("arms or locked MMD weight changed")
    if config["regularization"]["bandwidth_multipliers"] != [0.5, 1.0, 2.0]:
        raise ValueError("MMD kernel scales changed")
    sampler = config["paired_sampler"]
    if (sampler["bins"], sampler["sources_per_fold"], sampler["samples_per_source_bin_batch"],
        sampler["batches_per_epoch"], sampler["batch_size"]) != (5, 2, 6, 10, 60):
        raise ValueError("paired sampler changed")
    if not sampler["same_batch_indices_across_arms"] or not sampler["within_bin_cyclic_sampling"]:
        raise ValueError("paired sampler control changed")
    if config["training"]["seeds"] != [20260918, 20260921, 20260924]:
        raise ValueError("training seeds changed")
    if config["training"]["fixed_epochs"] != 30 or config["training"]["batch_size"] != 60:
        raise ValueError("training budget changed")
    if (
        config["training"]["optimizer"], config["training"]["learning_rate"],
        config["training"]["weight_decay"], config["training"]["checkpoint_selection"],
    ) != ("AdamW", 0.001, 0.0001, "fixed_final_epoch_no_outer_test_selection"):
        raise ValueError("training optimizer or checkpoint rule changed")
    if config["input"]["cache_directory"] != "cache/xjtu_condition3_feature_lstm_v1":
        raise ValueError("development cache path changed")
    if [entry["bearing_id"] for entry in config["input"]["cache_entries"]] != config["development_bearings"]:
        raise ValueError("cache entries changed")
    if config["input"]["context_length"] != 128 or config["input"]["feature_count"] != 65:
        raise ValueError("input contract changed")
    if config["input"]["calibration_steps"] != 15 or config["input"]["scaling_mode"] != "signed_log1p":
        raise ValueError("scaling contract changed")
    if config["model"] != {"recurrent": "unidirectional_GRU", "hidden_size": 32,
                            "embedding_size": 16, "health_head": "linear_normalized_lifetime"}:
        raise ValueError("model contract changed")
    if config["causal_inference"]["future_context_allowed"] is not False:
        raise ValueError("future context is forbidden")
    if config["evaluation_gate"]["identity_maximum_accuracy"] != 0.8:
        raise ValueError("identity gate changed")
    if config["evaluation_gate"]["combined_rule"] != "candidate_health_and_identity_gates_must_both_pass":
        raise ValueError("combined gate changed")
    if config["execution_token"] != TOKEN:
        raise ValueError("execution token changed")
    if config["output_directory"] != "results/2026-09-19_source_conditional_mmd_development_v1":
        raise ValueError("output directory changed")
    output = (ROOT / config["output_directory"]).resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("output escapes results directory")
    verified = {}
    expected_artifacts = {
        "src/state_interpreter/conditional_mmd_preflight.py",
        "scripts/preflight_source_conditional_mmd_numerics.py",
        "scripts/train_source_conditional_mmd_condition3_development.py",
        "src/state_interpreter/encoders/domain_adversarial.py",
        "src/state_interpreter/data/stable_feature_scaling.py",
        "scripts/train_source_dann_condition3_development.py",
        "results/2026-09-19_source_conditional_mmd_numerical_preflight_v1/diagnostics.json",
    }
    if {item["path"] for item in config["frozen_artifacts"]} != expected_artifacts:
        raise ValueError("frozen artifact list changed")
    for artifact in config["frozen_artifacts"]:
        path = (ROOT / artifact["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen artifact mismatch: {artifact['path']}")
        verified[artifact["path"]] = artifact["sha256"]
    return verified


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_windows(sequences: dict[str, np.ndarray], train_bearings: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[int, int]]]:
    windows, lengths, positions = [], [], []
    for bearing in train_bearings:
        endpoints = evenly_spaced_indices(sequences[bearing].shape[0], 300)
        batch, valid = right_padded_causal_windows(sequences[bearing], endpoints, context_length=128)
        windows.append(batch)
        lengths.append(valid)
        positions.append({int(endpoint): index for index, endpoint in enumerate(endpoints)})
    return windows, lengths, positions


def batch_tensors(
    indices: np.ndarray, sources: np.ndarray, bins: np.ndarray,
    sequences: dict[str, np.ndarray], train_bearings: list[str],
    windows: list[torch.Tensor], lengths: list[torch.Tensor], positions: list[dict[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    parts, valid_parts, targets = [], [], []
    for source, bearing in enumerate(train_bearings):
        selected = indices[sources == source]
        offsets = torch.as_tensor([positions[source][int(endpoint)] for endpoint in selected], dtype=torch.long)
        parts.append(windows[source].index_select(0, offsets))
        valid_parts.append(lengths[source].index_select(0, offsets))
        targets.extend((selected / (sequences[bearing].shape[0] - 1)).tolist())
    if any(int(np.count_nonzero((sources == source) & (bins == time_bin))) != 6
           for source in range(2) for time_bin in range(5)):
        raise ValueError("paired batch lost a source/time-bin cell")
    return (torch.cat(parts), torch.cat(valid_parts),
            torch.tensor(targets, dtype=torch.float32),
            torch.as_tensor(sources, dtype=torch.long),
            torch.as_tensor(bins, dtype=torch.long))


def model_new(seed: int) -> DomainAdversarialHealthEncoder:
    seed_everything(seed)
    return DomainAdversarialHealthEncoder(
        n_features=65, hidden_size=32, embedding_size=16,
        domain_hidden_size=32, domain_count=2,
    )


def fixed_kernel_scale(
    seed: int, sequences: dict[str, np.ndarray], train_bearings: list[str],
    windows: list[torch.Tensor], lengths: list[torch.Tensor], positions: list[dict[int, int]],
    indices: np.ndarray, sources: np.ndarray, bins: np.ndarray,
) -> float:
    model = model_new(seed)
    batch, valid, _, _, _ = batch_tensors(indices, sources, bins, sequences, train_bearings, windows, lengths, positions)
    with torch.no_grad():
        output = model(batch, lengths=valid, grl_coefficient=0.0)
        return float(median_squared_distance(output.embedding))


def run(config: dict, config_path: Path, verified: dict) -> dict:
    destination = (ROOT / config["output_directory"]).resolve()
    staging = destination.with_name(destination.name + ".staging")
    if destination.exists() or staging.exists():
        raise FileExistsError("refusing to overwrite development output or staging")
    cache = (ROOT / config["input"]["cache_directory"]).resolve()
    for entry in config["input"]["cache_entries"]:
        if sha256(cache / f"{entry['bearing_id']}.npz") != entry["sha256"]:
            raise ValueError(f"development cache hash mismatch: {entry['bearing_id']}")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    started = time.time()
    staging.mkdir(parents=True)
    histories, health_rows, identity_rows, trajectories, scale_rows = [], [], [], [], []
    bearings = config["development_bearings"]
    for seed in config["training"]["seeds"]:
        for fold in config["outer_lobo_folds"]:
            train_bearings = fold["train"]
            sequences = source_sequences(config, train_bearings)
            windows, lengths, positions = train_windows(sequences, train_bearings)
            first_indices, first_sources, first_bins = paired_epoch_indices(
                tuple(sequences[b].shape[0] for b in train_bearings), seed=seed,
            )
            squared_scale = fixed_kernel_scale(
                seed, sequences, train_bearings, windows, lengths, positions,
                first_indices[0], first_sources[0], first_bins[0],
            )
            scale_rows.append({"seed": seed, "outer_test_bearing": fold["test"],
                               "train_bearings": "+".join(train_bearings),
                               "initial_squared_scale": squared_scale})
            for arm in ARMS:
                model = model_new(seed)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=config["training"]["learning_rate"],
                    weight_decay=config["training"]["weight_decay"],
                )
                for epoch in range(1, config["training"]["fixed_epochs"] + 1):
                    endpoints, source_ids, bin_ids = paired_epoch_indices(
                        tuple(sequences[b].shape[0] for b in train_bearings),
                        seed=seed + (epoch - 1) * 1000,
                    )
                    health_sum = mmd_sum = 0.0
                    for step in range(10):
                        batch, valid, target, sources, bins = batch_tensors(
                            endpoints[step], source_ids[step], bin_ids[step],
                            sequences, train_bearings, windows, lengths, positions,
                        )
                        output = model(batch, lengths=valid, grl_coefficient=0.0)
                        health_loss = F.mse_loss(output.health.squeeze(-1), target)
                        if arm == "source_only":
                            mmd_loss = health_loss.new_zeros(())
                        else:
                            losses = [paired_mmd2(
                                output.embedding, sources, bins,
                                health_loss.new_tensor(squared_scale * multiplier),
                            ) for multiplier in config["regularization"]["bandwidth_multipliers"]]
                            channel = 0 if arm == "global_mmd" else 1
                            mmd_loss = torch.stack([pair[channel] for pair in losses]).mean()
                        loss = health_loss + config["regularization"]["weight"] * mmd_loss
                        if not torch.isfinite(loss):
                            raise ValueError("non-finite training loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        health_sum += float(health_loss.detach())
                        mmd_sum += float(mmd_loss.detach())
                    histories.append({"arm": arm, "seed": seed, "outer_test_bearing": fold["test"],
                                      "epoch": epoch, "health_mse": health_sum / 10,
                                      "mmd2": mmd_sum / 10, "kernel_squared_scale": squared_scale})
                    if epoch in (1, 10, 20, 30):
                        print(f"arm={arm} seed={seed} test={fold['test']} epoch={epoch}/30 "
                              f"health={health_sum/10:.6f} mmd={mmd_sum/10:.6f}", flush=True)
                checkpoint = staging / f"{arm}_seed_{seed}_{fold['test']}_final.pt"
                torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed,
                            "train_bearings": train_bearings, "outer_test_bearing": fold["test"],
                            "epoch": 30}, checkpoint)
                evaluation_sequences = {**sequences, **source_sequences(config, [fold["test"]])}
                encoded = {bearing: encode_sequence(
                    model, evaluation_sequences[bearing], context_length=128, batch_size=64,
                ) for bearing in bearings}
                prediction, _ = encoded[fold["test"]]
                lifetime = np.linspace(0.0, 1.0, prediction.size)
                rho = float(spearmanr(lifetime, prediction).statistic)
                health_rows.append({"arm": arm, "seed": seed, "outer_test_bearing": fold["test"],
                                    "spearman_health": rho,
                                    "backward_step_fraction": backward_step_fraction(prediction),
                                    "checkpoint_sha256": sha256(checkpoint)})
                trajectories.extend({"arm": arm, "seed": seed, "bearing_id": fold["test"],
                                     "step_id": step, "normalized_lifetime": float(life),
                                     "predicted_health": float(value)}
                                    for step, (life, value) in enumerate(zip(lifetime, prediction)))
                accuracy, blocks = identity_probe(
                    {bearing: encoded[bearing][1] for bearing in bearings},
                    bearings, sample_count=300, folds=5,
                )
                identity_rows.extend({"arm": arm, "seed": seed, "outer_test_bearing": fold["test"],
                                      **row} for row in blocks)
                identity_rows.append({"arm": arm, "seed": seed, "outer_test_bearing": fold["test"],
                                      "held_lifetime_block": "all", "accuracy": accuracy,
                                      "test_samples": 900})
    summaries = []
    for arm in ARMS:
        for bearing in bearings:
            values = np.asarray([row["spearman_health"] for row in health_rows
                                 if row["arm"] == arm and row["outer_test_bearing"] == bearing])
            summaries.append({"arm": arm, "bearing_id": bearing,
                              "mean_spearman": float(values.mean()),
                              "std_spearman": float(values.std(ddof=0)),
                              "positive_seed_count": int(np.count_nonzero(values > 0))})
    identity_summary = {arm: float(np.mean([row["accuracy"] for row in identity_rows
                                           if row["arm"] == arm and row["held_lifetime_block"] == "all"]))
                        for arm in ARMS}
    candidate = [row for row in summaries if row["arm"] == "conditional_mmd"]
    health_gate = {"all_means_positive": all(row["mean_spearman"] > 0 for row in candidate),
                   "two_means_at_least_0_5": sum(row["mean_spearman"] >= 0.5 for row in candidate) >= 2,
                   "two_positive_seeds_each": all(row["positive_seed_count"] >= 2 for row in candidate)}
    health_gate["passed"] = all(health_gate.values())
    identity_gate = {"mean_accuracy": identity_summary["conditional_mmd"],
                     "maximum_allowed": 0.8,
                     "passed": identity_summary["conditional_mmd"] <= 0.8}
    for name, rows in (("training_history.csv", histories), ("per_seed_health_metrics.csv", health_rows),
                       ("health_summary_by_arm_and_bearing.csv", summaries), ("identity_probe.csv", identity_rows),
                       ("health_trajectories.csv", trajectories), ("kernel_scales.csv", scale_rows)):
        csv_write(staging / name, rows)
    report = {"experiment_id": config["experiment_id"], "status": "completed",
              "elapsed_seconds": time.time() - started, "config_sha256": sha256(config_path),
              "verified_frozen_artifacts": verified, "health_summary": summaries,
              "identity_mean_accuracy": identity_summary, "health_gate": health_gate,
              "identity_gate": identity_gate,
              "combined_gate_passed": bool(health_gate["passed"] and identity_gate["passed"]),
              "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
              "outer_test_used_for_checkpoint_selection": False,
              "future_context_used": False}
    (staging / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    staging.rename(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--confirm")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verified = validate_config(config)
    if args.check_only:
        print(json.dumps({"status": "preflight_passed_no_cache_read", "verified": verified}))
        return
    if args.confirm != TOKEN:
        raise PermissionError("exact development training token required")
    print(json.dumps(run(config, config_path, verified), indent=2))


if __name__ == "__main__":
    main()
