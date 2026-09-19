"""Read-only Condition 3 source-source MMD numerical preflight (no training)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.conditional_mmd_preflight import (  # noqa: E402
    median_squared_distance, paired_epoch_indices, paired_mmd2,
)
from state_interpreter.data.stable_feature_scaling import StableCausalFeatureScaler  # noqa: E402
from state_interpreter.encoders.domain_adversarial import DomainAdversarialHealthEncoder  # noqa: E402
from train_source_dann_condition3_development import right_padded_causal_windows  # noqa: E402

CONFIRMATION = "PREFLIGHT_SOURCE_CONDITIONAL_MMD_DEVELOPMENT_ONLY"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scope(config: dict, confirmation: str) -> None:
    if confirmation != CONFIRMATION:
        raise PermissionError("exact development-only preflight token required")
    if config["status"] != "preregistered_preflight_no_cache_read":
        raise ValueError("source contract status changed")
    if config["development_bearings"] != ["Bearing3_1", "Bearing3_2", "Bearing3_3"]:
        raise ValueError("development bearing scope changed")
    if config["protected_validation"] != ["Bearing3_4"] or config["protected_blind"] != ["Bearing3_5"]:
        raise ValueError("protected bearing scope changed")
    if sorted(item["bearing_id"] for item in config["input"]["cache_entries"]) != config["development_bearings"]:
        raise ValueError("cache scope changed")
    if len(config["outer_lobo_folds"]) != 3 or any(
        len(fold["train"]) != 2 or fold["test"] in fold["train"]
        or set(fold["train"] + [fold["test"]]) != set(config["development_bearings"])
        for fold in config["outer_lobo_folds"]
    ):
        raise ValueError("LOBO folds changed")
    if config["input"]["equal_training_endpoints_per_bearing"] != 300:
        raise ValueError("endpoint count changed")
    if config["input"]["cache_directory"] != "cache/xjtu_condition3_feature_lstm_v1":
        raise ValueError("development cache path changed")
    if config["input"]["context_length"] != 128 or config["input"]["feature_count"] != 65:
        raise ValueError("input contract changed")
    if config["training"]["seeds"] != [20260918, 20260921, 20260924]:
        raise ValueError("seed contract changed")


def source_sequences(config: dict, train_bearings: list[str]) -> dict[str, np.ndarray]:
    cache = (ROOT / config["input"]["cache_directory"]).resolve()
    if not cache.is_relative_to(ROOT.resolve()):
        raise ValueError("cache directory escapes repository")
    expected = {entry["bearing_id"]: entry["sha256"] for entry in config["input"]["cache_entries"]}
    sequences = {}
    for bearing in train_bearings:
        path = (cache / f"{bearing}.npz").resolve()
        if not path.is_relative_to(cache) or sha256(path) != expected[bearing]:
            raise ValueError(f"development cache hash mismatch: {bearing}")
        with np.load(path, allow_pickle=False) as payload:
            features = torch.as_tensor(np.asarray(payload["features"], dtype=np.float32))
        scaler = StableCausalFeatureScaler.fit(
            features, calibration_steps=config["input"]["calibration_steps"],
            mode=config["input"]["scaling_mode"],
        )
        sequences[bearing] = scaler.transform(features).numpy()
    return sequences


def gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=False)
    result = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).detach().item()
    if not np.isfinite(result):
        raise ValueError("non-finite encoder gradient norm")
    return float(result)


def inspect_pair(config: dict, fold: dict, seed: int) -> dict:
    train_bearings = fold["train"]
    sequences = source_sequences(config, train_bearings)
    indices, source_ids, bin_ids = paired_epoch_indices(
        tuple(sequences[b].shape[0] for b in train_bearings), seed=seed,
    )
    if indices.shape != (10, 60):
        raise ValueError("paired sampler shape changed")
    windows_parts, lengths_parts, targets = [], [], []
    for source, bearing in enumerate(train_bearings):
        selected = indices[0, source_ids[0] == source]
        windows, lengths = right_padded_causal_windows(
            sequences[bearing], selected, context_length=128,
        )
        windows_parts.append(windows)
        lengths_parts.append(lengths)
        targets.extend((selected / (sequences[bearing].shape[0] - 1)).tolist())
    torch.manual_seed(seed)
    model = DomainAdversarialHealthEncoder(
        n_features=65, hidden_size=32, embedding_size=16,
        domain_hidden_size=32, domain_count=2,
    )
    output = model(torch.cat(windows_parts), lengths=torch.cat(lengths_parts), grl_coefficient=0.0)
    health_loss = F.mse_loss(output.health.squeeze(-1), torch.tensor(targets, dtype=torch.float32))
    source_tensor = torch.as_tensor(source_ids[0], dtype=torch.long)
    bin_tensor = torch.as_tensor(bin_ids[0], dtype=torch.long)
    median_distance = median_squared_distance(output.embedding)
    global_terms, conditional_terms = [], []
    for multiplier in (0.5, 1.0, 2.0):
        global_loss, conditional_loss = paired_mmd2(
            output.embedding, source_tensor, bin_tensor, median_distance * multiplier,
        )
        global_terms.append(global_loss)
        conditional_terms.append(conditional_loss)
    global_loss = torch.stack(global_terms).mean()
    conditional_loss = torch.stack(conditional_terms).mean()
    shared_parameters = list(model.gru.parameters()) + list(model.projector.parameters())
    norms = {
        "health": gradient_norm(health_loss, shared_parameters),
        "global_mmd": gradient_norm(global_loss, shared_parameters),
        "conditional_mmd": gradient_norm(conditional_loss, shared_parameters),
    }
    scalars = {
        "health_mse": float(health_loss.detach()),
        "median_squared_embedding_distance": float(median_distance),
        "global_mmd2": float(global_loss.detach()),
        "conditional_mmd2": float(conditional_loss.detach()),
    }
    if any(not np.isfinite(value) for value in scalars.values()):
        raise ValueError("non-finite preflight scalar")
    if min(scalars["global_mmd2"], scalars["conditional_mmd2"]) < -1e-6:
        raise ValueError("MMD2 is substantially negative")
    if min(norms.values()) <= 1e-12:
        raise ValueError("degenerate encoder gradient")
    return {
        "outer_test_not_read": fold["test"], "train_bearings": train_bearings,
        "seed": seed, "batches_per_epoch": 10, "batch_size": 60,
        "cell_count_per_batch": 6, "scalars": scalars, "encoder_gradient_norms": norms,
        "global_to_health_gradient_ratio": norms["global_mmd"] / norms["health"],
        "conditional_to_health_gradient_ratio": norms["conditional_mmd"] / norms["health"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_scope(config, args.confirm)
    output_path = Path(args.output).resolve()
    if output_path.exists() or not output_path.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("preflight output must be new and inside results")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    rows = [inspect_pair(config, fold, seed)
            for seed in config["training"]["seeds"]
            for fold in config["outer_lobo_folds"]]
    report = {
        "status": "numerical_preflight_completed_no_training",
        "experiment_id": "xjtu_source_conditional_mmd_numerical_preflight_v1",
        "source_config_sha256": sha256(config_path),
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
        "model_training_performed": False,
        "kernel": "biased_three_scale_rbf_mmd2_median_initial_embedding_distance",
        "fold_seed_rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(rows), "output": str(output_path)}))


if __name__ == "__main__":
    main()
