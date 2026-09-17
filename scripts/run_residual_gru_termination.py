"""Run the predeclared residual-GRU termination experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from state_interpreter import PredictiveGRUStateInterpreter
from state_interpreter.forecast_evaluation import aligned_squared_errors, forecast_metrics
from state_interpreter.residual_gru import ResidualGRUStateInterpreter
from run_gru_interpreter_exploratory import center_sequence, load_latents, prediction_mse, sha256, to_latent


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def state_dict_max_difference(left: dict, right: dict) -> tuple[float, int]:
    if left.keys() != right.keys():
        raise ValueError("state dict keys differ")
    differences = [float((left[key] - right[key]).abs().max()) for key in left]
    return max(differences, default=0.0), sum(value != 0 for value in differences)


def train_residual(seed: int, sequences: list[torch.Tensor], validation: torch.Tensor, cfg: dict, started: float):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ResidualGRUStateInterpreter(
        embedding_dim=sequences[0].shape[1], hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"]
    )
    initial_head_exact_zero = bool(
        torch.count_nonzero(model.prediction_head.weight) == 0
        and torch.count_nonzero(model.prediction_head.bias) == 0
    )
    initial_persistence_difference = 0.0
    with torch.no_grad():
        for sequence in [*sequences, validation]:
            initial_persistence_difference = max(
                initial_persistence_difference,
                float((model(sequence[None]).next_embedding_prediction[0] - sequence).abs().max()),
            )
    if not initial_head_exact_zero or initial_persistence_difference != 0:
        raise RuntimeError("residual model is not exact persistence at initialization")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best, best_epoch, best_state, no_improvement = math.inf, 0, None, 0
    history = []
    for epoch in range(1, cfg["max_epochs"] + 1):
        if time.monotonic() - started > cfg["hard_timeout_seconds"]:
            raise TimeoutError("predeclared hard timeout reached")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = prediction_mse(model, sequences)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"]))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_mse = float(prediction_mse(model, [validation]))
        if not all(math.isfinite(value) for value in (float(loss.detach()), validation_mse, gradient_norm)):
            raise ValueError("nonfinite training result")
        history.append({
            "seed": seed, "epoch": epoch, "train_prediction_mse": float(loss.detach()),
            "validation_prediction_mse": validation_mse, "gradient_norm": gradient_norm,
        })
        if validation_mse < best - cfg["minimum_improvement"]:
            best, best_epoch, best_state, no_improvement = validation_mse, epoch, deepcopy(model.state_dict()), 0
        else:
            no_improvement += 1
        if epoch % 50 == 0:
            print(f"residual seed={seed} epoch={epoch} val_mse={validation_mse:.6f}", flush=True)
        if no_improvement >= cfg["patience"]:
            break
    if best_state is None:
        raise RuntimeError("no residual checkpoint selected")
    model.load_state_dict(best_state)
    return model.eval(), history, best_epoch, best, initial_persistence_difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output = (ROOT / cfg["output_directory"]).resolve()
    if not output.is_relative_to(ROOT / "results") or output.exists():
        raise ValueError("output must be a new directory inside results")
    started = time.monotonic()

    input_path = ROOT / cfg["input_latents"]
    direct_dir = ROOT / cfg["direct_checkpoint_directory"]
    persistence_dir = ROOT / cfg["persistence_reference_directory"]
    direct_report_path = direct_dir / "experiment_report.json"
    persistence_report_path = persistence_dir / "experiment_report.json"
    direct_report = json.loads(direct_report_path.read_text(encoding="utf-8"))
    persistence_report = json.loads(persistence_report_path.read_text(encoding="utf-8"))
    if not str(direct_report["status"]).startswith("completed") or not str(persistence_report["status"]).startswith("completed"):
        raise ValueError("reference experiment is incomplete")
    if direct_report["config"]["train_bearings"] != cfg["train_bearings"]:
        raise ValueError("training bearings differ from direct experiment")
    for field in ("calibration_steps", "hidden_dim", "num_layers", "learning_rate", "weight_decay", "max_epochs", "patience", "minimum_improvement", "gradient_clip_norm"):
        if direct_report["config"][field] != cfg[field]:
            raise ValueError(f"configuration differs for {field}")
    if persistence_report["config"]["primary_target_start_step"] != cfg["primary_target_start_step"]:
        raise ValueError("primary target range differs")

    direct_paths = {seed: direct_dir / f"control_seed_{seed}_checkpoint.pt" for seed in cfg["random_seeds"]}
    for path in direct_paths.values():
        if sha256(path) != direct_report["checkpoint_sha256"][path.name]:
            raise ValueError(f"direct checkpoint changed: {path.name}")
    protected = [
        config_path, input_path, direct_report_path, direct_dir / "per_seed_bearing_metrics.csv",
        persistence_report_path, persistence_dir / "per_seed_bearing_metrics.csv", *direct_paths.values(),
    ]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}

    by_bearing, columns = load_latents(input_path)
    expected = cfg["train_bearings"] + [cfg["validation_bearing"], cfg["old_holdout_bearing"]]
    if sorted(by_bearing) != sorted(expected) or columns != [f"z_{index}" for index in range(8)]:
        raise ValueError("unexpected input inventory")
    raw = {bearing: to_latent(rows, columns) for bearing, rows in by_bearing.items()}
    for bearing, rows in by_bearing.items():
        if [int(row["step_id"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"noncontiguous steps: {bearing}")
    centered = {bearing: center_sequence(sequence, cfg["calibration_steps"]) for bearing, sequence in raw.items()}
    fit = torch.cat([centered[bearing] for bearing in cfg["train_bearings"]])
    mean, std = fit.mean(0), fit.std(0, unbiased=False).clamp_min(1e-6)
    standardized = {bearing: (sequence - mean) / std for bearing, sequence in centered.items()}
    train_sequences = [standardized[bearing] for bearing in cfg["train_bearings"]]
    validation = standardized[cfg["validation_bearing"]]

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    write_json(output / "execution_record.json", {
        "status": "running", "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_sha256": sha256(config_path),
    })

    histories, training_rows, residual_models, residual_paths = [], [], {}, []
    for seed in cfg["random_seeds"]:
        # Same seed and constructor order establish identical initial GRU weights.
        torch.manual_seed(seed)
        direct_initial = PredictiveGRUStateInterpreter(embedding_dim=len(columns), hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])
        torch.manual_seed(seed)
        residual_initial = ResidualGRUStateInterpreter(embedding_dim=len(columns), hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])
        initial_gru_difference, initial_gru_mismatches = state_dict_max_difference(direct_initial.gru.state_dict(), residual_initial.gru.state_dict())
        if initial_gru_difference != 0 or initial_gru_mismatches != 0:
            raise RuntimeError("paired initial GRU parameters differ")

        model, history, best_epoch, best_mse, initial_difference = train_residual(
            seed, train_sequences, validation, cfg, started
        )
        checkpoint_path = output / f"residual_seed_{seed}_checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(), "embedding_dim": len(columns),
            "hidden_dim": cfg["hidden_dim"], "num_layers": cfg["num_layers"],
            "feature_mean": mean, "feature_std": std, "best_epoch": best_epoch,
            "best_validation_next_step_mse": best_mse, "random_seed": seed,
            "arm": "residual", "config_sha256": sha256(config_path),
        }, checkpoint_path)
        residual_models[seed] = model
        residual_paths.append(checkpoint_path)
        histories.extend(history)
        training_rows.append({
            "seed": seed, "epochs_completed": len(history), "best_epoch": best_epoch,
            "best_validation_mse": best_mse, "initial_head_exact_zero": True,
            "initial_persistence_max_absolute_difference": initial_difference,
            "paired_initial_gru_max_absolute_difference": initial_gru_difference,
        })
        write_csv(output / "training_history.csv", histories)
        print(f"LOCKED residual seed={seed} checkpoint_epoch={best_epoch}", flush=True)

    reference_primary = {
        (row["arm"], int(row["seed"]), row["bearing_id"]): row
        for row in read_csv(persistence_dir / "per_seed_bearing_metrics.csv") if row["scope"] == "primary"
    }
    metric_rows, target_rows, integrity_rows = [], [], []
    for seed in cfg["random_seeds"]:
        direct_checkpoint = torch.load(direct_paths[seed], weights_only=True, map_location="cpu")
        if not torch.equal(mean, direct_checkpoint["feature_mean"]) or not torch.equal(std, direct_checkpoint["feature_std"]):
            raise ValueError("normalization differs from direct checkpoint")
        direct_model = PredictiveGRUStateInterpreter(embedding_dim=len(columns), hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])
        direct_model.load_state_dict(direct_checkpoint["model_state_dict"])
        direct_model.eval()
        residual_model = residual_models[seed]
        for bearing in expected:
            sequence = standardized[bearing]
            for arm, model in (("direct", direct_model), ("residual", residual_model)):
                with torch.no_grad():
                    forecast = model(sequence[None]).next_embedding_prediction[0]
                    prefix_length = min(cfg["prefix_check_length"], len(sequence) - 1)
                    prefix = model(sequence[:prefix_length][None]).next_embedding_prediction[0]
                prefix_difference = float((prefix - forecast[:prefix_length]).abs().max())
                if prefix_difference > cfg["prefix_tolerance"]:
                    raise ValueError("causal prefix check failed")
                model_errors, persistence_errors = aligned_squared_errors(sequence, forecast)
                metrics = forecast_metrics(model_errors, persistence_errors, target_start_step=cfg["primary_target_start_step"])
                metric_rows.append({
                    "arm": arm, "seed": seed, "bearing_id": bearing,
                    "split": by_bearing[bearing][0]["split"], **metrics,
                })
                direct_reference_difference = None
                if arm == "direct":
                    archived = reference_primary[("control", seed, bearing)]
                    direct_reference_difference = abs(metrics["gru_mse"] - float(archived["gru_mse"]))
                    if direct_reference_difference != 0:
                        raise ValueError("direct reference MSE did not reproduce")
                integrity_rows.append({
                    "arm": arm, "seed": seed, "bearing_id": bearing,
                    "prefix_max_absolute_difference": prefix_difference,
                    "direct_reference_mse_absolute_difference": direct_reference_difference,
                })
                selected_model = model_errors[cfg["primary_target_start_step"] - 1 :].double().mean(1)
                selected_persistence = persistence_errors[cfg["primary_target_start_step"] - 1 :].double().mean(1)
                for offset in range(len(selected_model)):
                    target_rows.append({
                        "arm": arm, "seed": seed, "bearing_id": bearing,
                        "target_step": cfg["primary_target_start_step"] + offset,
                        "model_step_mse": float(selected_model[offset]),
                        "persistence_step_mse": float(selected_persistence[offset]),
                        "model_minus_persistence": float(selected_model[offset] - selected_persistence[offset]),
                    })

    summaries = []
    for arm in ("direct", "residual"):
        for bearing in expected:
            selected = [row for row in metric_rows if row["arm"] == arm and row["bearing_id"] == bearing]
            skills = [row["skill"] for row in selected]
            summaries.append({
                "arm": arm, "bearing_id": bearing, "split": selected[0]["split"],
                "seed_count": len(selected), "target_count": selected[0]["target_count"],
                "mse_ratio_mean": float(np.mean([row["mse_ratio"] for row in selected])),
                "mse_ratio_sample_std": float(np.std([row["mse_ratio"] for row in selected], ddof=1)),
                "skill_mean": float(np.mean(skills)),
                "skill_sample_std": float(np.std(skills, ddof=1)),
                "skill_min": min(skills), "skill_max": max(skills),
                "positive_skill_seed_count": sum(value > 0 for value in skills),
                "win_fraction_mean": float(np.mean([row["gru_win_fraction"] for row in selected])),
            })
    decisions = []
    required = cfg["decision"]["required_positive_skill_seed_count_each_bearing"]
    for bearing in cfg["decision"]["decision_bearings"]:
        residual = next(row for row in summaries if row["arm"] == "residual" and row["bearing_id"] == bearing)
        direct = next(row for row in summaries if row["arm"] == "direct" and row["bearing_id"] == bearing)
        decisions.append({
            "bearing_id": bearing,
            "residual_positive_skill_seed_count": residual["positive_skill_seed_count"],
            "required_positive_skill_seed_count": required,
            "all_residual_seeds_positive": residual["positive_skill_seed_count"] == required,
            "residual_mean_ratio_below_direct": residual["mse_ratio_mean"] < direct["mse_ratio_mean"],
        })
    supports_prediction_route = all(row["all_residual_seeds_positive"] for row in decisions)

    if not all(sha256(ROOT / relative) == digest for relative, digest in before.items()):
        raise RuntimeError("protected source artifact changed")
    write_csv(output / "training_summary.csv", training_rows)
    write_csv(output / "per_seed_bearing_metrics.csv", metric_rows)
    write_csv(output / "per_target_errors.csv", target_rows)
    write_csv(output / "summary_by_bearing.csv", summaries)
    write_csv(output / "decision_checks.csv", decisions)
    write_csv(output / "integrity_checks.csv", integrity_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(expected))
    for arm, color in (("direct", "#3274A1"), ("residual", "#C44E52")):
        selected = [next(row for row in summaries if row["arm"] == arm and row["bearing_id"] == bearing) for bearing in expected]
        axes[0].plot(x, [row["mse_ratio_mean"] for row in selected], marker="o", color=color, label=arm)
        axes[1].plot(x, [row["win_fraction_mean"] for row in selected], marker="o", color=color, label=arm)
    labels = [bearing.replace("Bearing", "B") for bearing in expected]
    axes[0].axhline(1, color="black", linestyle=":", label="persistence")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE / persistence MSE (log)")
    axes[0].set_title("One-step forecast ratio")
    axes[1].axhline(0.5, color="black", linestyle=":")
    axes[1].set_ylabel("Fraction of target steps beating persistence")
    axes[1].set_title("Per-target wins")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Direct versus residual GRU; mean across three seeds")
    fig.savefig(output / "residual_gru_comparison.png", dpi=170)
    plt.close(fig)

    report = {
        "status": "completed",
        "verification_status": "predeclared_termination_experiment_with_exact_direct_reproduction",
        "config": cfg, "protected_sha256": before, "protected_files_unchanged": True,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in (
                Path(__file__), ROOT / "src/state_interpreter/residual_gru.py",
                ROOT / "src/state_interpreter/gru_temporal.py",
                ROOT / "src/state_interpreter/forecast_evaluation.py",
            )
        },
        "checkpoint_sha256": {path.name: sha256(path) for path in residual_paths},
        "environment": {
            "executable": sys.executable, "python": sys.version, "torch": str(torch.__version__),
            "numpy": np.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "training_runs": len(cfg["random_seeds"]), "checkpoint_selection": "Bearing1_4 full-sequence next-step MSE",
        "supports_one_step_prediction_as_primary_route": supports_prediction_route,
        "decision": "continue_prediction_route" if supports_prediction_route else "stop_prediction_as_primary_health_level_route",
        "summaries": summaries, "decision_checks": decisions,
        "row_counts": {
            "training_history": len(histories), "per_seed_bearing_metrics": len(metric_rows),
            "per_target_errors": len(target_rows), "summary_by_bearing": len(summaries),
            "integrity_checks": len(integrity_rows),
        },
        "duration_seconds": time.monotonic() - started,
        "fallacy_scan_coverage": "11/11",
        "limitations": [
            "All bearings were previously inspected; this is not a blind test.",
            "Bearing1_4 selects checkpoints and Bearing1_5 has been repeatedly inspected.",
            "Three seeds quantify initialization variability, not between-device uncertainty.",
            "The decision threshold is a predeclared engineering gate, not a significance test.",
            "Positive forecast skill would not itself establish physical health semantics.",
        ],
    }
    write_json(output / "experiment_report.json", report)
    write_json(output / "execution_record.json", {
        "status": "completed", "duration_seconds": report["duration_seconds"],
        "config_sha256": sha256(config_path), "decision": report["decision"],
    })
    print(json.dumps({
        "status": report["status"], "decision": report["decision"],
        "row_counts": report["row_counts"], "summaries": summaries,
        "decision_checks": decisions,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
