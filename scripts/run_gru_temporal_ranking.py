"""Paired prediction-only versus temporal-ranking GRU objective experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(path))

import numpy as np
import torch

from state_interpreter import PredictiveGRUStateInterpreter
from state_interpreter.temporal_ranking import full_window_distance_level, temporal_ranking_loss
from run_gru_interpreter_exploratory import (
    center_sequence, distance_baseline_level, gru_level, load_latents,
    prediction_mse, sha256, to_latent, trajectory_metrics,
)
from run_gru_multiseed_stability import compare_checkpoint_parameters, write_csv


def save_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def auxiliary_loss(model, sequences, cfg):
    losses = []
    for sequence in sequences:
        hidden = model(sequence[None]).hidden_sequence[0]
        level = full_window_distance_level(
            hidden, calibration_steps=cfg["calibration_steps"], temporal_window=cfg["temporal_window"]
        )
        losses.append(temporal_ranking_loss(level, pair_lag=cfg["pair_lag"], margin=cfg["ranking_margin"]))
    return torch.stack(losses).mean()


def train(seed, arm, sequences, validation, cfg, started):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = PredictiveGRUStateInterpreter(embedding_dim=sequences[0].shape[1], hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])
    initial_state = deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best = math.inf
    best_epoch = 0
    best_state = None
    no_improvement = 0
    history = []
    weight = cfg["ranking_weight"] if arm == "ranking" else 0.0
    for epoch in range(1, cfg["max_epochs"] + 1):
        if time.monotonic() - started > cfg["hard_timeout_seconds"]:
            raise TimeoutError("predeclared experiment hard timeout reached")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = prediction_mse(model, sequences)
        # Preserve the exact legacy computation path for the zero-weight control.
        if weight:
            rank = auxiliary_loss(model, sequences, cfg)
            loss = prediction + weight * rank
        else:
            with torch.no_grad():
                rank = auxiliary_loss(model, sequences, cfg)
            loss = prediction
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"]))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(prediction_mse(model, [validation]))
        if not all(math.isfinite(x) for x in (float(loss.detach()), val, grad)):
            raise ValueError("nonfinite training result")
        history.append(dict(arm=arm, seed=seed, epoch=epoch, train_prediction_mse=float(prediction.detach()), train_ranking_loss=float(rank.detach()), train_total_loss=float(loss.detach()), validation_prediction_mse=val, gradient_norm=grad))
        if val < best - cfg["minimum_improvement"]:
            best, best_epoch, best_state, no_improvement = val, epoch, deepcopy(model.state_dict()), 0
        else:
            no_improvement += 1
        if epoch % 50 == 0:
            print(f"{arm} seed={seed} epoch={epoch} val_mse={val:.6f} rank_loss={float(rank.detach()):.6f}", flush=True)
        if no_improvement >= cfg["patience"]:
            break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    return model.eval(), initial_state, history, best_epoch, best


def evaluate_level(model, sequence, cfg):
    with torch.no_grad():
        output = model(sequence[None])
        level, _ = gru_level(output.hidden_sequence[0], cfg["calibration_steps"], cfg["temporal_window"])
        mse = float((output.next_embedding_prediction[0, :-1] - sequence[1:]).square().mean())
    return level, mse


def metrics(level, lifetime, cfg):
    mask = (np.arange(len(level)) >= cfg["common_score_start_step"]) & np.isfinite(level)
    result = trajectory_metrics(lifetime, level, mask)
    dynamic_range = float(np.ptp(level[mask]))
    result["dynamic_range"] = dynamic_range
    result["range_normalized_step_std"] = result["step_std"] / dynamic_range if dynamic_range > 1e-12 else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    output_dir = (ROOT / cfg["output_dir"]).resolve()
    if not output_dir.is_relative_to(ROOT / "results"):
        raise ValueError("output must stay within the repository results directory")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    input_path = ROOT / cfg["input_latents"]
    reference_dir = ROOT / cfg["reference_checkpoints"]
    by_bearing, columns = load_latents(input_path)
    train_names = cfg["train_bearings"]
    validation = cfg["validation_bearing"]
    holdout = cfg["old_holdout_bearing"]
    if set(train_names) & {validation, holdout} or validation == holdout:
        raise ValueError("overlapping split")
    if set(by_bearing) != set(train_names + [validation, holdout]):
        raise ValueError("unexpected bearing population")
    if any(any(row["split"] != "train" for row in by_bearing[b]) for b in train_names):
        raise ValueError("training split does not match source metadata")
    if len(set(cfg["random_seeds"])) != 3:
        raise ValueError("three unique seeds required")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    raw = {b: to_latent(rows, columns) for b, rows in by_bearing.items()}
    for b, rows in by_bearing.items():
        if [int(row["step_id"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"noncontiguous steps: {b}")
    centered = {b: center_sequence(x, cfg["calibration_steps"]) for b, x in raw.items()}
    fit = torch.cat([centered[b] for b in train_names])
    mean, std = fit.mean(0), fit.std(0, unbiased=False).clamp_min(1e-6)
    standardized = {b: (x - mean) / std for b, x in centered.items()}
    sequences = [standardized[b] for b in train_names]
    reference_paths = [reference_dir / f"seed_{seed}_checkpoint.pt" for seed in cfg["random_seeds"]]
    immutable_paths = [input_path, config_path, *reference_paths]
    initial_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in immutable_paths}
    source_paths = [Path(__file__), ROOT / "src/state_interpreter/temporal_ranking.py", ROOT / "src/state_interpreter/gru_temporal.py", ROOT / "scripts/run_gru_interpreter_exploratory.py", ROOT / "scripts/run_gru_multiseed_stability.py"]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    output_dir.mkdir(parents=True)
    save_json(output_dir / "execution_record.json", dict(status="running", config_sha256=sha256(config_path), started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    all_history, training_rows, reproduction, checkpoint_paths = [], [], [], []
    for seed in cfg["random_seeds"]:
        paired_initial = None
        for arm in ("control", "ranking"):
            model, initial, history, epoch, best = train(seed, arm, sequences, standardized[validation], cfg, started)
            if arm == "control":
                paired_initial = initial
                old = torch.load(reference_dir / f"seed_{seed}_checkpoint.pt", map_location="cpu", weights_only=True)
                diff, count = compare_checkpoint_parameters(model.state_dict(), old["model_state_dict"])
                exact = diff == 0 and epoch == old["best_epoch"] and best == old["best_validation_next_step_mse"]
                exact = exact and torch.equal(mean, old["feature_mean"]) and torch.equal(std, old["feature_std"])
                reproduction.append(dict(seed=seed, parameter_max_abs_difference=diff, mismatched_parameters=count, exact_match=exact))
                if not exact:
                    raise RuntimeError(f"control reproduction mismatch at seed {seed}; see checkpoint verification")
            else:
                diff, count = compare_checkpoint_parameters(initial, paired_initial)
                if count:
                    raise RuntimeError("paired initializations differ")
            path = output_dir / f"{arm}_seed_{seed}_checkpoint.pt"
            torch.save(dict(model_state_dict=model.state_dict(), embedding_dim=len(columns), hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], feature_mean=mean, feature_std=std, best_epoch=epoch, best_validation_next_step_mse=best, random_seed=seed, arm=arm, config_sha256=sha256(config_path)), path)
            checkpoint_paths.append((arm, seed, path))
            all_history.extend(history)
            training_rows.append(dict(arm=arm, seed=seed, epochs_completed=len(history), best_epoch=epoch, validation_prediction_mse=best, paired_initialization_exact=True))
            write_csv(output_dir / "training_history.csv", all_history)
            print(f"LOCKED {arm} seed={seed} checkpoint_epoch={epoch}", flush=True)
    # No outcomes from the old holdout are computed until all six checkpoints lock.
    state_rows, metric_rows, diagnostic_rows = [], [], []
    levels = {}
    lifetimes = {b: np.array([float(row["normalized_lifetime"]) for row in rows]) for b, rows in by_bearing.items()}
    for b in sorted(raw):
        level = distance_baseline_level(raw[b], cfg["calibration_steps"], cfg["temporal_window"])
        levels[("distance", 0, b)] = level
        metric_rows.append(dict(arm="distance", seed=0, bearing_id=b, **metrics(level, lifetimes[b], cfg), next_step_mse=None))
    for arm, seed, path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = PredictiveGRUStateInterpreter(embedding_dim=len(columns), hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for bi, b in enumerate(sorted(raw)):
            sequence = standardized[b]
            level, mse = evaluate_level(model, sequence, cfg)
            levels[(arm, seed, b)] = level
            actual = metrics(level, lifetimes[b], cfg)
            metric_rows.append(dict(arm=arm, seed=seed, bearing_id=b, **actual, next_step_mse=mse))
            for step in range(len(level)):
                state_rows.append(dict(arm=arm, seed=seed, bearing_id=b, step_id=step, level=float(level[step]) if np.isfinite(level[step]) else "", distance_level=levels[("distance", 0, b)][step] if np.isfinite(levels[("distance", 0, b)][step]) else "", included_in_score=step >= cfg["common_score_start_step"]))
            for diagnostic in ("constant_after_calibration", "shuffled_after_calibration"):
                altered = sequence.clone()
                c = cfg["calibration_steps"]
                if diagnostic.startswith("constant"):
                    altered[c:] = altered[:c].mean(0)
                else:
                    order = np.random.default_rng(cfg["negative_control_seed"] + bi).permutation(len(sequence) - c)
                    altered[c:] = sequence[c:][torch.from_numpy(order)]
                diagnostic_level, _ = evaluate_level(model, altered, cfg)
                dm = metrics(diagnostic_level, lifetimes[b], cfg)
                ratio = dm["dynamic_range"] / max(actual["dynamic_range"], 1e-12)
                diagnostic_rows.append(dict(arm=arm, seed=seed, bearing_id=b, diagnostic=diagnostic, **dm, range_fraction_of_real=ratio, range_flag=bool(diagnostic.startswith("constant") and ratio > cfg["decision"]["constant_input_range_fraction_flag"])))

    summaries = []
    for arm in ("distance", "control", "ranking"):
        for b in sorted(raw):
            selected = [r for r in metric_rows if r["arm"] == arm and r["bearing_id"] == b]
            rhos = np.array([r["spearman_rho"] for r in selected])
            summaries.append(dict(arm=arm, bearing_id=b, rho_mean=float(rhos.mean()), rho_std=float(rhos.std(ddof=1)) if len(rhos) > 1 else None, rho_min=float(rhos.min()), rho_max=float(rhos.max()), next_step_mse_mean=float(np.mean([r["next_step_mse"] for r in selected])) if arm != "distance" else None, backward_fraction_mean=float(np.mean([r["backward_step_fraction"] for r in selected])), normalized_step_std_mean=float(np.mean([r["range_normalized_step_std"] for r in selected if r["range_normalized_step_std"] is not None])), collapse_count=sum(r["collapsed"] for r in selected)))
    decisions = []
    for b in (validation, holdout):
        control = next(r for r in summaries if r["arm"] == "control" and r["bearing_id"] == b)
        ranking = next(r for r in summaries if r["arm"] == "ranking" and r["bearing_id"] == b)
        rule = cfg["decision"]
        checks = dict(std_reduced_20_percent=ranking["rho_std"] <= (1-rule["required_std_reduction_fraction_each_bearing"]) * control["rho_std"], mean_rho_preserved=ranking["rho_mean"] >= control["rho_mean"] - rule["maximum_mean_rho_drop_each_bearing"], worst_seed_preserved=ranking["rho_min"] >= control["rho_min"], prediction_preserved=ranking["next_step_mse_mean"] <= (1+rule["maximum_prediction_mse_increase_fraction_each_bearing"]) * control["next_step_mse_mean"], no_collapse=ranking["collapse_count"] == 0, constant_input_check=not any(r["range_flag"] for r in diagnostic_rows if r["arm"] == "ranking" and r["bearing_id"] == b))
        decisions.append(dict(bearing_id=b, **checks, supports_next_validation=all(checks.values())))
    for name, rows in (("training_summary.csv", training_rows), ("per_seed_bearing_metrics.csv", metric_rows), ("per_step_levels.csv", state_rows), ("negative_control_metrics.csv", diagnostic_rows), ("summary_by_bearing.csv", summaries), ("decision_checks.csv", decisions)):
        write_csv(output_dir / name, rows)
    save_json(output_dir / "reproduction_check.json", reproduction)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, b in zip(axes.flat, sorted(raw)):
        for arm, color in (("distance", "black"), ("control", "tab:blue"), ("ranking", "tab:orange")):
            seeds = [0] if arm == "distance" else cfg["random_seeds"]
            for si, seed in enumerate(seeds):
                y = levels[(arm, seed, b)]
                mask = np.arange(len(y)) >= cfg["common_score_start_step"]
                selected = y[mask]
                normalized = (selected - selected.min()) / max(float(np.ptp(selected)), 1e-12)
                ax.plot(lifetimes[b][mask], normalized, color=color, alpha=0.7, label=arm if si == 0 else None, linewidth=1.2)
        ax.set_title(b)
        ax.set_xlabel("Normalized lifetime (evaluation only)")
        ax.set_ylabel("Plot-only rescaled Level")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center")
    fig.suptitle("Prediction versus temporal-ranking objective: all three seeds")
    fig.savefig(output_dir / "level_comparison.png", dpi=160)
    plt.close(fig)
    unchanged = all(sha256(ROOT / p) == h for p, h in initial_hashes.items())
    if not unchanged:
        raise RuntimeError("immutable input/config/checkpoint changed")
    report = dict(status="completed", verification_status="paired_exploratory_with_exact_control_reproduction", config=cfg, immutable_hashes=initial_hashes, immutable_files_unchanged=unchanged, source_sha256=source_hashes, checkpoint_sha256={p.name: sha256(p) for _, _, p in checkpoint_paths}, environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform(), git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()), reproduction=reproduction, summaries=summaries, decisions=decisions, supports_next_validation=all(r["supports_next_validation"] for r in decisions), pure_time_counter_rho=1.0, duration_seconds=time.monotonic()-started, limitations=["Time order is weak training supervision, so lifetime correlation alone is not independent health evidence.", "All bearings were previously inspected; Bearing1_4 selects checkpoints and Bearing1_5 is an old repeatedly inspected holdout.", "Only one fixed loss weight, margin, lag and three seeds were tested; failure does not rule out GRU or ranking in general.", "Constant/shuffled input are artificial diagnostics, not physical fault labels."])
    save_json(output_dir / "experiment_report.json", report)
    save_json(output_dir / "execution_record.json", dict(status="completed", duration_seconds=report["duration_seconds"], config_sha256=sha256(config_path)))
    print(json.dumps(dict(status="completed", duration_seconds=report["duration_seconds"], decisions=decisions, summaries=summaries), indent=2), flush=True)


if __name__ == "__main__":
    main()
