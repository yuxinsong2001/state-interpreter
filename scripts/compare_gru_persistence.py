"""Evaluate frozen predictive GRUs against the same-origin persistence forecast."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
# Dependency preflight occurs before any experiment output is created.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from state_interpreter import PredictiveGRUStateInterpreter
from state_interpreter.forecast_evaluation import aligned_squared_errors, forecast_metrics
from run_gru_interpreter_exploratory import center_sequence, load_latents, sha256, to_latent


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg["forecast_horizon"] != 1:
        raise ValueError("only the predeclared one-step horizon is implemented")
    output = (ROOT / cfg["output_directory"]).resolve()
    if not output.is_relative_to(ROOT / "results") or output.exists():
        raise ValueError("output must be a new directory inside results")
    started = time.monotonic()
    source = ROOT / cfg["checkpoint_directory"]
    input_path = ROOT / cfg["input_latents"]
    archive_path = source / "experiment_report.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    training_config_path = ROOT / "configs/xjtu_gru_temporal_ranking_v1.json"
    if sha256(training_config_path) != archive["config_sha256"]:
        raise ValueError("archived training configuration changed")
    if cfg["calibration_steps"] != archive["config"]["calibration_steps"]:
        raise ValueError("calibration differs from archived training")
    if sha256(input_path) != archive["input_sha256"]:
        raise ValueError("input hash differs from archived training run")
    checkpoints = {(arm, seed): source / f"{arm}_seed_{seed}_checkpoint.pt" for arm in cfg["arms"] for seed in cfg["random_seeds"]}
    for path in checkpoints.values():
        if sha256(path) != archive["checkpoint_sha256"][path.name]:
            raise ValueError(f"checkpoint hash changed: {path.name}")
    protected = [config_path, training_config_path, input_path, archive_path, source / "per_seed_bearing_metrics.csv", *checkpoints.values()]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    by_bearing, columns = load_latents(input_path)
    if columns != [f"z_{i}" for i in range(8)]:
        raise ValueError("expected exactly eight latent columns")
    if sorted(by_bearing) != sorted(cfg["bearings"]):
        raise ValueError("unexpected bearing inventory")
    for b, rows in by_bearing.items():
        if [int(r["step_id"]) for r in rows] != list(range(len(rows))):
            raise ValueError(f"noncontiguous step IDs in {b}")
    old_metrics = {(r["arm"], int(r["seed"]), r["bearing_id"]): float(r["next_step_mse"]) for r in read_csv(source / "per_seed_bearing_metrics.csv") if r["arm"] in cfg["arms"]}
    centered = {b: center_sequence(to_latent(rows, columns), cfg["calibration_steps"]) for b, rows in by_bearing.items()}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metric_rows, step_rows, checks = [], [], []
    baseline_by_bearing = {}
    reference_scaler = None
    for (arm, seed), path in checkpoints.items():
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        if checkpoint["arm"] != arm or checkpoint["random_seed"] != seed or checkpoint["embedding_dim"] != len(columns):
            raise ValueError("checkpoint identity/dimension mismatch")
        if checkpoint["config_sha256"] != archive["config_sha256"]:
            raise ValueError("checkpoint training config mismatch")
        mean, std = checkpoint["feature_mean"], checkpoint["feature_std"]
        if mean.shape != (len(columns),) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("invalid saved normalization statistics")
        if reference_scaler is None:
            reference_scaler = (mean.clone(), std.clone())
        if not torch.equal(mean, reference_scaler[0]) or not torch.equal(std, reference_scaler[1]):
            raise ValueError("arms or seeds have different normalization scales")
        model = PredictiveGRUStateInterpreter(embedding_dim=checkpoint["embedding_dim"], hidden_dim=checkpoint["hidden_dim"], num_layers=checkpoint["num_layers"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for b in cfg["bearings"]:
            sequence = (centered[b] - mean) / std
            with torch.no_grad():
                forecast = model(sequence[None]).next_embedding_prediction[0]
                k = min(cfg["prefix_check_length"], len(sequence)-1)
                prefix = model(sequence[:k][None]).next_embedding_prediction[0]
            prefix_difference = float((prefix-forecast[:k]).abs().max())
            if prefix_difference > cfg["prefix_tolerance"]:
                raise ValueError("causal prefix consistency failure")
            g, p = aligned_squared_errors(sequence, forecast)
            historical_mse = float(g.mean())
            reproduction_difference = abs(historical_mse - old_metrics[(arm, seed, b)])
            if reproduction_difference != 0:
                raise ValueError(f"archived MSE mismatch for {arm}/{seed}/{b}")
            if b in baseline_by_bearing:
                torch.testing.assert_close(p, baseline_by_bearing[b], atol=0, rtol=0)
            else:
                baseline_by_bearing[b] = p.clone()
            independent_baseline = torch.diff(sequence, dim=0).square()
            torch.testing.assert_close(p, independent_baseline, atol=0, rtol=0)
            checks.append(dict(arm=arm, seed=seed, bearing_id=b, archived_mse_absolute_difference=reproduction_difference, prefix_max_absolute_difference=prefix_difference, baseline_identity_exact=True))
            for scope, start in (("primary", cfg["primary_target_start_step"]), ("full_sequence_diagnostic", 1)):
                metric_rows.append(dict(arm=arm, seed=seed, bearing_id=b, split=by_bearing[b][0]["split"], scope=scope, target_start_step=start, **forecast_metrics(g, p, target_start_step=start)))
            g_steps, p_steps = g.double().mean(1), p.double().mean(1)
            for origin in range(len(sequence)-1):
                step_rows.append(dict(arm=arm, seed=seed, bearing_id=b, origin_step=origin, target_step=origin+1, included_in_primary=origin+1 >= cfg["primary_target_start_step"], gru_step_mse=float(g_steps[origin]), persistence_step_mse=float(p_steps[origin]), gru_minus_persistence=float(g_steps[origin]-p_steps[origin])))
        print(f"Verified and scored frozen {arm} seed={seed}", flush=True)
    summaries = []
    for scope in ("primary", "full_sequence_diagnostic"):
        for arm in cfg["arms"]:
            for b in cfg["bearings"]:
                selected = [r for r in metric_rows if r["scope"] == scope and r["arm"] == arm and r["bearing_id"] == b]
                skills = [r["skill"] for r in selected if r["skill"] is not None]
                ratios = [r["mse_ratio"] for r in selected if r["mse_ratio"] is not None]
                summaries.append(dict(scope=scope, arm=arm, bearing_id=b, split=selected[0]["split"], target_count=selected[0]["target_count"], persistence_mse=selected[0]["persistence_mse"], gru_mse_mean=float(np.mean([r["gru_mse"] for r in selected])), ratio_mean=float(np.mean(ratios)) if ratios else None, skill_mean=float(np.mean(skills)) if skills else None, skill_sample_std=float(np.std(skills, ddof=1)) if len(skills)>1 else None, skill_min=min(skills) if skills else None, skill_max=max(skills) if skills else None, positive_skill_seed_count=sum(s>0 for s in skills), gru_win_fraction_mean=float(np.mean([r["gru_win_fraction"] for r in selected]))))
    decisions = []
    for arm in cfg["arms"]:
        rows = [r for r in summaries if r["scope"] == "primary" and r["arm"] == arm and r["bearing_id"] in ("Bearing1_4", "Bearing1_5")]
        decisions.append(dict(arm=arm, all_seeds_positive_on_both_bearings=all(r["positive_skill_seed_count"] == len(cfg["random_seeds"]) for r in rows)))
    if not all(sha256(ROOT/p) == digest for p, digest in before.items()):
        raise RuntimeError("protected input/config/checkpoint was modified")
    # Create artifacts only after inference and integrity checks have passed.
    output.mkdir(parents=True)
    write_csv(output / "per_target_errors.csv", step_rows)
    write_csv(output / "per_seed_bearing_metrics.csv", metric_rows)
    write_csv(output / "summary_by_bearing.csv", summaries)
    write_csv(output / "integrity_checks.csv", checks)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, arm in zip(axes, cfg["arms"]):
        for si, seed in enumerate(cfg["random_seeds"]):
            values = [next(r["mse_ratio"] for r in metric_rows if r["scope"]=="primary" and r["arm"]==arm and r["seed"]==seed and r["bearing_id"]==b) for b in cfg["bearings"]]
            ax.plot(np.arange(len(values)), values, marker="o", label=str(seed), alpha=0.8)
        ax.axhline(1, color="black", linestyle="--", label="Persistence")
        ax.set_xticks(np.arange(5), [b.replace("Bearing", "B") for b in cfg["bearings"]])
        ax.set_yscale("log")
        ax.set_ylabel("GRU MSE / persistence MSE (log scale)")
        ax.set_title(arm)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Frozen one-step forecasts; lower than 1 beats persistence")
    fig.savefig(output / "prediction_skill.png", dpi=170)
    plt.close(fig)
    report = dict(status="completed", verification_status="exact_archived_mse_and_baseline_identity", config=cfg, protected_sha256=before, protected_files_unchanged=True, source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__), ROOT/"src/state_interpreter/forecast_evaluation.py", ROOT/"src/state_interpreter/gru_temporal.py", ROOT/"scripts/run_gru_interpreter_exploratory.py")}, environment=dict(executable=sys.executable, python=sys.version, torch=str(torch.__version__), numpy=np.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(), git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()), training_performed=False, checkpoint_selection_performed=False, duration_seconds=time.monotonic()-started, row_counts=dict(per_target_errors=len(step_rows), per_seed_bearing_metrics=len(metric_rows), summary_by_bearing=len(summaries), integrity_checks=len(checks)), max_archived_mse_difference=max(r["archived_mse_absolute_difference"] for r in checks), max_prefix_difference=max(r["prefix_max_absolute_difference"] for r in checks), summaries=summaries, decisions=decisions, limitations=["Already inspected bearings, not a new blind test.","All checkpoints selected using Bearing1_4 prediction MSE; Bearing1_5 repeatedly inspected.","Seed variability is not between-device uncertainty.","One-step prediction skill is not proof of physical health semantics or downstream decision utility.","MSE and ratio can be driven by rare large errors; median and per-target win fraction are also reported.","Full-sequence pre-calibration scores reproduce historical arithmetic but do not demonstrate online forecasting before calibration ends."])
    write_json(output / "experiment_report.json", report)
    print(json.dumps(dict(status="completed", summaries=[r for r in summaries if r["scope"]=="primary"], decisions=decisions, row_counts=report["row_counts"], max_archived_mse_difference=report["max_archived_mse_difference"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
