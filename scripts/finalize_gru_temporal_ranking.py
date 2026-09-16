"""Verify and finalize a saved ranking experiment after plot dependency failure.

No training or checkpoint selection occurs here. Existing numeric artifacts must
match frozen-checkpoint inference before a report can be marked completed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(directory))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_gru_temporal_ranking import save_json, evaluate_level
from run_gru_interpreter_exploratory import center_sequence, load_latents, sha256, to_latent
from run_gru_multiseed_stability import compare_checkpoint_parameters
from state_interpreter import PredictiveGRUStateInterpreter


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    directory = ROOT / cfg["output_dir"]
    if (directory / "experiment_report.json").exists() or (directory / "level_comparison.png").exists():
        raise FileExistsError("final artifacts already exist; refusing to overwrite")
    row_counts = {name: len(read_csv(directory / name)) for name in ("training_history.csv", "training_summary.csv", "per_seed_bearing_metrics.csv", "per_step_levels.csv", "negative_control_metrics.csv", "summary_by_bearing.csv", "decision_checks.csv")}
    if row_counts != {"training_history.csv": 1800, "training_summary.csv": 6, "per_seed_bearing_metrics.csv": 35, "per_step_levels.csv": 3696, "negative_control_metrics.csv": 60, "summary_by_bearing.csv": 15, "decision_checks.csv": 2}:
        raise ValueError(f"unexpected artifact counts: {row_counts}")
    protected = [p for p in directory.iterdir() if p.suffix in (".csv", ".pt")] + [directory / "reproduction_check.json", config_path]
    before = {str(p): sha256(p) for p in protected}
    by_bearing, columns = load_latents(ROOT / cfg["input_latents"])
    rows = read_csv(directory / "per_step_levels.csv")
    training_rows = read_csv(directory / "training_summary.csv")
    metric_rows = read_csv(directory / "per_seed_bearing_metrics.csv")
    levels, checkpoint_hashes, exact_controls = {}, {}, []
    maximum = 0.0
    maximum_mse_difference = 0.0
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for arm in ("control", "ranking"):
        for seed in cfg["random_seeds"]:
            path = directory / f"{arm}_seed_{seed}_checkpoint.pt"
            checkpoint_hashes[path.name] = sha256(path)
            saved = torch.load(path, weights_only=True, map_location="cpu")
            if saved["config_sha256"] != sha256(config_path):
                raise ValueError("configuration no longer matches training checkpoints")
            if arm == "control":
                old = torch.load(ROOT / cfg["reference_checkpoints"] / f"seed_{seed}_checkpoint.pt", weights_only=True, map_location="cpu")
                delta, count = compare_checkpoint_parameters(saved["model_state_dict"], old["model_state_dict"])
                exact = delta == 0 and saved["best_epoch"] == old["best_epoch"] and saved["best_validation_next_step_mse"] == old["best_validation_next_step_mse"]
                exact = exact and old["input_sha256"] == sha256(ROOT / cfg["input_latents"])
                if not exact:
                    raise ValueError("original checkpoint or input reproduction mismatch")
                exact_controls.append(dict(seed=seed, exact_match=exact))
            model = PredictiveGRUStateInterpreter(embedding_dim=len(columns), hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])
            model.load_state_dict(saved["model_state_dict"])
            model.eval()
            for b in sorted(by_bearing):
                selected = [r for r in rows if r["arm"] == arm and int(r["seed"]) == seed and r["bearing_id"] == b]
                stored_level = np.array([float(r["level"]) if r["level"] else np.nan for r in selected])
                values = center_sequence(to_latent(by_bearing[b], columns), cfg["calibration_steps"])
                values = (values - saved["feature_mean"]) / saved["feature_std"]
                fresh_level, mse = evaluate_level(model, values, cfg)
                np.testing.assert_array_equal(fresh_level, stored_level)
                valid = np.isfinite(fresh_level)
                maximum = max(maximum, float(np.max(np.abs(fresh_level[valid] - stored_level[valid]))))
                stored_metrics = next(r for r in metric_rows if r["arm"] == arm and int(r["seed"]) == seed and r["bearing_id"] == b)
                maximum_mse_difference = max(maximum_mse_difference, abs(mse - float(stored_metrics["next_step_mse"])))
                levels[(arm, seed, b)] = stored_level
                if arm == "control" and seed == cfg["random_seeds"][0]:
                    levels[("distance", 0, b)] = np.array([float(r["distance_level"]) if r["distance_level"] else np.nan for r in selected])
    if maximum_mse_difference != 0:
        raise ValueError("prediction MSE does not reproduce exactly")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, b in zip(axes.flat, sorted(by_bearing)):
        lifetime = np.array([float(r["normalized_lifetime"]) for r in by_bearing[b]])
        for arm, color in (("distance", "black"), ("control", "tab:blue"), ("ranking", "tab:orange")):
            for si, seed in enumerate([0] if arm == "distance" else cfg["random_seeds"]):
                y = levels[(arm, seed, b)]
                mask = np.arange(len(y)) >= cfg["common_score_start_step"]
                y = y[mask]
                y = (y-y.min()) / max(float(np.ptp(y)), 1e-12)
                ax.plot(lifetime[mask], y, color=color, alpha=0.7, label=arm if si == 0 else None, linewidth=1.2)
        ax.set_title(b)
        ax.set_xlabel("Normalized lifetime (evaluation only)")
        ax.set_ylabel("Plot-only rescaled Level")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center")
    fig.suptitle("Prediction versus temporal-ranking objective: all three seeds")
    fig.savefig(directory / "level_comparison.png", dpi=160)
    plt.close(fig)

    summaries = read_csv(directory / "summary_by_bearing.csv")
    decisions = read_csv(directory / "decision_checks.csv")
    unchanged = all(sha256(Path(p)) == digest for p, digest in before.items())
    if not unchanged:
        raise RuntimeError("finalization changed existing numeric artifacts")
    original_execution = json.loads((directory / "execution_record.json").read_text(encoding="utf-8"))
    save_json(directory / "recovery_record.json", dict(original_execution_record=original_execution, original_exit_code=1, failure="ModuleNotFoundError: matplotlib during final plotting after all training and metric tables completed", action="installed plotting dependency, reloaded frozen checkpoints, verified inference, finalized report", retrained=False, finalization_max_level_difference=maximum, finalization_max_mse_difference=maximum_mse_difference, numeric_artifacts_unchanged=unchanged, row_counts=row_counts))
    report = dict(status="completed_after_plot_dependency_recovery", verification_status="exact_control_and_inference_reproduction", config=cfg, input_sha256=sha256(ROOT / cfg["input_latents"]), config_sha256=sha256(config_path), source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [ROOT / "scripts/run_gru_temporal_ranking.py", Path(__file__), ROOT / "src/state_interpreter/temporal_ranking.py", ROOT / "src/state_interpreter/gru_temporal.py", ROOT / "scripts/run_gru_interpreter_exploratory.py"]}, checkpoint_sha256=checkpoint_hashes, original_checkpoint_sha256={f"seed_{seed}": sha256(ROOT / cfg["reference_checkpoints"] / f"seed_{seed}_checkpoint.pt") for seed in cfg["random_seeds"]}, environment=dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), numpy=np.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(), git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()), row_counts=row_counts, reproduction=exact_controls, finalization_max_level_difference=maximum, finalization_max_mse_difference=maximum_mse_difference, numeric_artifacts_unchanged=unchanged, summaries=summaries, training_summary=training_rows, decisions=decisions, supports_next_validation=all(r["supports_next_validation"] == "True" for r in decisions), pure_time_counter_rho_analytical=1.0, limitations=["Time order is weak training supervision; time correlation does not independently validate health semantics.", "All bearings were inspected previously; Bearing1_4 selects checkpoints and Bearing1_5 is a reused old holdout.", "Only one ranking weight, margin and lag were evaluated.", "Artificial constant and shuffled inputs do not replace physical fault labels."], recovery_record="recovery_record.json")
    save_json(directory / "experiment_report.json", report)
    save_json(directory / "execution_record.json", dict(status="completed_after_plot_dependency_recovery", config_sha256=sha256(config_path), recovery_record="recovery_record.json"))
    print(json.dumps(dict(status=report["status"], supports_next_validation=report["supports_next_validation"], exact_inference_max_difference=maximum, row_counts=row_counts), indent=2))


if __name__ == "__main__":
    main()
