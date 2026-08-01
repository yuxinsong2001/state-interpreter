"""Extract ordered AutoEncoder latents and analyze degradation structure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

repository_src = Path(__file__).resolve().parents[1] / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.data import ChannelStandardizer, materialize_stft_data
from state_interpreter.encoders import SmallConvAutoEncoder
from state_interpreter.preprocessing import LogSTFTPreprocessor


def first_fraction_mask(
    episode_ids: list[str], step_ids: list[int], fraction: float
) -> np.ndarray:
    """Select at least one earliest sample per episode."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    mask = np.zeros(len(episode_ids), dtype=bool)
    for episode_id in sorted(set(episode_ids)):
        indices = [i for i, value in enumerate(episode_ids) if value == episode_id]
        indices.sort(key=lambda i: step_ids[i])
        count = max(1, math.ceil(len(indices) * fraction))
        mask[indices[:count]] = True
    return mask


def consecutive_displacement(
    z: np.ndarray, episode_ids: list[str], step_ids: list[int]
) -> np.ndarray:
    """Return ||z_t-z_(t-1)|| within each ordered episode."""

    displacement = np.full(len(z), np.nan, dtype=np.float64)
    previous: dict[str, tuple[int, np.ndarray]] = {}
    for index, (episode_id, step_id) in enumerate(zip(episode_ids, step_ids)):
        if episode_id in previous:
            previous_step, previous_z = previous[episode_id]
            if step_id <= previous_step:
                raise ValueError(f"steps are not increasing in {episode_id}")
            displacement[index] = float(np.linalg.norm(z[index] - previous_z))
        previous[episode_id] = (step_id, z[index])
    return displacement


def load_checkpoint(path: Path) -> tuple[dict, SmallConvAutoEncoder]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_config = checkpoint["model"]
    model = SmallConvAutoEncoder(
        input_channels=int(model_config["input_channels"]),
        latent_dim=int(model_config["latent_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def extract_bearing(
    *,
    root: Path,
    condition: str,
    bearing: str,
    preprocessor: LogSTFTPreprocessor,
    standardizer: ChannelStandardizer,
    model: SmallConvAutoEncoder,
    batch_size: int,
) -> dict[str, object]:
    adapter = XJTUSYDatasetAdapter(
        root, conditions=[condition], bearings=[bearing]
    )
    data = materialize_stft_data(adapter, preprocessor)
    inputs = standardizer.transform(data.tensors)
    latent_batches: list[torch.Tensor] = []
    reconstruction_mse: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            output = model(batch)
            latent_batches.append(output.z.cpu())
            reconstruction_mse.append(
                (output.reconstruction - batch).square().mean(dim=(1, 2, 3)).cpu()
            )
    return {
        "z": torch.cat(latent_batches).numpy(),
        "reconstruction_mse": torch.cat(reconstruction_mse).numpy(),
        "episode_ids": list(data.episode_ids),
        "step_ids": list(data.step_ids),
        "source_paths": list(data.source_paths),
    }


def save_line_plot(
    output: Path,
    values: np.ndarray,
    episode_ids: list[str],
    step_ids: list[int],
    ylabel: str,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    for episode_id in sorted(set(episode_ids)):
        indices = [i for i, value in enumerate(episode_ids) if value == episode_id]
        axis.plot(
            [step_ids[i] for i in indices],
            values[indices],
            linewidth=1.6,
            label=episode_id.split("/")[-1],
        )
    axis.set_xlabel("Measurement step (1 minute interval)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--healthy-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    checkpoint_path = Path(args.checkpoint)
    checkpoint, model = load_checkpoint(checkpoint_path)
    condition = str(checkpoint["condition"])
    split = checkpoint["split"]
    bearings = split["train"] + split["validation"] + split["holdout_test"]
    split_by_bearing = {
        bearing: split_name
        for split_name, selected in split.items()
        for bearing in selected
    }
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

    all_z: list[np.ndarray] = []
    all_reconstruction_mse: list[np.ndarray] = []
    episode_ids: list[str] = []
    step_ids: list[int] = []
    source_paths: list[str] = []
    bearing_ids: list[str] = []
    split_names: list[str] = []
    for bearing in bearings:
        print(f"EXTRACTING {bearing}", flush=True)
        result = extract_bearing(
            root=Path(args.root),
            condition=condition,
            bearing=bearing,
            preprocessor=preprocessor,
            standardizer=standardizer,
            model=model,
            batch_size=args.batch_size,
        )
        z = result["z"]
        count = len(z)
        all_z.append(z)
        all_reconstruction_mse.append(result["reconstruction_mse"])
        episode_ids.extend(result["episode_ids"])
        step_ids.extend(result["step_ids"])
        source_paths.extend(result["source_paths"])
        bearing_ids.extend([bearing] * count)
        split_names.extend([split_by_bearing[bearing]] * count)

    z = np.concatenate(all_z)
    reconstruction_mse = np.concatenate(all_reconstruction_mse)
    if not np.isfinite(z).all() or not np.isfinite(reconstruction_mse).all():
        raise RuntimeError("analysis contains non-finite model outputs")

    train_mask = np.asarray([name == "train" for name in split_names])
    early_mask = first_fraction_mask(episode_ids, step_ids, args.healthy_fraction)
    healthy_reference_mask = train_mask & early_mask
    healthy_center = z[healthy_reference_mask].mean(axis=0)
    health_distance = np.linalg.norm(z - healthy_center, axis=1)
    delta_z = consecutive_displacement(z, episode_ids, step_ids)

    pca = PCA(n_components=2, random_state=args.seed)
    pca.fit(z[train_mask])
    pca_coordinates = pca.transform(z)
    tsne_coordinates = TSNE(
        n_components=2,
        perplexity=min(30.0, max(5.0, (len(z) - 1) / 3.0)),
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(z)

    normalized_time = np.empty(len(z), dtype=np.float64)
    for episode_id in sorted(set(episode_ids)):
        indices = [i for i, value in enumerate(episode_ids) if value == episode_id]
        maximum = max(step_ids[i] for i in indices)
        denominator = max(1, maximum)
        normalized_time[indices] = [step_ids[i] / denominator for i in indices]

    fieldnames = (
        [
            "condition",
            "bearing_id",
            "split",
            "episode_id",
            "step_id",
            "time_minutes",
            "normalized_lifetime",
            "reconstruction_mse",
            "health_center_distance",
            "delta_z",
            "pca_1",
            "pca_2",
            "tsne_1",
            "tsne_2",
            "source_path",
        ]
        + [f"z_{index}" for index in range(z.shape[1])]
    )
    with (output_dir / "latent_trajectories.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(z)):
            row = {
                "condition": condition,
                "bearing_id": bearing_ids[index],
                "split": split_names[index],
                "episode_id": episode_ids[index],
                "step_id": step_ids[index],
                "time_minutes": float(step_ids[index]),
                "normalized_lifetime": float(normalized_time[index]),
                "reconstruction_mse": float(reconstruction_mse[index]),
                "health_center_distance": float(health_distance[index]),
                "delta_z": "" if np.isnan(delta_z[index]) else float(delta_z[index]),
                "pca_1": float(pca_coordinates[index, 0]),
                "pca_2": float(pca_coordinates[index, 1]),
                "tsne_1": float(tsne_coordinates[index, 0]),
                "tsne_2": float(tsne_coordinates[index, 1]),
                "source_path": source_paths[index],
            }
            row.update({f"z_{j}": float(value) for j, value in enumerate(z[index])})
            writer.writerow(row)

    summaries: list[dict[str, object]] = []
    for bearing in bearings:
        indices = np.asarray([i for i, value in enumerate(bearing_ids) if value == bearing])
        rho, p_value = spearmanr(
            normalized_time[indices], health_distance[indices]
        )
        summaries.append(
            {
                "bearing_id": bearing,
                "split": split_by_bearing[bearing],
                "samples": len(indices),
                "reconstruction_mse_mean": float(reconstruction_mse[indices].mean()),
                "reconstruction_mse_last": float(reconstruction_mse[indices[-1]]),
                "health_distance_first": float(health_distance[indices[0]]),
                "health_distance_last": float(health_distance[indices[-1]]),
                "health_distance_spearman_rho": float(rho),
                "health_distance_spearman_p": float(p_value),
                "delta_z_mean": float(np.nanmean(delta_z[indices])),
                "delta_z_max": float(np.nanmax(delta_z[indices])),
            }
        )
    with (output_dir / "bearing_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    fig, axis = plt.subplots(figsize=(9, 7))
    for bearing in bearings:
        indices = np.asarray([i for i, value in enumerate(bearing_ids) if value == bearing])
        scatter = axis.scatter(
            pca_coordinates[indices, 0],
            pca_coordinates[indices, 1],
            c=normalized_time[indices],
            cmap="viridis",
            s=18,
            alpha=0.8,
            label=bearing,
        )
        axis.plot(pca_coordinates[indices, 0], pca_coordinates[indices, 1], alpha=0.25)
    axis.set_xlabel("PCA 1")
    axis.set_ylabel("PCA 2")
    axis.set_title("Latent trajectories (PCA fitted on training bearings only)")
    axis.legend(fontsize=8)
    fig.colorbar(scatter, ax=axis, label="Normalized lifetime")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_trajectories.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 7))
    for bearing in bearings:
        indices = np.asarray([i for i, value in enumerate(bearing_ids) if value == bearing])
        axis.scatter(
            tsne_coordinates[indices, 0],
            tsne_coordinates[indices, 1],
            c=normalized_time[indices],
            cmap="viridis",
            s=18,
            alpha=0.8,
            label=bearing,
        )
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.set_title("t-SNE visualization (qualitative only)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_latent_space.png", dpi=180)
    plt.close(fig)

    save_line_plot(
        output_dir / "health_center_distance.png",
        health_distance,
        episode_ids,
        step_ids,
        "Euclidean distance",
        "Distance from training-early healthy center",
    )
    save_line_plot(
        output_dir / "delta_z.png",
        delta_z,
        episode_ids,
        step_ids,
        "Consecutive latent displacement",
        "Latent movement between adjacent measurements",
    )
    save_line_plot(
        output_dir / "reconstruction_mse.png",
        reconstruction_mse,
        episode_ids,
        step_ids,
        "MSE on standardized log-STFT",
        "AutoEncoder reconstruction error",
    )

    report = {
        "status": "completed",
        "checkpoint": str(checkpoint_path.resolve()),
        "condition": condition,
        "split": split,
        "latent_dim": int(z.shape[1]),
        "sample_count": int(len(z)),
        "healthy_reference": {
            "definition": "first fraction of each training bearing",
            "fraction": args.healthy_fraction,
            "sample_count": int(healthy_reference_mask.sum()),
            "center": healthy_center.tolist(),
        },
        "pca": {
            "fit_on": split["train"],
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        },
        "tsne": {
            "role": "qualitative visualization only",
            "random_state": args.seed,
        },
        "bearing_summaries": summaries,
        "interpretation_gate": {
            "passed": False,
            "reason": "Metrics are generated; scientific interpretation remains required.",
        },
        "outputs": [
            "latent_trajectories.csv",
            "bearing_summary.csv",
            "pca_trajectories.png",
            "tsne_latent_space.png",
            "health_center_distance.png",
            "delta_z.png",
            "reconstruction_mse.png",
        ],
    }
    with (output_dir / "analysis_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
