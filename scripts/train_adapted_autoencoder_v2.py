"""Train arm B AutoEncoder on the preregistered target-condition split."""

from __future__ import annotations

import argparse
import csv
import json
import random
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
import torch
from torch import nn
from torch.utils.data import DataLoader

from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.data import ChannelStandardizer, materialize_stft_data
from state_interpreter.encoders import SmallConvAutoEncoder
from state_interpreter.experiment_config import (
    file_sha256,
    load_cross_condition_config,
)
from state_interpreter.preprocessing import LogSTFTPreprocessor


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for (batch,) in loader:
            output = model(batch)
            loss = loss_fn(output.reconstruction, batch)
            total_loss += float(loss) * batch.shape[0]
            total_samples += batch.shape[0]
    if total_samples == 0:
        raise ValueError("evaluation loader is empty")
    return total_loss / total_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    started = time.monotonic()
    config_path = Path(args.config).resolve()
    config = load_cross_condition_config(config_path)
    train_bearings = config.bearings_for("arm_b_train")
    validation_bearings = config.bearings_for("arm_b_validation")
    config.split.assert_excludes_holdout(train_bearings, purpose="arm B training")
    config.split.assert_excludes_holdout(
        validation_bearings, purpose="arm B validation"
    )

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    random.seed(config.arm_b_seed)
    torch.manual_seed(config.arm_b_seed)
    torch.use_deterministic_algorithms(True)

    preprocessor = LogSTFTPreprocessor(
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        output_size=config.output_size,
    )
    train_adapter = XJTUSYDatasetAdapter(
        args.root,
        conditions=[config.target_condition],
        bearings=train_bearings,
    )
    validation_adapter = XJTUSYDatasetAdapter(
        args.root,
        conditions=[config.target_condition],
        bearings=validation_bearings,
    )
    print(f"MATERIALIZING_TRAIN bearings={','.join(train_bearings)}", flush=True)
    train_data = materialize_stft_data(train_adapter, preprocessor)
    print(
        f"MATERIALIZING_VALIDATION bearings={','.join(validation_bearings)}",
        flush=True,
    )
    validation_data = materialize_stft_data(validation_adapter, preprocessor)
    standardizer = ChannelStandardizer.fit(train_data.tensors)

    generator = torch.Generator().manual_seed(config.arm_b_seed)
    train_loader = DataLoader(
        train_data.as_tensor_dataset(standardizer),
        batch_size=config.arm_b_batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data.as_tensor_dataset(standardizer),
        batch_size=config.arm_b_batch_size,
        shuffle=False,
    )

    model = SmallConvAutoEncoder(input_channels=2, latent_dim=config.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.arm_b_learning_rate)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.arm_b_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for (batch,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = loss_fn(output.reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.shape[0]
            total_samples += batch.shape[0]
        train_loss = total_loss / total_samples
        validation_loss = evaluate(model, validation_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"EPOCH={epoch} TRAIN_LOSS={train_loss:.8f} "
            f"VALIDATION_LOSS={validation_loss:.8f}",
            flush=True,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    checkpoint = {
        "model_state_dict": best_state,
        "model": {"input_channels": 2, "latent_dim": config.embedding_dim},
        "standardizer": {
            "mean": standardizer.mean.cpu(),
            "std": standardizer.std.cpu(),
        },
        "stft": {
            "n_fft": preprocessor.n_fft,
            "win_length": preprocessor.win_length,
            "hop_length": preprocessor.hop_length,
            "output_size": preprocessor.output_size,
        },
        "split": {
            "train": list(train_bearings),
            "validation": list(validation_bearings),
            "holdout_test": [config.split.blind_holdout],
        },
        "condition": config.target_condition,
        "seed": config.arm_b_seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "config_id": config.config_id,
        "config_sha256": file_sha256(config_path),
        "blind_holdout_read": False,
    }
    checkpoint_path = output_dir / "best_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    with (output_dir / "training_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "train_loss", "validation_loss"]
        )
        writer.writeheader()
        writer.writerows(history)

    fig, axis = plt.subplots(figsize=(8, 5))
    epochs = [int(row["epoch"]) for row in history]
    axis.plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    axis.plot(
        epochs,
        [float(row["validation_loss"]) for row in history],
        label="validation",
    )
    axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.6, label="best")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Reconstruction MSE")
    axis.set_title("Arm B target-condition AutoEncoder training")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)

    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-08-02",
            "verification_status": "UNVERIFIED",
            "version_label": "exp_result_v1",
        },
        "experiment_id": "arm_b_target_autoencoder_training_20260802",
        "type": "training",
        "status": "completed",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "condition": config.target_condition,
        "split": checkpoint["split"],
        "sample_counts": {
            "train": len(train_data.tensors),
            "validation": len(validation_data.tensors),
            "holdout_test": 0,
        },
        "blind_holdout": config.split.blind_holdout,
        "blind_holdout_read": False,
        "normalization_fitted_on": list(train_bearings),
        "normalization_mean": standardizer.mean.flatten().tolist(),
        "normalization_std": standardizer.std.flatten().tolist(),
        "latent_dim": config.embedding_dim,
        "epochs": config.arm_b_epochs,
        "batch_size": config.arm_b_batch_size,
        "learning_rate": config.arm_b_learning_rate,
        "seed": config.arm_b_seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "final_train_loss": history[-1]["train_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "duration_seconds": time.monotonic() - started,
        "outputs": [
            "best_checkpoint.pt",
            "training_curve.csv",
            "training_curve.png",
            "training_report.json",
        ],
        "interpretation_gate": {
            "passed": False,
            "reason": "Training completed; latent/state development analysis remains required.",
        },
    }
    with (output_dir / "training_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

