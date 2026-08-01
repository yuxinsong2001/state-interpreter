"""Train the first small AutoEncoder baseline on one XJTU-SY condition."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

repository_src = Path(__file__).resolve().parents[1] / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import torch
from torch import nn
from torch.utils.data import DataLoader

from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.data import ChannelStandardizer, materialize_stft_data
from state_interpreter.encoders import SmallConvAutoEncoder
from state_interpreter.preprocessing import LogSTFTPreprocessor


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for (batch,) in loader:
            output = model(batch)
            loss = loss_fn(output.reconstruction, batch)
            total_loss += float(loss) * batch.shape[0]
            total_samples += batch.shape[0]
    return total_loss / total_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", default="35Hz12kN")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    split = {
        "train": ["Bearing1_1", "Bearing1_2", "Bearing1_3"],
        "validation": ["Bearing1_4"],
        "holdout_test": ["Bearing1_5"],
    }
    preprocessor = LogSTFTPreprocessor()
    train_adapter = XJTUSYDatasetAdapter(
        args.root, conditions=[args.condition], bearings=split["train"]
    )
    validation_adapter = XJTUSYDatasetAdapter(
        args.root, conditions=[args.condition], bearings=split["validation"]
    )

    print("MATERIALIZING_TRAIN", flush=True)
    train_data = materialize_stft_data(train_adapter, preprocessor)
    print("MATERIALIZING_VALIDATION", flush=True)
    validation_data = materialize_stft_data(validation_adapter, preprocessor)
    standardizer = ChannelStandardizer.fit(train_data.tensors)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data.as_tensor_dataset(standardizer),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data.as_tensor_dataset(standardizer),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SmallConvAutoEncoder(input_channels=2, latent_dim=args.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
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
        "model": {"input_channels": 2, "latent_dim": args.latent_dim},
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
        "split": split,
        "condition": args.condition,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }
    torch.save(checkpoint, output_dir / "best_checkpoint.pt")

    with (output_dir / "training_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "train_loss", "validation_loss"]
        )
        writer.writeheader()
        writer.writerows(history)

    report = {
        "status": "completed",
        "condition": args.condition,
        "split": split,
        "sample_counts": {
            "train": len(train_data.tensors),
            "validation": len(validation_data.tensors),
            "holdout_test": 52,
        },
        "holdout_test_evaluated": False,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "normalization_fitted_on": split["train"],
        "normalization_mean": standardizer.mean.flatten().tolist(),
        "normalization_std": standardizer.std.flatten().tolist(),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "final_train_loss": history[-1]["train_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "outputs": {
            "checkpoint": str(output_dir / "best_checkpoint.pt"),
            "training_curve": str(output_dir / "training_curve.csv"),
        },
    }
    with (output_dir / "training_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

