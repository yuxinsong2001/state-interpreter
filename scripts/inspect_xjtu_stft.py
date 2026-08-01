"""Create early/middle/late log-STFT plots for one XJTU-SY bearing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repository_src = Path(__file__).resolve().parents[1] / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import torch
from PIL import Image, ImageDraw, ImageOps

from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.preprocessing import LogSTFTPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--condition", default="35Hz12kN")
    parser.add_argument("--bearing", default="Bearing1_1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = XJTUSYDatasetAdapter(
        args.root,
        conditions=[args.condition],
        bearings=[args.bearing],
    )
    measurements = list(adapter.iter_measurements())
    selected_indices = [0, len(measurements) // 2, len(measurements) - 1]
    selected = [measurements[index] for index in selected_indices]
    preprocessor = LogSTFTPreprocessor()
    transformed = [preprocessor(item.vibration) for item in selected]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    global_min = min(float(item.min()) for item in transformed)
    global_max = max(float(item.max()) for item in transformed)
    stage_names = ["Early", "Middle", "Late"]
    channel_names = ["Horizontal", "Vertical"]
    panel_size = 256
    title_height = 32
    margin = 16
    canvas = Image.new(
        "RGB",
        (
            margin * 4 + panel_size * 3,
            margin * 3 + title_height * 3 + panel_size * 2,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, margin),
        f"{args.condition}/{args.bearing}: 32x32 log-STFT "
        f"(global range {global_min:.3f} to {global_max:.3f})",
        fill="black",
    )
    scale = max(global_max - global_min, 1e-12)
    for column, (stage, measurement, tensor) in enumerate(
        zip(stage_names, selected, transformed, strict=True)
    ):
        for row, channel_name in enumerate(channel_names):
            normalized = (
                (tensor[row].detach().cpu() - global_min) / scale * 255.0
            )
            grayscale = Image.new("L", (32, 32))
            grayscale.putdata(
                normalized.clamp(0, 255).to(torch.uint8).flatten().tolist()
            )
            heatmap = ImageOps.colorize(
                grayscale,
                black="#000004",
                mid="#B63679",
                white="#FCFDBF",
            ).resize((panel_size, panel_size), Image.Resampling.NEAREST)
            x = margin + column * (panel_size + margin)
            y = margin * 2 + title_height + row * (panel_size + title_height)
            draw.text(
                (x, y - title_height + 8),
                f"{stage} m={measurement.metadata['measurement_number']} | "
                f"{channel_name}",
                fill="black",
            )
            canvas.paste(heatmap, (x, y))
    canvas.save(output_path)

    report = {
        "episode_id": selected[0].episode_id,
        "measurement_numbers": [
            item.metadata["measurement_number"] for item in selected
        ],
        "output_shapes": [list(item.shape) for item in transformed],
        "all_finite": [bool(torch.isfinite(item).all()) for item in transformed],
        "minimums": [float(item.min()) for item in transformed],
        "maximums": [float(item.max()) for item in transformed],
        "means": [float(item.mean()) for item in transformed],
        "stft_config": {
            "n_fft": preprocessor.n_fft,
            "win_length": preprocessor.win_length,
            "hop_length": preprocessor.hop_length,
            "output_size": list(preprocessor.output_size),
            "magnitude_transform": "log1p",
            "normalization": None,
        },
        "figure": str(output_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
