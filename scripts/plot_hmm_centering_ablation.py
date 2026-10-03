from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRECTORY = REPOSITORY_ROOT / "results" / "2026-09-16_hmm_bearing_centering_ablation"
SOURCE = RESULT_DIRECTORY / "per_step_states.csv"
OUTPUT = RESULT_DIRECTORY / "bearing1_4_hmm_centering_comparison.png"


def main() -> None:
    data = pd.read_csv(SOURCE)
    b = data.loc[data["bearing_id"] == "Bearing1_4"].copy()
    if b.empty:
        raise RuntimeError("Bearing1_4 not found")

    colors = {"raw": "#D78232", "centered": "#147A90"}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True, sharey=True)

    panels = [
        (axes[0], "raw_expected_stage", "raw_stage", colors["raw"],
         "Ohne Bearing-Zentrierung", "ρ nicht definiert · 0 Zustandswechsel"),
        (axes[1], "centered_expected_stage", "centered_stage", colors["centered"],
         "Mit Bearing-Zentrierung", "ρ = 0,962 · 2 Zustandswechsel"),
    ]
    for ax, expected, stage, color, title, subtitle in panels:
        ax.plot(
            b["normalized_lifetime"], b[expected], color=color, linewidth=2.8,
            label="Erwartete HMM-Stufe"
        )
        ax.step(
            b["normalized_lifetime"], b[stage], where="post", color="#142B3D",
            linewidth=1.2, alpha=0.75, label="Zugewiesene Stufe"
        )
        ax.axvspan(0, 14 / (len(b) - 1), color="#B8C2C8", alpha=0.22,
                   label="Kalibrierung" if ax is axes[0] else None)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.text(0.02, 0.94, subtitle, transform=ax.transAxes, va="top", fontsize=11,
                color=color, fontweight="bold")
        ax.set_xlabel("Normalisierte Laufzeit", fontsize=12)
        ax.set_yticks([0, 1, 2], ["Stufe 0", "Stufe 1", "Stufe 2"])
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.12, 2.18)
        ax.grid(True, alpha=0.22)

    axes[0].set_ylabel("Geordnete HMM-Stufe", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Bearing1_4: HMM reagiert stark auf die Repräsentationskalibrierung",
                 fontsize=17, fontweight="bold", color="#142B3D", y=0.98)
    fig.text(
        0.5, 0.008,
        "Die HMM-Stufen sind geordnete Modellzustände, keine physikalisch bestätigten Schadensklassen.",
        ha="center", fontsize=10, color="#53616B"
    )
    fig.tight_layout(rect=(0.02, 0.14, 0.98, 0.91))
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
