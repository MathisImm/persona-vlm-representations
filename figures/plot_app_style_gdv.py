"""Appendix figure: GDV by prompt style vs persona (and image, rating) with all
five prompt styles pooled, per decoder layer, for both persona schemas.

Input: gdv_results.csv of the `personas_*_prototypical_all_styles` datasets
(pipeline/calculate_persona_gdv.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import GDV  # noqa: E402
from style_guide import FS_LABEL, FS_LEGEND, FS_TICK, FS_TITLE, set_ccn_style  # noqa: E402

GDV_ROOT = GDV
SCHEMAS = [("personas_big5_prototypical_all_styles", "Big Five"),
           ("personas_prototypical_all_styles", "Socio-dem.")]
# Neutral colours, distinct from the S1-S5 palette used for prompt styles;
# line and marker style follow the per-condition GDV curves (Fig. 4).
GROUPINGS = [
    ("PromptStyle", "by Prompt Style (S1–S5)", "#222222"),
    ("Persona", "by Persona (P1–P5)", "#b2182b"),
    ("Image", "by Image", "#8c8c8c"),
    ("Rating", "by Rating", "#8c6d31"),
]


def load(dataset):
    df = pd.read_csv(GDV_ROOT / dataset / "gdv_results.csv")
    df = df[(df.Layer_Type == "LLM") & (df.Space == "Original")].copy()
    df["L"] = df.Layer.str.extract(r"(\d+)$").astype(int)
    return df.sort_values("L")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=float, default=7.0)
    args = parser.parse_args()

    set_ccn_style()
    fig, axes = plt.subplots(2, 1, figsize=(args.width, 3.9 * args.width / 7.0), sharex=True, sharey=True)
    for ax, (dataset, title), tag in zip(axes, SCHEMAS, "AB"):
        df = load(dataset)
        for key, label, color in GROUPINGS:
            s = df[df.Grouping == key]
            ax.plot(s.L, s.GDV, color=color, marker="o", markersize=4, lw=1.6, label=label)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.set_ylabel(f"GDV\n{title}", fontsize=FS_LABEL, fontweight="bold")
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.text(-0.085, 1.02, tag, transform=ax.transAxes, fontsize=FS_TITLE + 1,
                fontweight="bold", ha="left", va="bottom")
    axes[1].set_xlabel("LLM layer index", fontsize=FS_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    if args.output.suffix == ".pdf":
        fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
