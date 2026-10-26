"""Paper Fig. 2 (file figure1_combined_scatter.pdf) — combined persona x promptstyle projection scatter.

Reuses the projections.npz cache produced by pipeline/compute_projections.py
for the 'combined_prototypical_all_conditions' dataset (both persona
schemas x all 5 conditions x D1/D2/baseline defaults, 1060 points). One
panel per requested LLM layer, wrapped over a --ncols grid (default: four
layers 2,10,28,36 in a 2x2, which keeps each panel large enough to read at
the 7in page width), all on the same cached 2D MDS projection.

Coloring: every point is assigned to one of up to 53 categories - the 10
real personas (5 Descriptive x 5 Big-5+) crossed with the 5 promptstyle
conditions (original_persona/P1-P4), plus the 3 defaults (baseline/D1/D2)
as their own singular categories. Persona identity is encoded as hue (cool
arc = Descriptive, warm arc = Big-5+); promptstyle is encoded as marker
shape (same shapes across every persona); defaults reuse the existing
neutral black/gray/olive styling from style_guide.CONDITION_STYLES.

GDV is computed live per panel (src.gdv.cmpGDV on that panel's own cached
2D coordinates + the 53-category group ids) and shown in the title as
"L<idx> (<gdv>)" - there is no precomputed GDV for this combined grouping
to fall back to.
"""
from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _common import DEFAULT_OUTPUT_DIR, set_ccn_style
from paths import PROJECTIONS
from style_guide import FS_LEGEND, FS_TITLE
from _cache_io import (
    _layer_sort_key,
    _short_layer_label,
    coords_dict_from_cache,
    load_cache,
)
from _scatter_common import (
    SCHEMA_DISPLAY,
    PROMPTSTYLE_MARKERS,
    PROMPTSTYLE_ORDER,
    add_layers_arg,
    build_combined_groups,
    compute_layer_gdv,
    gdv_title,
    layer_keys,
    promptstyle_label,
)

FILLED_MARKERS = set(Line2D.filled_markers)
BASELINE_PERSONA = "baseline"

DEFAULT_CACHE_PATH = PROJECTIONS / "combined_prototypical_all_conditions" / "projections.npz"


def _header(text: str) -> Line2D:
    """Invisible handle used as an in-legend section header."""
    return Line2D([], [], linestyle="none", marker="none", label=text)


def _swatch(color: str, marker: str, label: str, emphasize: bool = False,
            markersize: float = 7) -> Line2D:
    """Legend swatch. `emphasize` mirrors how the defaults are drawn in the
    panels (outlined) so the legend matches what's on the plot."""
    return Line2D([0], [0], marker=marker, linestyle="None", markerfacecolor=color,
                  markeredgecolor="black" if emphasize and marker in FILLED_MARKERS else color,
                  markeredgewidth=1.4 if emphasize else 1.0,
                  markersize=markersize, label=label)


def _build_legend(fig, group_style: dict, fontsize: float = FS_LEGEND,
                  y_center: float = 0.13) -> None:
    """One consolidated legend: four equal-length columns (Descriptive
    personas / Big-5+ personas / prompt-style markers / defaults), each
    introduced by a header row.

    Matplotlib fills legend columns by splitting the handle list into ncol
    near-equal chunks, so padding every column to the same length is what
    keeps the sections in their own columns.
    """
    # Swatches scale with the legend text so the box stays balanced at any
    # figure width.
    ms = fontsize * 0.78
    persona_swatches = {"Descriptive": {}, "Big-5+": {}}
    default_swatches = {}
    for gid in sorted(group_style):
        style = group_style[gid]
        if style["schema"] == "Defaults":
            # Baseline is the reference condition and stays the largest; D1/D2
            # sit between it and the persona swatches.
            default_swatches.setdefault(
                style["legend_label"],
                _swatch(style["color"], style["marker"], style["legend_label"],
                        emphasize=True,
                        markersize=ms * (1.45 if style["persona"] == BASELINE_PERSONA else 1.1)))
        else:
            # Hue is the persona channel, so one swatch per persona (drawn
            # with a plain circle); marker meaning lives in its own column.
            persona_swatches[style["schema"]].setdefault(
                style["legend_label"],
                _swatch(style["color"], "o", style["legend_label"], markersize=ms))

    columns = [
        [_header(f"Persona - {SCHEMA_DISPLAY['Descriptive']}")]
        + list(persona_swatches["Descriptive"].values()),
        [_header(f"Persona - {SCHEMA_DISPLAY['Big-5+']}")]
        + list(persona_swatches["Big-5+"].values()),
        [_header("Prompt style (marker)")] + [
            _swatch("#808080", PROMPTSTYLE_MARKERS[s], promptstyle_label(s), markersize=ms)
            for s in PROMPTSTYLE_ORDER
        ],
        [_header("Reference conditions")] + list(default_swatches.values()),
    ]
    height = max(len(c) for c in columns)
    handles = []
    for col in columns:
        handles.extend(col + [_header("")] * (height - len(col)))

    fig.legend(handles=handles, ncol=len(columns), loc="center",
               bbox_to_anchor=(0.5, y_center), fontsize=fontsize, frameon=True,
               handletextpad=0.5, columnspacing=1.0, labelspacing=0.35,
               borderpad=0.6, handlelength=1.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    add_layers_arg(parser, default=[2, 10, 28, 36])
    parser.add_argument("--ncols", type=int, default=2,
                        help="Panels per row; layers wrap onto as many rows as needed.")
    parser.add_argument("--method", choices=["mds"], default="mds")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "projections_gdv")
    parser.add_argument("--fig-width", type=float, default=7.0,
                        help="Figure width in inches. 7.0 is the paper text width used "
                             "across the figures; everything else "
                             "is sized relative to it.")
    parser.add_argument("--fig-height", type=float, default=None,
                        help="Figure height in inches. Default keeps the layer panels "
                             "square and reserves a fixed band for the legend.")
    parser.add_argument("--point-size", type=float, default=9.0)
    parser.add_argument("--legend-fontsize", type=float, default=6.5)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--baseline-point-scale", type=float, default=2.6,
                        help="Size multiplier for the baseline points (the no-preamble "
                             "reference condition), which are few (20) and otherwise "
                             "lost in the persona cloud.")
    parser.add_argument("--default-point-scale", type=float, default=1.5,
                        help="Size multiplier for the D1/D2 points - raised above the "
                             "persona points but kept below baseline.")
    args = parser.parse_args()

    if not args.cache_path.exists():
        sys.exit(f"Cache not found: {args.cache_path} (run pipeline/compute_projections.py first)")

    cached = load_cache(args.cache_path)
    methods_available = [str(m) for m in cached["__methods__"]]
    if args.method not in methods_available:
        sys.exit(f"Method '{args.method}' not cached (available: {methods_available})")

    coords_by_layer = coords_dict_from_cache(cached, [args.method])
    requested_keys = layer_keys(args.layers)
    missing = [k for k in requested_keys if k not in coords_by_layer]
    if missing:
        sys.exit(f"Layers not found in cache: {missing} (have: "
                  f"{sorted(coords_by_layer, key=_layer_sort_key)})")

    persona_names = [str(x) for x in cached["names__persona"]]
    promptstyle_names = [str(x) for x in cached["names__promptstyle"]]
    persona_labels = cached["labels__persona"]
    promptstyle_labels = cached["labels__promptstyle"]

    group_ids, group_style = build_combined_groups(
        persona_names, promptstyle_names, persona_labels, promptstyle_labels
    )
    valid = group_ids != -1

    # Draw the persona groups in a fixed random (not insertion) order:
    # insertion order runs Descriptive->Big-5+, persona 1..n within each, so
    # drawing in that order would make later personas/schemas systematically
    # paint over earlier ones wherever clusters overlap, biasing every panel
    # toward the same "last-drawn" categories. Shuffling once (fixed seed,
    # reused across panels) removes that systematic bias. The three defaults
    # are exempt and always drawn last, on top, so they stay legible against
    # the dense persona cloud.
    default_gids = [gid for gid, s in group_style.items() if s["schema"] == "Defaults"]
    persona_gids = [gid for gid in group_style if gid not in default_gids]
    np.random.default_rng(0).shuffle(persona_gids)
    draw_order = persona_gids + default_gids

    set_ccn_style()
    n_layers = len(requested_keys)
    # Size everything off the target page width: panels stay square, layers
    # wrap over --ncols columns, and the legend gets a band underneath, so the
    # figure lands at exactly --fig-width inches and needs no LaTeX rescaling.
    n_cols = max(1, min(args.ncols, n_layers))
    n_rows = -(-n_layers // n_cols)  # ceil
    # legend_band is sized to the 6-row legend box at the default font; the
    # panels take whatever is left so the figure has no dead space. legend_gap
    # is a thin strip of extra whitespace between the bottom panel row and the
    # legend so the two don't crowd each other.
    title_band, legend_band, wspace = 0.22, 0.88, 0.08
    legend_gap = 0.12
    row_gap = 0.10  # extra vertical breathing room between subplot rows
    side_margin = 0.03
    panel_w = (args.fig_width - side_margin) / (n_cols + wspace * (n_cols - 1))
    fig_height = args.fig_height or (
        n_rows * panel_w + n_rows * title_band + (n_rows - 1) * row_gap
        + legend_band + legend_gap)
    fig, axes_grid = plt.subplots(n_rows, n_cols,
                                  figsize=(args.fig_width, fig_height),
                                  squeeze=False)
    axes = axes_grid.flat
    # Hide any trailing cells when the layer count doesn't fill the grid.
    for spare in range(n_layers, n_rows * n_cols):
        axes[spare].set_visible(False)

    for col, ln in enumerate(requested_keys):
        ax = axes[col]
        pts = coords_by_layer[ln][args.method]
        for gid in draw_order:
            style = group_style[gid]
            mask = group_ids == gid
            if not mask.any():
                continue
            is_default = style["schema"] == "Defaults"
            if not is_default:
                scale = 1.0
            elif style["persona"] == BASELINE_PERSONA:
                scale = args.baseline_point_scale
            else:
                scale = args.default_point_scale
            scatter_kwargs = dict(
                marker=style["marker"], c=[style["color"]],
                s=args.point_size * scale,
                alpha=1.0 if is_default else args.alpha,
                zorder=3 if is_default else 2,
            )
            if style["marker"] not in FILLED_MARKERS:
                # Unfilled markers ('x', '+', ...) are drawn entirely from
                # their stroke - a zero linewidth renders them invisible.
                scatter_kwargs.update(linewidths=1.6)
            elif style.get("edge_linestyle"):
                scatter_kwargs.update(edgecolors="black", linewidths=0.9,
                                       linestyle=style["edge_linestyle"])
            elif is_default:
                scatter_kwargs.update(edgecolors="black", linewidths=0.9)
            else:
                scatter_kwargs.update(linewidths=0)
            ax.scatter(pts[mask, 0], pts[mask, 1], **scatter_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        gdv_val = compute_layer_gdv(pts[valid], group_ids[valid])
        ax.set_title(gdv_title(_short_layer_label(ln), gdv_val), fontsize=FS_TITLE, loc="center")
        # Panel label (A, B, C, ...). Using matplotlib's own loc="left" title
        # (rather than a manually positioned ax.text) guarantees it shares
        # the exact same baseline as the centered "L<n> (gdv)" title.
        ax.set_title(string.ascii_uppercase[col], loc="left",
                     fontsize=FS_TITLE + 1, fontweight="bold")

    # Margins in figure fractions derived from the inch budget above.
    fig.subplots_adjust(left=side_margin / args.fig_width,
                        right=1.0 - side_margin / args.fig_width,
                        wspace=wspace, hspace=(title_band + row_gap) / panel_w,
                        bottom=(legend_band + legend_gap) / fig_height,
                        top=1.0 - title_band / fig_height)
    _build_legend(fig, group_style, fontsize=args.legend_fontsize,
                  y_center=(legend_band * 0.5) / fig_height)

    out_dir = args.output_dir / "combined_prototypical_all_conditions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure1_combined_scatter"
    # Explicit full-figure bbox: style_guide.set_ccn_style sets
    # rcParams["savefig.bbox"]="tight", which would crop to the ink and land
    # the file at some arbitrary width. Passing the figure's own bbox keeps
    # the output exactly --fig-width inches, so it drops into LaTeX at scale 1.
    plt.savefig(f"{out_path}.png", dpi=300, bbox_inches=fig.bbox_inches)
    plt.savefig(f"{out_path}.pdf", bbox_inches=fig.bbox_inches)
    plt.close()
    print(f"Saved {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
