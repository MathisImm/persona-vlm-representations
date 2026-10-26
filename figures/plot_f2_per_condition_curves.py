"""Paper Fig. 4 and B1 — GDV-by-layer curves interleaved with per-promptstyle projection scatter.

Two flavors selected via --persona-set:
  big5         → main figure: personas_big5_prototypical{,_P1..P4}
  descriptive  → appendix figure: personas_prototypical{,_P1..P4}

Three GDV groupings (Image, Persona, Rating; CI dropped), each followed
immediately by a row of small MDS projection panels — one per promptstyle
(Original, P1-P4) — colored by that grouping's own category, with the live
GDV value in each panel's title (see _scatter_common.gdv_title). GDV curves
show LLM layers only (no ViT segment), matching the LLM-only projection
layer.

The projection panels reuse per-promptstyle projections.npz caches, one
per condition, written by pipeline/compute_projections.py, e.g.:
    python pipeline/compute_projections.py \
        --dataset personas_big5_prototypical_all_conditions_with_defaults \
        --colorings promptstyle --promptstyle-only P1
for each of the 5 promptstyles x 2 persona-sets (10 caches total; see
pipeline/run_gdv_and_projections.sh). Each promptstyle gets its own independently-fit
2D MDS (not a shared/joint projection sliced per condition), so every panel
spends its full 2D budget on that condition's own persona/image/rating
structure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from _common import (
    CONDITION_STYLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_ROOT,
    draw_gdv_panel,
    load_curves,
    set_ccn_style,
)
from paths import PROJECTIONS
from style_guide import CMAP_DIST, FS_LABEL, FS_TITLE
from _cache_io import coords_dict_from_cache, load_cache
from _scatter_common import (
    PROMPTSTYLE_ORDER,
    RATING_ORDER,
    RATING_SHORT,
    compute_layer_gdv,
    discrete_legend_handles,
    discrete_rgba_array,
    gdv_title,
    image_tab20_palette,
    n_per_schema_from_names,
    persona_display_label,
    persona_hue_palette,
    promptstyle_label,
    rating_level_colors,
    rating_rank_array,
)

PERSONA_SETS = {
    "big5": {
        "base": "personas_big5_prototypical",
        "title_human": "Big-5+ prototypical personas",
        "filename": "figure3_main_big5_per_condition",
    },
    "descriptive": {
        "base": "personas_prototypical",
        "title_human": "Descriptive prototypical personas (appendix)",
        "filename": "figure3_appendix_descriptive_per_condition",
    },
}

# 5 curves: original_persona is the baseline framing, then P1..P4. No D1/D2,
# no original_baseline (the no-persona conditions have a single persona and
# would collapse the by-Persona panel).
CONDITIONS = ["original_persona", "P1", "P2", "P3", "P4"]

GROUPINGS = ("image", "persona", "rating")
GROUPING_ROW_LABELS = {"image": "Image", "persona": "Persona", "rating": "Rating"}
# Row-pair labels for text cross-references (GDV curve row / projection row).
ROW_LETTER = {"image": "A", "persona": "B", "rating": "C"}

DEFAULT_CACHE_DIR = PROJECTIONS


def load_projection_panels(
    cache_dir: Path, projection_dataset: str, conditions: list, layer: int,
) -> dict:
    """Load each condition's independently-projected MDS cache.

    Each cache under `<cache_dir>/<projection_dataset>__style_<cond>/projections.npz`
    was produced with `--promptstyle-only <cond>`, so its 2D layout only
    reflects that one condition's own points (not a shared/joint space).

    Returns: dict condition -> {coords, image_names, image_labels,
        persona_names, persona_labels, rating_names, rating_labels}.
    """
    panels = {}
    layer_key = f"layer_{layer}"
    for cond in conditions:
        cache_path = cache_dir / f"{projection_dataset}__style_{cond}" / "projections.npz"
        if not cache_path.exists():
            print(f"load_projection_panels: missing cache for '{cond}': {cache_path}")
            continue
        cached = load_cache(cache_path)
        coords_by_layer = coords_dict_from_cache(cached, ["mds"])
        if layer_key not in coords_by_layer:
            print(f"load_projection_panels: '{layer_key}' missing in {cache_path}")
            continue
        panels[cond] = {
            "coords": coords_by_layer[layer_key]["mds"],
            "image_names": [str(x) for x in cached["names__image"]],
            "image_labels": cached["labels__image"],
            "persona_names": [str(x) for x in cached["names__persona"]],
            "persona_labels": cached["labels__persona"],
            "rating_names": [str(x) for x in cached["names__rating"]],
            "rating_labels": cached["labels__rating"],
        }
    return panels


def _grouping_colors_and_gdv(grouping, panel, persona_palette, image_palette, rating_colors):
    """Return (rgba colors array, live GDV value) for one (grouping,
    condition-panel) combination."""
    if grouping == "persona":
        colors = discrete_rgba_array(panel["persona_labels"], panel["persona_names"], persona_palette)
        gdv_val = compute_layer_gdv(panel["coords"], panel["persona_labels"])
    elif grouping == "image":
        colors = discrete_rgba_array(panel["image_labels"], panel["image_names"], image_palette)
        gdv_val = compute_layer_gdv(panel["coords"], panel["image_labels"])
    else:  # rating
        rating_ranks = rating_rank_array(panel["rating_labels"], panel["rating_names"])
        colors = discrete_rgba_array(rating_ranks, RATING_ORDER, rating_colors)
        gdv_val = compute_layer_gdv(panel["coords"], rating_ranks)
    return colors, gdv_val


def plot_interleaved_gdv_projection(
    gdv_df,
    projection_panels: dict,
    output_path: Path,
    style_map: dict,
    groupings=GROUPINGS,
    layer: int = 36,
    fig_width: float = 7.0,
    point_size: float = 6.0,
    alpha: float = 0.75,
    gdv_to_proj_gap: float = 0.85,
    proj_to_gdv_gap: float = 0.16,
    label_x_offset_1: float = 0.0,
    label_x_offset_2: float = 0.0,
    label_y_offset_1: float = 0.008,
    label_y_offset_2: float = 0.008,
) -> None:
    """3 GDV-curve rows (LLM-layers-only), each immediately followed by a row
    of per-promptstyle MDS projection panels colored by that grouping's own
    category. Sized to land at exactly `fig_width` inches (the paper text
    width standard used across the figures), like Figure 1/2's scatter panels.

    Args:
        gdv_to_proj_gap: GridSpec `hspace` between a grouping's own GDV curve
            row and its projection row (larger = more room between the GDV
            row's xlabel and the projection row's per-condition titles).
        proj_to_gdv_gap: extra inches of blank space between one grouping's
            projection row and the next grouping's GDV row. Kept separate
            from `gdv_to_proj_gap` since GridSpec's `hspace` is a single
            fraction applied to every row boundary - it can't give these two
            gaps different sizes on its own.
        label_x_offset_1: extra inches added to the shared x position (see
            below) of the "_1" corner labels (A1/B1/C1).
        label_x_offset_2: extra inches added to the shared x position (see
            below) of the "_2" corner labels (A2/B2/C2). All six labels share
            an x position measured off A1's own y-axis label; these offsets
            nudge the "_1" and "_2" groups independently from there.
        label_y_offset_1: inches of clearance between the GDV row's y-axis
            label and its "_1" corner label (A1/B1/C1) above it.
        label_y_offset_2: inches of clearance between the projection row's
            y-axis label and its "_2" corner label (A2/B2/C2) above it.
    """
    set_ccn_style()
    conditions_present = [c for c in PROMPTSTYLE_ORDER if c in projection_panels]
    n_cond = len(conditions_present)
    n_groupings = len(groupings)

    # Palettes shared across all condition panels for a grouping, built from
    # the union of names actually present (persona/image ids are stable
    # identifiers across promptstyles, so palettes should not be refit per
    # condition).
    all_persona_names = sorted(
        {n for p in projection_panels.values() for n in p["persona_names"]},
        key=lambda n: int(n.rsplit("_", 1)[-1]),
    )
    n_per_schema = n_per_schema_from_names(all_persona_names)
    persona_palette = persona_hue_palette(n_per_schema)
    all_image_names = sorted({n for p in projection_panels.values() for n in p["image_names"]})
    image_palette = image_tab20_palette(all_image_names)
    rating_colors = rating_level_colors(CMAP_DIST)
    rating_present = {r for p in projection_panels.values() for r in p["rating_names"]}

    # Size off the target page width: n_cond square panel columns + one
    # narrower legend column; GDV rows span the full width above each
    # projection row. Each grouping gets its own independent 2-row GridSpec
    # band (gdv row <-> proj row uses intra_hspace, tuned so the GDV row's
    # xlabel and the proj row's titles don't collide), stacked with an
    # explicit, much smaller inter_group_gap between groupings - a single
    # GridSpec-wide hspace can't give these two gaps different sizes since
    # it's one fraction applied to every row boundary.
    legend_col_ratio = 1.2
    side_margin, title_band, top_band = 0.95, 0.20, 0.42
    gdv_row_h = 1.05
    panel_w = (fig_width - side_margin) / (n_cond + legend_col_ratio)
    band_h = gdv_row_h + panel_w + title_band
    fig_height = top_band + n_groupings * band_h + (n_groupings - 1) * proj_to_gdv_gap + 0.02

    fig = plt.figure(figsize=(fig_width, fig_height))

    legend_conditions = [c for c in CONDITIONS if c in set(gdv_df["condition"].unique())]
    top_legend_handles = [
        Line2D(
            [0], [0], color=style_map.get(c, {}).get("color", "gray"),
            linestyle=style_map.get(c, {}).get("linestyle", "-"),
            marker=style_map.get(c, {}).get("marker", "o"), markersize=4, lw=1.6,
        )
        for c in legend_conditions
    ]
    top_legend_labels = [promptstyle_label(c) for c in legend_conditions]

    label_x_ref = None
    for gi, grouping in enumerate(groupings):
        band_top_in = top_band + gi * (band_h + proj_to_gdv_gap)
        band_bottom_in = band_top_in + band_h
        gs = GridSpec(
            2, n_cond + 1,
            width_ratios=[1.0] * n_cond + [legend_col_ratio],
            height_ratios=[gdv_row_h, panel_w + title_band],
            wspace=0.14, hspace=gdv_to_proj_gap, figure=fig,
            left=side_margin / fig_width, right=0.995,
            top=1.0 - band_top_in / fig_height, bottom=1.0 - band_bottom_in / fig_height,
        )

        letter = ROW_LETTER.get(grouping, chr(65 + gi))

        gdv_ax = fig.add_subplot(gs[0, :])
        draw_gdv_panel(gdv_ax, gdv_df, grouping, style_map)
        gdv_ax.set_xlabel("LLM layer index", fontsize=FS_LABEL)
        gdv_ax.set_ylabel(
            gdv_ax.get_ylabel().replace(" ", "\n", 1), fontsize=FS_LABEL, fontweight="bold",
        )

        proj_axes = [fig.add_subplot(gs[1, c]) for c in range(n_cond)]
        legend_ax = fig.add_subplot(gs[1, n_cond])
        legend_ax.axis("off")
        proj_axes[0].set_ylabel(
            GROUPING_ROW_LABELS.get(grouping, grouping), fontsize=FS_LABEL, fontweight="bold",
        )
        for ax in proj_axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # Letters sit directly above each row's own (rotated) y-axis title,
        # but all six share the x position measured off A1's label (the
        # widest - the two-line "GDV\nby ..." label), so they form a single
        # clean left-aligned column instead of drifting with each row's own
        # (possibly narrower) label width.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if label_x_ref is None:
            label_x_ref = gdv_ax.yaxis.label.get_window_extent(
                renderer=renderer).transformed(fig.transFigure.inverted()).x0
        for ax, num, x_offset, y_offset in (
            (gdv_ax, "1", label_x_offset_1, label_y_offset_1),
            (proj_axes[0], "2", label_x_offset_2, label_y_offset_2),
        ):
            label_bbox_fig = ax.yaxis.label.get_window_extent(
                renderer=renderer).transformed(fig.transFigure.inverted())
            fig.text(
                label_x_ref + x_offset, label_bbox_fig.y1 + y_offset, f"{letter}{num}",
                fontsize=FS_TITLE + 1, fontweight="bold", ha="left", va="bottom",
            )

        for c, cond in enumerate(conditions_present):
            ax = proj_axes[c]
            panel = projection_panels[cond]
            colors, gdv_val = _grouping_colors_and_gdv(
                grouping, panel, persona_palette, image_palette, rating_colors,
            )
            pts = panel["coords"]
            ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=point_size, alpha=alpha, linewidths=0)
            ax.set_title(
                gdv_title(promptstyle_label(cond, with_descr=False), gdv_val), fontsize=FS_TITLE,
                color=style_map.get(cond, {}).get("color", "black"),
            )

        # Per-grouping legend in the rightmost column.
        if grouping == "persona":
            legend_ax.legend(
                handles=discrete_legend_handles(
                    all_persona_names, persona_palette,
                    label_by_name={n: persona_display_label(n) for n in all_persona_names}),
                loc="center left", fontsize=6, frameon=True, title="Persona", title_fontsize=6)
        elif grouping == "image":
            legend_ax.legend(
                handles=discrete_legend_handles(
                    all_image_names, image_palette,
                    label_by_name={n: f"#{i + 1}" for i, n in enumerate(all_image_names)}, markersize=3.5),
                loc="center left", fontsize=4.5, frameon=True, title="Image", title_fontsize=5.5, ncol=2,
                columnspacing=0.6, handletextpad=0.3)
        else:  # rating
            rating_handles = discrete_legend_handles(
                RATING_ORDER, rating_colors,
                label_by_name={
                    n: (RATING_SHORT[n] if n in rating_present else f"{RATING_SHORT[n]} (n/a)")
                    for n in RATING_ORDER
                })
            for handle, name in zip(rating_handles, RATING_ORDER):
                if name not in rating_present:
                    handle.set_alpha(0.35)
            legend_ax.legend(handles=rating_handles, loc="center left", fontsize=5.5,
                              frameon=True, title="Rating", title_fontsize=6)

    fig.legend(
        top_legend_handles, top_legend_labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.0), ncol=3,
        fontsize=6.5, frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_str = str(output_path)
    plt.savefig(f"{out_str}.png", dpi=300, bbox_inches=fig.bbox_inches)
    plt.savefig(f"{out_str}.pdf", bbox_inches=fig.bbox_inches)
    plt.close()
    print(f"Saved {out_str}.png and {out_str}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persona-set",
        choices=sorted(PERSONA_SETS.keys()),
        required=True,
        help="Which persona-set to plot (selects main vs appendix figure).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing per-dataset GDV result subfolders.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory holding the per-promptstyle projections.npz caches "
             "(see module docstring for how to generate them).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the figure (PDF + PNG).",
    )
    parser.add_argument(
        "--layer", type=int, default=36,
        help="LLM layer index used for the projection panels (default: last LLM layer).",
    )
    parser.add_argument(
        "--fig-width", type=float, default=7.0,
        help="Figure width in inches. 7.0 is the paper text width used across "
             "figures (see plot_figure1_combined_scatter.py); the figure is "
             "sized to land at exactly this width.",
    )
    parser.add_argument(
        "--gdv-to-proj-gap", type=float, default=0.75,
        help="GridSpec hspace between a grouping's GDV curve row and its own "
             "projection row (larger = more room between the GDV row's xlabel "
             "and the projection row's per-condition titles).",
    )
    parser.add_argument(
        "--proj-to-gdv-gap", type=float, default=0.2,
        help="Inches of blank space between one grouping's projection row and "
             "the next grouping's GDV row.",
    )
    parser.add_argument(
        "--label-x-offset-1", type=float, default=-0.01,
        help="Extra inches added to the shared x position of the '_1' corner "
             "labels (A1/B1/C1). All six corner labels share a base position "
             "measured off A1's own y-axis label.",
    )
    parser.add_argument(
        "--label-x-offset-2", type=float, default=0.05,
        help="Extra inches added to the shared x position of the '_2' corner "
             "labels (A2/B2/C2). All six corner labels share a base position "
             "measured off A1's own y-axis label.",
    )
    parser.add_argument(
        "--label-y-offset-1", type=float, default=0.001,
        help="Inches of clearance between the GDV row's y-axis label and its "
             "'_1' corner label (A1/B1/C1) above it.",
    )
    parser.add_argument(
        "--label-y-offset-2", type=float, default=0.04,
        help="Inches of clearance between the projection row's y-axis label "
             "and its '_2' corner label (A2/B2/C2) above it.",
    )
    args = parser.parse_args()

    spec = PERSONA_SETS[args.persona_set]
    base = spec["base"]

    # condition_id "original_persona" uses the *_no_baseline run so the persona
    # set matches the suffixed _P1..P4 datasets (5 personas each, no baseline
    # subdir). P1..P4 live in folders with the matching suffix.
    sources = []
    for cond in CONDITIONS:
        if cond == "original_persona":
            ds_name = f"{base}_no_baseline"
        else:
            ds_name = f"{base}_{cond}"
        sources.append((cond, args.results_root / ds_name))

    df = load_curves(sources)

    projection_dataset = f"{base}_all_conditions_with_defaults"
    projection_panels = load_projection_panels(
        args.cache_dir, projection_dataset, CONDITIONS, args.layer,
    )

    plot_interleaved_gdv_projection(
        df,
        projection_panels,
        args.output_dir / spec["filename"],
        style_map=CONDITION_STYLES,
        groupings=GROUPINGS,
        layer=args.layer,
        fig_width=args.fig_width,
        gdv_to_proj_gap=args.gdv_to_proj_gap,
        proj_to_gdv_gap=args.proj_to_gdv_gap,
        label_x_offset_1=args.label_x_offset_1,
        label_x_offset_2=args.label_x_offset_2,
        label_y_offset_1=args.label_y_offset_1,
        label_y_offset_2=args.label_y_offset_2,
    )


if __name__ == "__main__":
    main()
