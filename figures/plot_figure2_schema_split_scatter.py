"""Paper Fig. 3 (file figure2_schema_split_scatter.pdf) — schema-split projection scatter, 5 rows x N layers.

Reuses the projections.npz cache produced by pipeline/compute_projections.py
for the 'combined_prototypical_all_conditions__schema_split' dataset
(baseline/D1/D2 defaults dropped pre-projection, 1000 points, MDS-only,
LLM layers only). Rows are five different colorings of the exact same
points/layout: Prompt Style, Schema, Persona, Image, Rating. Columns are
the requested LLM layers (default 2,10,22,28,36).

The Persona row reuses the identical persona-hue palette as Figure 1
(plot_figure1_combined_scatter.py) so the two figures read consistently;
the Schema row uses a mid-arc anchor color per schema from the same
palette family. Rating is ordinal, so it gets a sequential colormap plus
one horizontal colorbar (not a discrete legend) - ratings must first be
remapped from LabelEncoder's alphabetical order to the natural
Not<Slightly<Moderately<Very order (see _scatter_common.rating_rank_array).

GDV is computed live per (row, layer) panel on that panel's own cached 2D
MDS coordinates + that row's label array - there is no precomputed GDV for
the 'Schema' grouping to fall back to, and the other groupings' precomputed
values were computed on a different point set (defaults included).
"""
from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _common import DEFAULT_OUTPUT_DIR, set_ccn_style
from paths import PROJECTIONS
from style_guide import CMAP_DIST, CONDITION_STYLES, FS_LABEL, FS_TITLE
from _cache_io import (
    _layer_sort_key,
    _short_layer_label,
    coords_dict_from_cache,
    load_cache,
)
from matplotlib.lines import Line2D

from _scatter_common import (
    SCHEMA_DISPLAY,
    BIG5_PREFIX,
    DESCRIPTIVE_PREFIX,
    PROMPTSTYLE_ORDER,
    RATING_ORDER,
    RATING_SHORT,
    add_layers_arg,
    compute_layer_gdv,
    discrete_legend_handles,
    discrete_rgba_array,
    gdv_title,
    image_tab20_palette,
    layer_keys,
    n_per_schema_from_names,
    persona_display_label,
    persona_hue_palette,
    promptstyle_label,
    rating_level_colors,
    rating_rank_array,
    schema_anchor_colors,
)

DEFAULT_CACHE_PATH = PROJECTIONS / "combined_prototypical_all_conditions__schema_split" / "projections.npz"

ROWS = [
    ("promptstyle", "Prompt Style"),
    ("schema", "Schema"),
    ("persona", "Persona"),
    ("image", "Image"),
    ("rating", "Rating"),
]


def _persona_order_and_labels(persona_names, n_per_schema):
    order = (
        [f"{DESCRIPTIVE_PREFIX}{i}" for i in range(1, n_per_schema + 1) if f"{DESCRIPTIVE_PREFIX}{i}" in persona_names]
        + [f"{BIG5_PREFIX}{i}" for i in range(1, n_per_schema + 1) if f"{BIG5_PREFIX}{i}" in persona_names]
    )
    return order, {n: persona_display_label(n) for n in order}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    add_layers_arg(parser)
    parser.add_argument("--method", choices=["mds"], default="mds")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "projections_gdv")
    parser.add_argument("--fig-width", type=float, default=7.0,
                        help="Figure width in inches. 7.0 is the paper text width used "
                             "across the figures; panel and legend "
                             "sizes derive from it.")
    parser.add_argument("--fig-height", type=float, default=None,
                        help="Figure height in inches. Default keeps the panels square.")
    parser.add_argument("--point-size", type=float, default=2.5)
    parser.add_argument("--legend-fontsize", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    args = parser.parse_args()

    if not args.cache_path.exists():
        sys.exit(f"Cache not found: {args.cache_path} (run pipeline/compute_projections.py "
                  f"--persona-schemas-only first)")

    cached = load_cache(args.cache_path)
    methods_available = [str(m) for m in cached["__methods__"]]
    if args.method not in methods_available:
        sys.exit(f"Method '{args.method}' not cached (available: {methods_available}) - "
                  f"this cache only has MDS by design.")

    coords_by_layer = coords_dict_from_cache(cached, [args.method])
    requested_keys = layer_keys(args.layers)
    missing = [k for k in requested_keys if k not in coords_by_layer]
    if missing:
        sys.exit(f"Layers not found in cache: {missing} (have: "
                  f"{sorted(coords_by_layer, key=_layer_sort_key)})")

    persona_names = [str(x) for x in cached["names__persona"]]
    promptstyle_names = [str(x) for x in cached["names__promptstyle"]]
    schema_names = [str(x) for x in cached["names__schema"]]
    image_names = [str(x) for x in cached["names__image"]]
    rating_names = [str(x) for x in cached["names__rating"]]

    persona_labels = cached["labels__persona"]
    promptstyle_labels = cached["labels__promptstyle"]
    schema_labels = cached["labels__schema"]
    image_labels = cached["labels__image"]
    rating_labels = cached["labels__rating"]

    n_per_schema = n_per_schema_from_names(persona_names)
    persona_palette = persona_hue_palette(n_per_schema)
    schema_palette = schema_anchor_colors()
    promptstyle_present = [s for s in PROMPTSTYLE_ORDER if s in promptstyle_names]
    promptstyle_palette = {s: CONDITION_STYLES[s]["color"] for s in promptstyle_present}
    rating_ranks = rating_rank_array(rating_labels, rating_names)
    rating_colors = rating_level_colors(CMAP_DIST)

    image_palette = image_tab20_palette(image_names)

    set_ccn_style()
    n_layers = len(requested_keys)
    # Size off the target page width: the grid is n_layers panel columns plus
    # one legend column, panels stay square, and every band is budgeted in
    # inches so the figure lands at exactly --fig-width with nothing clipped.
    # legend_col_ratio must leave room for the widest legend text
    # ("S3 (persona after task)"), otherwise the box runs off the page edge.
    legend_col_ratio = 1.0
    side_margin = 0.40           # row y-labels
    title_band = 0.26            # per-panel "L<n> (gdv)" title + gap to the row above
    bottom_margin = 0.06
    wspace = 0.12
    n_cols = n_layers + 1
    # Width the grid actually has for axes, after margins and inter-column gaps.
    avail_w = args.fig_width - side_margin - 0.03
    unit_w = avail_w / ((n_layers + legend_col_ratio) + wspace * (n_cols - 1))
    panel_w = unit_w
    hspace = title_band / panel_w
    fig_height = args.fig_height or (
        len(ROWS) * panel_w + len(ROWS) * title_band + bottom_margin)
    fig = plt.figure(figsize=(args.fig_width, fig_height))
    gs = GridSpec(len(ROWS), n_cols,
                  width_ratios=[1.0] * n_layers + [legend_col_ratio],
                  wspace=wspace, hspace=hspace, figure=fig,
                  left=side_margin / args.fig_width, right=1.0 - 0.03 / args.fig_width,
                  top=1.0 - title_band / fig_height,
                  bottom=bottom_margin / fig_height)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_layers)] for r in range(len(ROWS))]
    legend_axes = [fig.add_subplot(gs[r, n_layers]) for r in range(len(ROWS))]
    for lax in legend_axes:
        lax.axis("off")
    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    for r, (row_key, row_label) in enumerate(ROWS):
        axes[r][0].set_ylabel(f"{r + 1}  {row_label}", fontsize=FS_LABEL, fontweight="bold")
        for c, ln in enumerate(requested_keys):
            ax = axes[r][c]
            pts = coords_by_layer[ln][args.method]
            scatter_common = dict(s=args.point_size, alpha=args.alpha, linewidths=0)

            if row_key == "promptstyle":
                colors = discrete_rgba_array(promptstyle_labels, promptstyle_names, promptstyle_palette)
                ax.scatter(pts[:, 0], pts[:, 1], c=colors, **scatter_common)
                gdv_val = compute_layer_gdv(pts, promptstyle_labels)
            elif row_key == "schema":
                colors = discrete_rgba_array(schema_labels, schema_names, schema_palette)
                ax.scatter(pts[:, 0], pts[:, 1], c=colors, **scatter_common)
                gdv_val = compute_layer_gdv(pts, schema_labels)
            elif row_key == "persona":
                colors = discrete_rgba_array(persona_labels, persona_names, persona_palette)
                ax.scatter(pts[:, 0], pts[:, 1], c=colors, **scatter_common)
                gdv_val = compute_layer_gdv(pts, persona_labels)
            elif row_key == "image":
                colors = discrete_rgba_array(image_labels, image_names, image_palette)
                ax.scatter(pts[:, 0], pts[:, 1], c=colors, **scatter_common)
                gdv_val = compute_layer_gdv(pts, image_labels)
            else:  # rating
                # Colored from the same per-level lookup the legend uses, so
                # swatches and points can never drift apart.
                colors = discrete_rgba_array(rating_ranks, RATING_ORDER, rating_colors)
                ax.scatter(pts[:, 0], pts[:, 1], c=colors, **scatter_common)
                gdv_val = compute_layer_gdv(pts, rating_ranks)

            ax.set_title(gdv_title(_short_layer_label(ln), gdv_val), fontsize=FS_TITLE, loc="center")
            if r == 0:
                # Column label (A, B, C, ...), shown once above the top row.
                # Using matplotlib's own loc="left" title (rather than a
                # manually positioned ax.text) guarantees it shares the exact
                # same baseline as the centered "L<n> (gdv)" title.
                letter = ax.set_title(string.ascii_uppercase[c], loc="left",
                                      fontsize=FS_TITLE + 1, fontweight="bold")
                # Nudge left past the panel edge so it cannot touch the
                # centred title in narrow panels (same baseline as before).
                letter.set_x(-0.1)

    # Per-row legends. Swatches scale with the legend text so the boxes stay
    # balanced at any --fig-width.
    fs = args.legend_fontsize
    ms = fs * 0.78
    legend_kw = dict(loc="center left", fontsize=fs, frameon=True, title_fontsize=fs,
                     handletextpad=0.4, labelspacing=0.3, borderpad=0.4, handlelength=1.0)

    legend_axes[0].legend(
        handles=discrete_legend_handles(
            promptstyle_present, promptstyle_palette,
            label_by_name={s: promptstyle_label(s) for s in promptstyle_present},
            markersize=ms),
        title="Style", **legend_kw)

    schema_present = [s for s in ("Descriptive", "Big-5+") if s in schema_names]
    legend_axes[1].legend(
        handles=discrete_legend_handles(schema_present, schema_palette,
                                         label_by_name=SCHEMA_DISPLAY, markersize=ms),
        title="Schema", **legend_kw)

    # Personas are labelled P1-P5 within each schema, so each schema gets an
    # invisible header row instead of a prefix on every entry.
    persona_order, persona_label_by_name = _persona_order_and_labels(persona_names, n_per_schema)
    persona_handles = []
    for key, prefix in (("Descriptive", DESCRIPTIVE_PREFIX), ("Big-5+", BIG5_PREFIX)):
        names = [n for n in persona_order if n.startswith(prefix)]
        if names:
            persona_handles.append(Line2D([], [], linestyle="none", marker="none",
                                          label=SCHEMA_DISPLAY[key]))
            persona_handles += discrete_legend_handles(
                names, persona_palette, label_by_name=persona_label_by_name, markersize=ms)
    legend_axes[2].legend(handles=persona_handles, title="Persona", **legend_kw)

    legend_axes[3].legend(
        handles=discrete_legend_handles(
            image_names, image_palette,
            label_by_name={name: f"#{i + 1}" for i, name in enumerate(image_names)},
            markersize=ms * 0.85),
        title="Image", ncol=2, columnspacing=0.8,
        **{**legend_kw, "fontsize": fs * 0.9})

    # Rating gets a discrete legend like every other row (not a colorbar), with
    # one entry per level of the full 1-5 scale. Levels the dataset never
    # produced are still listed, marked, so the scale stays readable as a whole.
    rating_present = set(rating_names)
    rating_handles = discrete_legend_handles(
        RATING_ORDER, rating_colors,
        label_by_name={
            name: (RATING_SHORT[name] if name in rating_present
                   else f"{RATING_SHORT[name]} (n/a)")
            for name in RATING_ORDER
        },
        markersize=ms)
    for handle, name in zip(rating_handles, RATING_ORDER):
        if name not in rating_present:
            handle.set_alpha(0.35)
    legend_axes[4].legend(handles=rating_handles, title="Rating", **legend_kw)

    out_dir = args.output_dir / "combined_prototypical_all_conditions__schema_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure2_schema_split_scatter"
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
