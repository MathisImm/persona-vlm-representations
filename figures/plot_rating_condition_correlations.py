#!/usr/bin/env python3
"""
Publication Plot: Rating agreement across all prompt conditions (100 images)

Layout:
- Panel a (Top): 3 bubble scatter plots of per-image ratings, comparing the
  no-persona default conditions pairwise: Base vs D1, Base vs D2,
  D1 vs D2. Marker area encodes the number of images sharing that
  (rating, rating) cell (ratings are discrete 1-5, so raw points overlap).
- Panel b (Bottom, full width): Pearson-r rating-correlation heatmap across
  all 28 rating-generating conditions - Base, D1, D2, then each of the
  5 personas x 5 prompt-styles (S1=original persona ... S5=P4) - ordered
  persona-major / style-minor as requested.

Reuses the loader/ordering helpers from plot_rating_analysis_publication.py
(same ratings/ folders, default schema =
personas_big5_prototypical for the main figure; personas_prototypical is the
descriptive-schema supplementary version).
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.append(str(Path(__file__).parent))
from plot_rating_analysis_publication import (
    load_uniform_dataset, filter_image_ids, PROMPTSTYLE_ORDER, UNIFORM_RATINGS_ROOT,
    _persona_order,
)
from paths import OUTPUT
try:
    from style_guide import (set_ccn_style, CONDITION_STYLES, CMAP_CORR,
                             FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT)
except ImportError:
    def set_ccn_style():
        sns.set_theme(style="whitegrid")
    CONDITION_STYLES = {}
    CMAP_CORR = 'YlGnBu'
    FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = 10, 9, 8, 7

def _fmt_corr(v, decimals=2):
    """APA-style correlation formatting: '1.' for unity, '.xx' (no leading zero) otherwise."""
    if pd.isna(v):
        return ""
    r = round(v, decimals)
    if r == 1.0:
        return "1."
    if r == -1.0:
        return "-1."
    s = f"{v:.{decimals}f}"
    if s.startswith("-0."):
        return "-" + s[2:]
    if s.startswith("0."):
        return s[1:]
    return s


DEFAULT_CONDITIONS = ['Base', 'D1', 'D2']
# S1..S5 promptstyle display convention shared with Figures 1/2/4
# (figures/_scatter_common.py: PROMPTSTYLE_DISPLAY / _DESCR /
# promptstyle_label) - kept as a local copy since persona_paper's helpers
# pull in src.gdv and a projections cache this figure doesn't need. The
# S1-S5 -> description mapping itself is spelled out in Figure 4's legend,
# not repeated here.
STYLE_CODE = {s: f'S{i + 1}' for i, s in enumerate(PROMPTSTYLE_ORDER)}
SCATTER_PAIRS = [('Base', 'D1'), ('Base', 'D2'), ('D1', 'D2')]


def build_condition_table(df):
    """Wide image_id x condition rating table.

    Columns ordered: Base, D1, D2, then per persona (numeric order) the
    5 prompt-style columns S1..S5 (S1=original_persona ... S5=P4).
    Returns (wide_df, label_rows) where label_rows[i] = (top, bottom) for
    the two-line tick label of column i.
    """
    persona_display, persona_ids = _persona_order(df)

    cols = {}
    label_rows = []
    for name, style in zip(DEFAULT_CONDITIONS, ['original_baseline', 'D1', 'D2']):
        cols[name] = df[df['promptstyle'] == style].set_index('image_id')['rating']
        label_rows.append(('', name))

    for p_lab, p_id in zip(persona_display, persona_ids):
        for style in PROMPTSTYLE_ORDER:
            code = STYLE_CODE[style]
            colname = f'{p_lab}__{code}'
            sub = df[(df['persona_id'] == p_id) & (df['promptstyle'] == style)]
            cols[colname] = sub.set_index('image_id')['rating']
            label_rows.append((p_lab, code))

    col_order = list(cols.keys())
    wide = pd.DataFrame(cols)[col_order]
    return wide, label_rows


def _bubble_sizes(counts, min_s=20, max_s=400):
    """Map raw cell counts to marker areas, shared across panels."""
    lo, hi = counts.min(), counts.max()
    if hi == lo:
        return np.full_like(counts, (min_s + max_s) / 2, dtype=float)
    return min_s + (counts - lo) / (hi - lo) * (max_s - min_s)


def plot_bubble_scatter(ax, wide, xcol, ycol, color, global_max_count):
    sub = wide[[xcol, ycol]].dropna()
    cross = pd.crosstab(sub[xcol], sub[ycol])

    xs, ys, counts = [], [], []
    for x_val in cross.index:
        for y_val in cross.columns:
            c = cross.loc[x_val, y_val]
            if c > 0:
                xs.append(x_val); ys.append(y_val); counts.append(c)
    counts = np.array(counts, dtype=float)
    sizes = _bubble_sizes(counts, min_s=20, max_s=20 + (380 * counts.max() / global_max_count))

    ax.plot([0.5, 5.5], [0.5, 5.5], ls='--', color='gray', lw=1.0, zorder=1)
    ax.scatter(xs, ys, s=sizes, color=color, alpha=0.8,
               edgecolor='white', linewidth=0.7, zorder=3)

    ax.set_title(f"{xcol} vs. {ycol}", fontsize=FS_TITLE)
    ax.set_xlabel(f"{xcol} rating", fontsize=FS_LABEL)
    ax.set_ylabel(f"{ycol} rating", fontsize=FS_LABEL)
    ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_aspect('equal')
    ax.xaxis.grid(False); ax.yaxis.grid(False)


def plot_condition_heatmap(ax, wide, label_rows, annot_size=8.0):
    corr = wide.corr(method='pearson')
    n = len(corr)
    vmin = min(0.0, float(np.nanmin(corr.values)))
    annot_labels = corr.map(_fmt_corr)

    sns.heatmap(corr, cmap=CMAP_CORR, vmin=vmin, vmax=1.0, square=True,
                cbar_kws={'label': 'Pearson $r$', 'shrink': 0.8,
                         'fraction': 0.035, 'pad': 0.015},
                xticklabels=False, yticklabels=False, linewidths=0,
                annot=annot_labels, fmt='s', annot_kws={'size': annot_size}, ax=ax)
    ax.grid(False)

    # Defaults (Base/D1/D2) have no persona-number "top" line, so give
    # them a single-line label - it then sits centered at the tick height
    # instead of being pushed off-center by a blank first line.
    tick_labels = [f"{top}\n{bot}" if top else bot for top, bot in label_rows]
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(tick_labels, fontsize=FS_TICK - 1, rotation=0)
    ax.set_yticklabels(tick_labels, fontsize=FS_TICK - 1, rotation=0)

    n_defaults = len(DEFAULT_CONDITIONS)
    # "Base" is wider than a cell, so the reference labels on the x axis are
    # set vertically; the two-line persona/style labels stay horizontal.
    for lbl in ax.get_xticklabels()[:n_defaults]:
        lbl.set_rotation(90)
    ax.axhline(n_defaults, color='black', lw=1.3)
    ax.axvline(n_defaults, color='black', lw=1.3)
    n_styles = len(PROMPTSTYLE_ORDER)
    boundary = n_defaults + n_styles
    while boundary < n:
        ax.axhline(boundary, color='white', lw=0.9)
        ax.axvline(boundary, color='white', lw=0.9)
        boundary += n_styles


def plot_figure(df, args):
    set_ccn_style()
    wide, label_rows = build_condition_table(df)

    fig = plt.figure(figsize=args.figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=args.height_ratios,
                           hspace=args.hspace)
    gs_top = gs[0].subgridspec(1, 3, wspace=args.wspace)

    # Precompute the largest single-cell count across all 3 scatter panels
    # so bubble sizes are comparable between panels.
    global_max_count = 1
    for xcol, ycol in SCATTER_PAIRS:
        sub = wide[[xcol, ycol]].dropna()
        cross = pd.crosstab(sub[xcol], sub[ycol])
        global_max_count = max(global_max_count, int(cross.values.max()))

    # The scatter panels compare the no-persona defaults, so they take Base's colour.
    scatter_color = CONDITION_STYLES.get('original_baseline', {}).get('color', '#000000')
    for i, (xcol, ycol) in enumerate(SCATTER_PAIRS):
        ax = fig.add_subplot(gs_top[0, i])
        plot_bubble_scatter(ax, wide, xcol, ycol, scatter_color, global_max_count)

    ax_heat = fig.add_subplot(gs[1])
    plot_condition_heatmap(ax_heat, wide, label_rows, args.annot_size)

    plt.subplots_adjust(left=args.left, right=args.right, top=args.top,
                        bottom=args.bottom)

    # Panel labels use a shared figure-fraction x via fig.text rather than
    # each axes' own loc="left" title: the scatter panels' aspect="equal"
    # re-centers their actual drawn box within the gridspec cell, which
    # would otherwise throw off x-alignment between "A" and "B" - the
    # gridspec cells themselves aren't affected by that per-axes shift.
    top_row_box = gs[0].get_position(fig)
    bottom_row_box = gs[1].get_position(fig)
    fig.text(args.label_x, top_row_box.y1 + 0.003, "A", fontsize=FS_TITLE + 1,
             fontweight='bold', va='bottom', ha='left')
    fig.text(args.label_x, bottom_row_box.y1 + 0.003, "B", fontsize=FS_TITLE + 1,
             fontweight='bold', va='bottom', ha='left')

    plt.savefig(f"{args.output_path}.png", dpi=300)
    plt.savefig(f"{args.output_path}.pdf")
    print(f"Saved rating-condition-correlations plot to {args.output_path}.png/pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Rating agreement across all prompt conditions (100 images)')
    parser.add_argument('--schema', type=str, default='personas_big5_prototypical',
                        help='Schema folder name in ratings/ '
                             '(default: personas_big5_prototypical)')
    parser.add_argument('--ratings-root', type=Path, default=UNIFORM_RATINGS_ROOT)
    parser.add_argument('--image-ids-file', type=Path, default=None,
                        help='Restrict all analyses to these image ids (one per line).')
    parser.add_argument('--output-path', type=Path,
                        default=OUTPUT / "figures")

    # Layout parameters
    parser.add_argument('--label-x', type=float, default=0.035,
                        help='Shared figure-fraction x for the A/B panel labels')
    parser.add_argument('--wspace', type=float, default=0.32)
    parser.add_argument('--hspace', type=float, default=0.14)
    parser.add_argument('--height-ratios', type=float, nargs=2, default=[0.32, 2.2])
    parser.add_argument('--annot-size', type=float, default=8.0,
                        help='Font size (pt) of the r values in the heatmap cells.')
    parser.add_argument('--figsize', type=float, nargs=2, default=[7, 7.6],
                        help='Paper figures use a 7-inch standard width '
                             '(see age_vc.py FIG_WIDTH / '
                             'plot_persona_agreement_master.py)')
    parser.add_argument('--left', type=float, default=0.06)
    parser.add_argument('--right', type=float, default=0.98)
    parser.add_argument('--top', type=float, default=0.96)
    parser.add_argument('--bottom', type=float, default=0.03)

    args = parser.parse_args()

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_path = out_dir / f"figure5_rating_condition_correlations_{args.schema}"

    df = load_uniform_dataset(args.schema, ratings_root=args.ratings_root,
                              include_defaults=True)
    df = filter_image_ids(df, args.image_ids_file)
    plot_figure(df, args)


if __name__ == "__main__":
    main()
