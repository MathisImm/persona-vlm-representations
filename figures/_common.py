"""Shared layout helpers for the persona-paper figure family.

All paper-figure scripts in this folder use the same vertical multi-panel
layout (one panel per GDV grouping, several curves per panel) and the same
output convention. Centralize that here so each figure script only has to
declare its data sources and curve styles.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Make the repo root (paths.py) importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import GDV, OUTPUT  # noqa: E402
from style_guide import (  # noqa: E402
    CONDITION_STYLES,
    FS_LABEL,
    FS_TICK,
    set_ccn_style,
)

DEFAULT_RESULTS_ROOT = GDV
DEFAULT_OUTPUT_DIR = OUTPUT / "figures"

GROUPING_LABELS = {
    "image": "by Image",
    "persona": "by Persona",
    "promptstyle": "by Prompt Style",
    "rating": "by Rating",
    "ci": "by CI",
}

# GDV loaders, copied from the original plot_persona_clustering.py.
def load_long_format_gdv(results_dir: Path) -> pd.DataFrame:
    """
    Load the decoder-layer, original-space rows of a long-format gdv_results.csv
    and pivot them into the wide format expected by the plotting functions.
    """
    gdv_path = results_dir / "gdv_results.csv"

    if not gdv_path.exists():
        print(f"   Warning: GDV results not found at {gdv_path}")
        return pd.DataFrame()

    df = pd.read_csv(gdv_path)
    # The figures plot decoder layers in the original space (the only rows
    # calculate_persona_gdv.py writes).
    df = df[(df['Layer_Type'] == 'LLM') & (df['Space'] == 'Original')]

    # Drop rows with NaN GDV to avoid issues during pivoting
    n_before = len(df)
    df = df.dropna(subset=['GDV'])
    if len(df) < n_before:
        print(f"   Note: Dropped {n_before - len(df)} rows with missing GDV values")
        
    if len(df) == 0:
        print("   Warning: No valid GDV values found in CSVs")
        return pd.DataFrame()

    # Standardize spaces and groupings for column names
    df['Space'] = df['Space'].str.lower()
    df['Grouping'] = df['Grouping'].str.lower()
    # Map CI_Bin to ci
    df['Grouping'] = df['Grouping'].replace({'ci_bin': 'ci'})
    
    # Pivot to wide format
    df_wide = df.pivot_table(
        index=['Layer', 'Layer_Type'], 
        columns=['Grouping', 'Space'], 
        values='GDV'
    ).reset_index()
    
    # Flatten multi-index columns: ('image', 'original') -> 'gdv_by_image_original'
    new_cols = []
    for col in df_wide.columns:
        if isinstance(col, tuple):
            # Pivot table with reset_index results in ('Layer', '') and ('image', 'original')
            group, space = col
            if group in ['Layer', 'Layer_Type']:
                new_cols.append(group)
            elif group and space:
                new_cols.append(f'gdv_by_{group}_{space}')
            else:
                new_cols.append(group or space)
        else:
            new_cols.append(col)
    
    df_wide.columns = new_cols
    
    # layer_0 .. layer_36 -> layer index
    df_wide['layer_idx'] = df_wide['Layer'].str.split('_').str[-1].astype(int)
    
    return df_wide


def load_long_format_gdv_multi(
    sources: List[Tuple[str, Path]],
) -> pd.DataFrame:
    """Load and concatenate GDV results from multiple (condition_id, results_dir) pairs.

    Each entry is loaded via load_long_format_gdv() and tagged with a
    `condition` column equal to the supplied id. Missing or empty
    sources are skipped with a warning. Returns the concatenated wide-format
    frame ready for plot_gdv_progression_by_condition().
    """
    frames = []
    for condition_id, results_dir in sources:
        sub = load_long_format_gdv(Path(results_dir))
        if sub.empty:
            print(f"   Skipping condition '{condition_id}': no GDV rows at {results_dir}")
            continue
        sub = sub.copy()
        sub['condition'] = condition_id
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_curves(sources: List[Tuple[str, Path]]) -> pd.DataFrame:
    """Wrapper around `load_long_format_gdv_multi` for consistent error handling."""
    df = load_long_format_gdv_multi(sources)
    if df.empty:
        raise SystemExit(
            f"No GDV rows loaded from any of: {[str(p) for _, p in sources]}"
        )
    return df


def draw_gdv_panel(
    ax,
    gdv_df: pd.DataFrame,
    grouping: str,
    style_map: Dict[str, dict],
    line_width: float = 1.6,
    marker_size: int = 4,
    drop_curves_with_all_nan: bool = True,
) -> Tuple[List, List[str]]:
    """Draw one grouping's promptstyle-colored GDV-by-layer curves onto `ax`.

    Args:
        ax: matplotlib axes to draw onto.
        gdv_df: long-format frame from `load_long_format_gdv_multi` (must
            contain a 'condition' column — used as the curve identifier).
        grouping: grouping name (lowercased), e.g. 'image', 'persona', 'rating'.
        style_map: maps the values of gdv_df['condition'] to {label, color,
            linestyle, marker} dicts. Curves with no entry get a fallback.
        drop_curves_with_all_nan: if a curve is all-NaN for this panel
            (e.g. by-Persona for a single-persona dataset), don't draw it.

    Returns:
        (legend_handles, legend_labels) for the curves drawn in this panel.
    """
    col = f"gdv_by_{grouping}_original"

    legend_handles, legend_labels = [], []
    legend_seen = set()

    for cond_id in gdv_df["condition"].unique():
        cond_df = gdv_df[gdv_df["condition"] == cond_id]
        if drop_curves_with_all_nan and cond_df[col].isna().all():
            continue

        style = style_map.get(
            cond_id,
            {"label": cond_id, "color": "gray", "linestyle": "-", "marker": "o"},
        )

        l_part = cond_df.sort_values("layer_idx")
        line = ax.plot(
            l_part["layer_idx"].values,
            l_part[col].values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style.get("marker", "o"),
            markersize=marker_size,
            lw=line_width,
        )

        if cond_id not in legend_seen:
            legend_handles.append(line[0])
            legend_labels.append(style["label"])
            legend_seen.add(cond_id)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_ylabel(
        f"GDV {GROUPING_LABELS.get(grouping, grouping)}",
        fontsize=FS_LABEL,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FS_TICK)
    return legend_handles, legend_labels
