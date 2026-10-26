"""Shared helpers for the GDV-annotated projection scatter figures (Figure 1/2).

Both plot_figure1_combined_scatter.py and plot_figure2_schema_split_scatter.py
load a projections.npz cache (via _cache_io.load_cache /
coords_dict_from_cache), color points by persona identity / promptstyle /
schema / image / rating, and title each panel with a live-computed GDV
(src.gdv.cmpGDV on that panel's own 2D MDS coords + label array — never the
precomputed gdv_results.csv, so the number always matches the rendered
points, and both the "combined persona x promptstyle" and "schema" groupings
have no precomputed equivalent to fall back to). Centralize layer-CLI
parsing, color/marker maps, and title formatting here so neither script
duplicates it.
"""
from __future__ import annotations

import colorsys
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_FIGURES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FIGURES_DIR.parent
for _p in (_FIGURES_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from src.gdv import cmpGDV  # noqa: E402
from style_guide import CONDITION_STYLES  # noqa: E402

DEFAULT_LAYERS = [2, 10, 22, 28, 36]

DESCRIPTIVE_PREFIX = "personas_prototypical_persona_"
BIG5_PREFIX = "personas_big5_prototypical_persona_"

PROMPTSTYLE_ORDER = ["original_persona", "P1", "P2", "P3", "P4"]
PROMPTSTYLE_MARKERS = {"original_persona": "o", "P1": "s", "P2": "^", "P3": "D", "P4": "v"}

# Display numbering: prompt styles are S1..S5 and personas are P1..P5, so the
# two axes never collide in a legend. (The raw condition ids call four of the
# five styles "P1".."P4", which reads as a persona index otherwise.)
PROMPTSTYLE_DISPLAY = {"original_persona": "S1", "P1": "S2", "P2": "S3", "P3": "S4", "P4": "S5"}
PROMPTSTYLE_DESCR = {
    "original_persona": "first-person",
    "P1": "third-person",
    "P2": "persona after task",
    "P3": "explicit override",
    "P4": "statistical framing",
}

# Full 1-5 interestingness scale. The projection caches only ever contain the
# first four levels ("Extremely Interesting" is never produced), but colors are
# anchored to the full scale so shades stay comparable across datasets.
RATING_ORDER = ["Not Interesting", "Slightly Interesting", "Moderately Interesting",
                "Very Interesting", "Extremely Interesting"]
RATING_SHORT = {
    "Not Interesting": "Not",
    "Slightly Interesting": "Slightly",
    "Moderately Interesting": "Moderately",
    "Very Interesting": "Very",
    "Extremely Interesting": "Extremely",
}

# Default (baseline/D1/D2) categories, pulled from style_guide.CONDITION_STYLES
# so this new figure family stays visually consistent with the rest of the
# persona-paper figures. Keyed by the persona-id string used in the
# projections.npz caches.
DEFAULT_CATEGORY_STYLES = {
    "baseline": {
        "label": CONDITION_STYLES["original_baseline"]["label"],
        "color": CONDITION_STYLES["original_baseline"]["color"],
        "marker": CONDITION_STYLES["original_baseline"]["marker"],
    },
    "defaults_D1_no_persona": {
        "label": CONDITION_STYLES["D1"]["label"],
        "color": CONDITION_STYLES["D1"]["color"],
        "marker": CONDITION_STYLES["D1"]["marker"],
    },
    "defaults_D2_no_persona": {
        "label": CONDITION_STYLES["D2"]["label"],
        "color": CONDITION_STYLES["D2"]["color"],
        "marker": CONDITION_STYLES["D2"]["marker"],
    },
}


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_layers(spec: str) -> List[int]:
    """Parse a comma-separated layer-index spec, e.g. '2,10,22,28,36'."""
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def add_layers_arg(parser, default: Sequence[int] = None) -> None:
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=list(default) if default is not None else list(DEFAULT_LAYERS),
        help="Comma-separated LLM layer indices, e.g. 2,10,22,28,36.",
    )


def layer_keys(layers: Sequence[int]) -> List[str]:
    return [f"layer_{i}" for i in layers]


# ---------------------------------------------------------------------------
# GDV + panel title
# ---------------------------------------------------------------------------

def compute_layer_gdv(coords_2d: np.ndarray, label_ids: np.ndarray) -> float:
    """Live GDV for one panel: cmpGDV on that panel's own 2D coords + labels.

    Groups with fewer than 2 points are dropped (cmpGDV needs intra-cluster
    distances). Returns NaN if fewer than 2 groups survive or on any error.
    """
    try:
        ids = np.asarray(label_ids)
        counts = {v: int((ids == v).sum()) for v in np.unique(ids)}
        keep_labels = {v for v, n in counts.items() if n >= 2}
        if len(keep_labels) < 2:
            return float("nan")
        mask = np.isin(ids, list(keep_labels))
        _, _, gdv = cmpGDV(np.asarray(coords_2d)[mask], ids[mask])
        return float(gdv)
    except Exception:
        return float("nan")


def gdv_title(short_label: str, gdv_val: float) -> str:
    if np.isnan(gdv_val):
        return short_label
    return f"{short_label} ({gdv_val:.2f})"


# ---------------------------------------------------------------------------
# Persona hue palette (schema-aware, n-parameterized)
# ---------------------------------------------------------------------------

def _hsv_hex(hue_deg: float, sat: float, val: float) -> str:
    r, g, b = colorsys.hsv_to_rgb((hue_deg % 360) / 360.0, sat, val)
    return mcolors.to_hex((r, g, b))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def persona_hue_palette(
    n_per_schema: int,
    cool_arc: Tuple[float, float] = (150, 250),
    warm_arc: Tuple[float, float] = (-20, 60),
    sat: float = 0.72,
    val_range: Tuple[float, float] = (0.55, 0.92),
) -> Dict[str, str]:
    """Descriptive personas (1..n) get cool hues, Big-5+ personas (1..n) get
    warm hues, so hue alone identifies both persona identity and schema.
    """
    palette: Dict[str, str] = {}
    for i in range(n_per_schema):
        t = 0.0 if n_per_schema == 1 else i / (n_per_schema - 1)
        val = _lerp(val_range[0], val_range[1], t)
        palette[f"{DESCRIPTIVE_PREFIX}{i + 1}"] = _hsv_hex(_lerp(cool_arc[0], cool_arc[1], t), sat, val)
        palette[f"{BIG5_PREFIX}{i + 1}"] = _hsv_hex(_lerp(warm_arc[0], warm_arc[1], t), sat, val)
    return palette


def n_per_schema_from_names(persona_names: Sequence[str]) -> int:
    n_desc = sum(1 for p in persona_names if p.startswith(DESCRIPTIVE_PREFIX))
    n_big5 = sum(1 for p in persona_names if p.startswith(BIG5_PREFIX))
    return max(n_desc, n_big5, 1)


def promptstyle_label(style: str, with_descr: bool = True) -> str:
    """'original_persona' -> 'S1 (Imagine you)'."""
    tag = PROMPTSTYLE_DISPLAY.get(style, style)
    if with_descr and style in PROMPTSTYLE_DESCR:
        return f"{tag} ({PROMPTSTYLE_DESCR[style]})"
    return tag


def persona_index(persona_name: str) -> str:
    """'personas_big5_prototypical_persona_3' -> 'P3'."""
    for prefix in (BIG5_PREFIX, DESCRIPTIVE_PREFIX):
        if persona_name.startswith(prefix):
            return f"P{persona_name[len(prefix):]}"
    return persona_name


# Paper names of the two schemas. The keys are the labels stored in the
# projection caches and used as palette keys, so they stay as they are.
SCHEMA_DISPLAY = {"Descriptive": "Socio-demographic", "Big-5+": "Big Five"}


def persona_display_label(persona_name: str) -> str:
    """'personas_big5_prototypical_persona_3' -> 'P3'. The schema is given by
    the figure (one schema per panel) or by a legend section header."""
    if persona_name.startswith((BIG5_PREFIX, DESCRIPTIVE_PREFIX)):
        return persona_index(persona_name)
    return persona_name


def rating_level_colors(cmap_name: str) -> Dict[str, str]:
    """One color per level of the full 1-5 scale, sampled from the same
    sequential colormap used for the scatter, so a discrete rating legend
    matches the plotted point colors exactly."""
    cmap = plt.get_cmap(cmap_name)
    n = len(RATING_ORDER)
    return {name: mcolors.to_hex(cmap(i / (n - 1))) for i, name in enumerate(RATING_ORDER)}


def image_tab20_palette(image_names: Sequence[str]) -> Dict[str, str]:
    """Assign each image id a distinct color from the 'tab20' colormap."""
    cmap = plt.get_cmap("tab20")
    norm = plt.Normalize(vmin=0, vmax=max(1, len(image_names) - 1))
    return {name: mcolors.to_hex(cmap(norm(i))) for i, name in enumerate(image_names)}


def schema_anchor_colors(sat: float = 0.72, val: float = 0.75) -> Dict[str, str]:
    """Mid-arc representative color per schema, from the same hue arcs used
    by persona_hue_palette, so a 'Schema' coloring visually rhymes with the
    per-persona hues."""
    return {
        "Descriptive": _hsv_hex((150 + 250) / 2, sat, val),
        "Big-5+": _hsv_hex((-20 + 60) / 2, sat, val),
    }


# ---------------------------------------------------------------------------
# Generic discrete coloring + legend helpers (Figure 2 rows)
# ---------------------------------------------------------------------------

def discrete_rgba_array(
    label_ids: np.ndarray, names: List[str], color_by_name: Dict[str, str],
    fallback: str = "#999999",
) -> np.ndarray:
    lut = np.array([mcolors.to_rgba(color_by_name.get(n, fallback)) for n in names])
    return lut[np.asarray(label_ids)]


def discrete_legend_handles(
    order: List[str],
    color_by_name: Dict[str, str],
    label_by_name: Dict[str, str] = None,
    marker: str = "o",
    markersize: float = 6,
) -> List[Line2D]:
    label_by_name = label_by_name or {}
    return [
        Line2D(
            [0], [0], marker=marker, linestyle="None",
            markerfacecolor=color_by_name[n], markeredgecolor=color_by_name[n],
            markersize=markersize, label=label_by_name.get(n, n),
        )
        for n in order if n in color_by_name
    ]


def rating_rank_array(rating_label_ids: np.ndarray, rating_names: List[str]) -> np.ndarray:
    """Remap LabelEncoder's alphabetical rating ids to the natural order
    Not(0) < Slightly(1) < Moderately(2) < Very(3). LabelEncoder sorts class
    names alphabetically, so the raw ids do NOT follow interestingness order
    (id 0 is 'Moderately Interesting', not 'Not Interesting') — using the raw
    ids directly with a sequential colormap would color classes out of order.
    """
    name_to_rank = {name: i for i, name in enumerate(RATING_ORDER)}
    ids = np.asarray(rating_label_ids)
    return np.array([name_to_rank[rating_names[i]] for i in ids])


# ---------------------------------------------------------------------------
# Combined persona x promptstyle + defaults grouping (Figure 1)
# ---------------------------------------------------------------------------

def build_combined_groups(
    persona_names: List[str],
    promptstyle_names: List[str],
    persona_labels: np.ndarray,
    promptstyle_labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, dict]]:
    """Combined persona x promptstyle categories, plus the 3 defaults as
    their own singular categories (10 personas x 5 styles + 3 defaults = up
    to 53 groups).

    Returns:
        group_id_per_point: int array, one id per input point (-1 if the
            point's (persona, promptstyle) pair matched nothing, which
            should not happen for these caches but is guarded defensively).
        group_style: dict keyed by group id -> {color, marker, schema,
            persona, style, legend_label, edge_linestyle (optional)}.
    """
    n = n_per_schema_from_names(persona_names)
    hues = persona_hue_palette(n)

    group_style: Dict[int, dict] = {}
    key_to_gid: Dict[Tuple[str, str], int] = {}
    gid = 0
    for schema_label, prefix in (("Descriptive", DESCRIPTIVE_PREFIX), ("Big-5+", BIG5_PREFIX)):
        for i in range(1, n + 1):
            pname = f"{prefix}{i}"
            if pname not in persona_names:
                continue
            for style in PROMPTSTYLE_ORDER:
                if style not in promptstyle_names:
                    continue
                key_to_gid[(pname, style)] = gid
                group_style[gid] = {
                    "color": hues[pname],
                    "marker": PROMPTSTYLE_MARKERS[style],
                    "schema": schema_label,
                    "persona": pname,
                    "style": style,
                    "legend_label": f"P{i}",
                }
                gid += 1

    for default_name, style in DEFAULT_CATEGORY_STYLES.items():
        if default_name not in persona_names:
            continue
        key_to_gid[(default_name, None)] = gid
        group_style[gid] = {
            "color": style["color"],
            "marker": style["marker"],
            "schema": "Defaults",
            "persona": default_name,
            "style": None,
            "edge_linestyle": style.get("edge_linestyle"),
            "legend_label": style["label"],
        }
        gid += 1

    persona_of_point = np.array(persona_names)[np.asarray(persona_labels)]
    style_of_point = np.array(promptstyle_names)[np.asarray(promptstyle_labels)]

    out = np.full(len(persona_labels), -1, dtype=int)
    for idx in range(len(persona_labels)):
        pname = persona_of_point[idx]
        key = (pname, style_of_point[idx])
        if key in key_to_gid:
            out[idx] = key_to_gid[key]
        elif (pname, None) in key_to_gid:
            out[idx] = key_to_gid[(pname, None)]

    n_unmatched = int((out == -1).sum())
    if n_unmatched:
        print(
            f"build_combined_groups: {n_unmatched} points had no recognized "
            f"(persona, promptstyle) combo — dropped from render."
        )
    return out, group_style
