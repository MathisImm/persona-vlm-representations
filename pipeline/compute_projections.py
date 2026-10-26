"""Per-layer 2-D MDS projections of a persona dataset (writes projections.npz).

Writes <cache-dir>/<dataset>[__schema_split][__style_X]/projections.npz, the
caches that figures/plot_figure*_scatter.py and plot_f2_per_condition_curves.py
read; pipeline/run_gdv_and_projections.sh lists the settings of each paper cache.

The projections use a CI-stratified subset of --max-images images (all prompts
of each image are kept). One overview figure per coloring (image / persona /
promptstyle / rating / ci / schema) shows the MDS of every decoder layer, and
--from-cache re-renders these overviews without recomputing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "figures"))

from paths import CI_SCORES, HIDDEN_STATES, OUTPUT, RATINGS  # noqa: E402
from pipeline.calculate_persona_gdv import load_persona_dataset  # noqa: E402
from style_guide import set_ccn_style  # noqa: E402


COLORINGS = ("image", "persona", "promptstyle", "rating", "ci", "schema")


def _schema_of(persona_id: str) -> str:
    """Map a display persona id to its persona-schema (or "" to drop the point).

    Big-5+ ids carry the longer prefix, so test them first; anything that is
    neither schema (baseline, defaults_D*) returns "" and gets excluded by the
    --persona-schemas-only filter.
    """
    if persona_id.startswith("personas_big5_prototypical"):
        return "Big-5+"
    if persona_id.startswith("personas_prototypical"):
        return "Descriptive"
    return ""


def select_representative_images(
    image_ids: np.ndarray,
    ci_scores: np.ndarray,
    n_images: int,
    seed: int = 42,
) -> set:
    """Pick n_images representative image_ids stratified by CI bin.

    Each unique image gets one representative CI score (mean across rows
    that share its id, since each persona × image row carries the same CI).
    Bins are CI percentiles (5 buckets); we pick ~n_images/5 ids per bucket
    and top up at random if a bucket is short.
    """
    rng = np.random.default_rng(seed)
    unique_ids, first_idx = np.unique(image_ids, return_index=True)
    if len(unique_ids) <= n_images:
        return set(unique_ids.tolist())

    unique_ci = ci_scores[first_idx]
    valid = ~np.isnan(unique_ci)
    if valid.sum() < 2:
        return set(rng.choice(unique_ids, size=n_images, replace=False).tolist())

    bins = np.percentile(unique_ci[valid], [20, 40, 60, 80])
    bin_assignments = np.digitize(unique_ci, bins)

    n_bins = 5
    per_bin = max(1, n_images // n_bins)
    picks = []
    for b in range(n_bins):
        idx_in_bin = np.where(bin_assignments == b)[0]
        if len(idx_in_bin) == 0:
            continue
        take = min(per_bin, len(idx_in_bin))
        chosen = rng.choice(idx_in_bin, size=take, replace=False)
        picks.extend(unique_ids[chosen].tolist())

    # Top up at random from anything we haven't picked yet.
    if len(picks) < n_images:
        remaining = [iid for iid in unique_ids if iid not in picks]
        extra = rng.choice(remaining, size=min(n_images - len(picks), len(remaining)), replace=False)
        picks.extend(extra.tolist())

    return set(picks[:n_images])


def save_cache(cache_path: Path, payload: dict) -> None:
    """Persist computed coords + label arrays so the user can replot locally
    without recomputing MDS. Object arrays are allowed for the string
    metadata."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), **{k: np.asarray(v, dtype=object) if isinstance(v, list) else v for k, v in payload.items()})
    print(f"Cached projection data → {cache_path}")


def load_cache(cache_path: Path) -> dict:
    print(f"Loading cached projection data from {cache_path}")
    with np.load(str(cache_path), allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def coords_dict_from_cache(cached: dict, methods: list) -> dict:
    """Reverse the flat layer/method storage in the npz back into a nested dict."""
    layer_order = list(cached["__layer_order__"])
    out = {}
    for ln in layer_order:
        out[ln] = {}
        for m in methods:
            key = f"coords__{ln}__{m}"
            if key in cached:
                out[ln][m] = cached[key]
    return out


def compute_projections(
    embeddings_by_layer: Dict[str, np.ndarray],
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Returns {layer_name: {"mds": (N, 2)}} for each layer."""
    out = {}
    for ln in sorted(embeddings_by_layer.keys()):
        arr = embeddings_by_layer[ln]
        if hasattr(arr, "ndim") and arr.ndim != 2:
            print(f"Skipping {ln}: shape {arr.shape}")
            continue
        print(f"  {ln}: MDS on {arr.shape}...")
        mds = MDS(n_components=2, normalized_stress="auto", random_state=seed)
        out[ln] = {"mds": mds.fit_transform(arr)}
    return out


def render_overview(
    coords_by_layer_method: Dict[str, Dict[str, np.ndarray]],
    color_arr: np.ndarray,
    color_names: List[str],
    output_path: Path,
    methods: List[str],
    n_cols: int = 8,
    point_size: float = 6,
    cmap_name: str = "tab20",
):
    """Per-coloring overview: rows = methods, sub-rows wrap layers across n_cols."""
    set_ccn_style()
    layer_names = sorted(coords_by_layer_method.keys(), key=_layer_sort_key)
    n_layers = len(layer_names)
    n_layer_rows = (n_layers + n_cols - 1) // n_cols
    n_total_rows = len(methods) * n_layer_rows

    fig, axes = plt.subplots(
        n_total_rows,
        n_cols,
        figsize=(n_cols * 1.6, n_total_rows * 1.6),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    cmap = plt.get_cmap(cmap_name)
    n_unique = len(set(color_arr.tolist()))
    norm = plt.Normalize(vmin=0, vmax=max(1, n_unique - 1))

    for m_idx, method in enumerate(methods):
        for l_idx, ln in enumerate(layer_names):
            r_within = l_idx // n_cols
            c = l_idx % n_cols
            row = m_idx * n_layer_rows + r_within
            ax = axes[row, c]
            if method not in coords_by_layer_method[ln]:
                ax.set_visible(False)
                continue
            coords = coords_by_layer_method[ln][method]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=color_arr,
                cmap=cmap,
                norm=norm,
                s=point_size,
                alpha=0.7,
                linewidths=0,
            )
            ax.set_title(_short_layer_label(ln), fontsize=6)
            if c == 0:
                ax.set_ylabel(method.upper(), fontsize=8, fontweight="bold")
        # Hide unused panels in this method's rows
        for hidden in range(n_layers, n_layer_rows * n_cols):
            r_within = hidden // n_cols
            c = hidden % n_cols
            row = m_idx * n_layer_rows + r_within
            axes[row, c].set_visible(False)

    # Build a discrete legend strip beneath the grid.
    handles = []
    for i, name in enumerate(color_names):
        color = cmap(norm(i))
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o", color="w",
                markerfacecolor=color, markersize=6, label=str(name),
            )
        )
    n_legend_cols = min(8, max(1, len(handles)))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=n_legend_cols,
        fontsize=7,
        frameon=True,
        bbox_to_anchor=(0.5, -0.005),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06 + 0.02 * (len(handles) // n_legend_cols))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_path}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}.png / .pdf")


def _layer_sort_key(name: str):
    return int(name.split("_")[-1])


def _short_layer_label(name: str) -> str:
    if name.startswith("layer_"):
        return f"L{name.split('_')[-1]}"
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name (single config or combined).")
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=HIDDEN_STATES,
    )
    parser.add_argument(
        "--text-outputs-root",
        type=Path,
        default=RATINGS,
    )
    parser.add_argument("--ci-scores", type=Path, default=CI_SCORES,
                        help="CSV with image_id, ci_score (used for the CI-stratified image subset).")
    parser.add_argument("--max-images", type=int, default=20,
                        help="Pick this many CI-stratified representative images "
                             "and keep ALL personas for each. 0 = use all images.")
    parser.add_argument("--n-cols", type=int, default=8,
                        help="Grid columns (layers per row).")
    parser.add_argument("--colorings", default=",".join(COLORINGS),
                        help="Which color schemes to render.")
    parser.add_argument("--persona-schemas-only", action="store_true",
                        help="Drop baseline/defaults points BEFORE projecting and add a "
                             "2-class 'schema' coloring (Descriptive vs Big-5+). Routes "
                             "cache+output under '<dataset>__schema_split'.")
    parser.add_argument("--promptstyle-only", default=None,
                        help="Keep only points whose condition matches this value "
                             "(e.g. P3, original_persona). Filter applied BEFORE "
                             "projecting. Routes cache+output under suffix "
                             "'__style_<value>'. Combines with --persona-schemas-only.")
    parser.add_argument("--cmap", default="tab20",
                        help="Matplotlib colormap name for the scatter (e.g. Set1 for a "
                             "clear 2-color split).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT / "projection_overviews")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=OUTPUT / "projections",
        help="Where to write/read the per-dataset projections.npz cache. "
             "Iterate locally with --from-cache to skip recomputation.",
    )
    parser.add_argument("--from-cache", action="store_true",
                        help="Load cached coords+labels and only re-render figures.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    methods = ["mds"]
    colorings = [c.strip().lower() for c in args.colorings.split(",") if c.strip()]
    dataset_key = args.dataset + ("__schema_split" if args.persona_schemas_only else "")
    if args.promptstyle_only:
        dataset_key += f"__style_{args.promptstyle_only}"
    cache_path = args.cache_dir / dataset_key / "projections.npz"
    out_dir = args.output_dir / dataset_key

    if args.from_cache:
        if not cache_path.exists():
            print(f"--from-cache set but {cache_path} does not exist.")
            sys.exit(1)
        cached = load_cache(cache_path)
        coords = coords_dict_from_cache(cached, methods)
        label_arrays = {scheme: cached[f"labels__{scheme}"] for scheme in COLORINGS if f"labels__{scheme}" in cached}
        label_names = {scheme: list(cached[f"names__{scheme}"]) for scheme in COLORINGS if f"names__{scheme}" in cached}
    else:
        print(f"Loading dataset '{args.dataset}'...")
        data = load_persona_dataset(
            args.dataset,
            analysis_root=str(args.analysis_root),
            text_outputs_root=str(args.text_outputs_root),
            ci_scores_csv=str(args.ci_scores) if args.ci_scores else None,
        )
        n = len(data["all_image_ids"])
        if n == 0:
            print("No data loaded — abort.")
            sys.exit(1)
        print(f"  loaded {n} per-(persona,image) points")

        # 1. Image-axis subsample: keep ALL personas for the chosen images.
        all_ids_arr = np.array(data["all_image_ids"])
        if args.max_images and args.max_images > 0:
            keep_ids = select_representative_images(
                all_ids_arr,
                data["all_ci_scores"],
                args.max_images,
                seed=args.seed,
            )
            keep_idx = np.where(np.isin(all_ids_arr, list(keep_ids)))[0]
            print(f"  picked {len(keep_ids)} CI-stratified images → {len(keep_idx)} points "
                  f"(personas × images)")
        else:
            keep_idx = np.arange(n)

        # 2. Optional persona-schema filter: drop baseline/defaults BEFORE
        # projecting so MDS fits only persona points (two schemas).
        if args.persona_schemas_only:
            persona_full = np.array(data["all_personas"])[keep_idx]
            schema_full = np.array([_schema_of(p) for p in persona_full])
            sel = schema_full != ""
            keep_idx = keep_idx[sel]
            print(f"  schemas-only: kept {len(keep_idx)} persona points "
                  f"(dropped {int((~sel).sum())} baseline/defaults)")

        if args.promptstyle_only:
            cond_full = np.array(data["all_conditions"])[keep_idx]
            sel = cond_full == args.promptstyle_only
            keep_idx = keep_idx[sel]
            print(f"  promptstyle-only={args.promptstyle_only}: kept {len(keep_idx)} points "
                  f"(dropped {int((~sel).sum())} other conditions)")

        image_arr = all_ids_arr[keep_idx]
        persona_arr = np.array(data["all_personas"])[keep_idx]
        condition_arr = np.array(data["all_conditions"])[keep_idx]
        rating_arr = np.array(data["all_ratings"])[keep_idx]
        ci_raw = data["all_ci_scores"][keep_idx]

        if not np.all(np.isnan(ci_raw)):
            valid = ci_raw[~np.isnan(ci_raw)]
            bins = np.percentile(valid, [20, 40, 60, 80])
            ci_bin = np.digitize(ci_raw, bins)
        else:
            ci_bin = np.zeros_like(ci_raw, dtype=int)

        label_arrays = {
            "image": LabelEncoder().fit_transform(image_arr),
            "persona": LabelEncoder().fit_transform(persona_arr),
            "promptstyle": LabelEncoder().fit_transform(condition_arr),
            "rating": LabelEncoder().fit_transform(rating_arr),
            "ci": ci_bin.astype(int),
        }
        label_names = {
            "image": [str(x) for x in LabelEncoder().fit(image_arr).classes_],
            "persona": [str(x) for x in LabelEncoder().fit(persona_arr).classes_],
            "promptstyle": [str(x) for x in LabelEncoder().fit(condition_arr).classes_],
            "rating": [str(x) for x in LabelEncoder().fit(rating_arr).classes_],
            "ci": [f"CI bin {i}" for i in range(int(ci_bin.max()) + 1)] if not np.all(np.isnan(ci_raw)) else ["all"],
        }

        if args.persona_schemas_only:
            schema_arr = np.array([_schema_of(p) for p in persona_arr])
            label_arrays["schema"] = LabelEncoder().fit_transform(schema_arr)
            label_names["schema"] = [str(x) for x in LabelEncoder().fit(schema_arr).classes_]

        embeddings_by_layer = {ln: np.array(vals)[keep_idx]
                               for ln, vals in data["llm_embeddings_by_layer"].items()}

        print(f"Computing MDS per layer ({len(embeddings_by_layer)} layers total)...")
        coords = compute_projections(embeddings_by_layer, seed=args.seed)

        # Cache for fast local replotting.
        layer_order = sorted(coords.keys(), key=_layer_sort_key)
        payload = {
            "__layer_order__": np.asarray(layer_order, dtype=object),
            "__methods__": np.asarray(methods, dtype=object),
            "__dataset__": np.asarray(args.dataset),
        }
        for ln in layer_order:
            for m, c in coords[ln].items():
                payload[f"coords__{ln}__{m}"] = c
        for scheme, labels in label_arrays.items():
            payload[f"labels__{scheme}"] = labels
            payload[f"names__{scheme}"] = np.asarray(label_names[scheme], dtype=object)
        save_cache(cache_path, payload)

    # Render one figure per coloring (skip degenerate single-class colorings).
    out_dir.mkdir(parents=True, exist_ok=True)
    for scheme in colorings:
        if scheme not in label_arrays:
            print(f"Skipping unknown coloring '{scheme}'")
            continue
        labels = label_arrays[scheme]
        if len(np.unique(labels)) < 2:
            print(f"Skipping coloring '{scheme}': only {len(np.unique(labels))} class(es).")
            continue
        names = label_names[scheme]
        legend_names = names if len(names) <= 30 else [f"({len(names)} unique)"]
        render_overview(
            coords,
            labels,
            legend_names,
            output_path=out_dir / f"projections_{scheme}",
            methods=methods,
            n_cols=args.n_cols,
            cmap_name=args.cmap,
        )


if __name__ == "__main__":
    main()
