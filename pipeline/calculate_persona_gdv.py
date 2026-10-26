"""Layer-wise GDV of the persona hidden states.

For one dataset (a single persona set, or a combination defined in
COMBINED_DATASETS) this loads every embeddings.h5 below <analysis-root>, labels
each (persona, image) point by image, persona, rating, CI bin and prompt style,
and writes the GDV of each grouping per layer to <output-dir>/gdv_results.csv.

The GDV is computed in the original hidden-state space of every decoder layer:
    python pipeline/calculate_persona_gdv.py --dataset personas_big5_prototypical_P1
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sys

# Ensure src module is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import CI_SCORES, HIDDEN_STATES, OUTPUT, RATINGS
from src.gdv import cmpGDV

P_CONDITIONS = ("P1", "P2", "P3", "P4")


def safe_cmpGDV(space_emb, label_arr):
    """cmpGDV wrapper that drops classes with fewer than 2 members first.

    computeGDV's intra-cluster distance divides by NP*(NP-1) per class; a
    singleton class (NP=1) makes that division 0/0, which poisons the whole
    intraMean (and therefore the GDV) with NaN for every layer/space, since
    the label composition doesn't change across layers. This showed up for
    e.g. a "Rating" grouping where one persona-condition produced exactly
    one "Extremely Interesting" response. Mirrors the same guard already
    used by figures/_scatter_common.compute_layer_gdv.
    """
    labels = np.asarray(label_arr)
    counts = {v: int((labels == v).sum()) for v in np.unique(labels)}
    keep = {v for v, n in counts.items() if n >= 2}
    if len(keep) < 2:
        return np.nan, np.nan, np.nan
    mask = np.isin(labels, list(keep))
    return cmpGDV(np.asarray(space_emb)[mask], labels[mask])


def _condition_from_config(config: str) -> str:
    """Map a persona-set config name to its prompt-condition id."""
    if config.startswith("defaults_"):
        return config.replace("defaults_", "", 1)
    for p in P_CONDITIONS:
        if config.endswith(f"_{p}"):
            return p
    return "original_persona"


def _persona_set_from_config(config: str) -> str:
    """Strip prompt-condition suffix to recover the underlying persona-set."""
    if config.startswith("defaults_"):
        return config
    for p in P_CONDITIONS:
        if config.endswith(f"_{p}"):
            return config[: -(len(p) + 1)]
    return config


# Each entry is (configs, share_labels, exclude_baseline).
#   share_labels=True merges the same persona id across conditions into a
#       single class for by-Persona GDV (i.e. persona_1 in P1 and P3 share
#       a label), so the by-Persona number answers "do personas cluster
#       regardless of how they're framed?".
#   exclude_baseline=True drops the `baseline` persona subdir during loading,
#       so a single-config dataset becomes 5 personas instead of 5+baseline
#       and lines up class-count-wise with the suffixed _P1..P4 datasets.
COMBINED_DATASETS = {
    "combined_prototypical": (
        ["personas_prototypical", "personas_big5_prototypical"],
        False,
        False,
    ),
    "personas_prototypical_with_defaults": (
        ["personas_prototypical", "defaults_D1", "defaults_D2"],
        False,
        False,
    ),
    "personas_big5_prototypical_with_defaults": (
        ["personas_big5_prototypical", "defaults_D1", "defaults_D2"],
        False,
        False,
    ),
    "combined_prototypical_with_defaults": (
        [
            "personas_prototypical",
            "personas_big5_prototypical",
            "defaults_D1",
            "defaults_D2",
        ],
        False,
        False,
    ),
    "combined_prototypical_all_conditions": (
        [
            "personas_prototypical",
            "personas_prototypical_P1",
            "personas_prototypical_P2",
            "personas_prototypical_P3",
            "personas_prototypical_P4",
            "personas_big5_prototypical",
            "personas_big5_prototypical_P1",
            "personas_big5_prototypical_P2",
            "personas_big5_prototypical_P3",
            "personas_big5_prototypical_P4",
            "defaults_D1",
            "defaults_D2",
        ],
        True,
        False,
    ),
    "personas_prototypical_all_conditions_with_defaults": (
        [
            "personas_prototypical",
            "personas_prototypical_P1",
            "personas_prototypical_P2",
            "personas_prototypical_P3",
            "personas_prototypical_P4",
            "defaults_D1",
            "defaults_D2",
        ],
        True,
        False,
    ),
    "personas_big5_prototypical_all_conditions_with_defaults": (
        [
            "personas_big5_prototypical",
            "personas_big5_prototypical_P1",
            "personas_big5_prototypical_P2",
            "personas_big5_prototypical_P3",
            "personas_big5_prototypical_P4",
            "defaults_D1",
            "defaults_D2",
        ],
        True,
        False,
    ),
    # All five prompt styles of one schema, without the task-only baseline and
    # without D1/D2: by-PromptStyle then measures clustering by style alone
    # (5 classes), and by-Persona uses labels shared across styles (5 classes).
    "personas_prototypical_all_styles": (
        [
            "personas_prototypical",
            "personas_prototypical_P1",
            "personas_prototypical_P2",
            "personas_prototypical_P3",
            "personas_prototypical_P4",
        ],
        True,
        True,
    ),
    "personas_big5_prototypical_all_styles": (
        [
            "personas_big5_prototypical",
            "personas_big5_prototypical_P1",
            "personas_big5_prototypical_P2",
            "personas_big5_prototypical_P3",
            "personas_big5_prototypical_P4",
        ],
        True,
        True,
    ),
    # Baseline-stripped variants for fair comparison against the suffixed
    # _P1..P4 runs in F2 (per-condition GDV curves).
    "personas_prototypical_no_baseline": (
        ["personas_prototypical"],
        False,
        True,
    ),
    "personas_big5_prototypical_no_baseline": (
        ["personas_big5_prototypical"],
        False,
        True,
    ),
}

# Per-style datasets S2-S5 contain only the five personas. Any Base run that
# sits in the same folder (the task-only prompt is re-run in every style folder
# for the ratings) is excluded, as for S1 via *_no_baseline.
for _schema in ("personas_prototypical", "personas_big5_prototypical"):
    for _p in P_CONDITIONS:
        COMBINED_DATASETS[f"{_schema}_{_p}"] = ([f"{_schema}_{_p}"], False, True)


# Order in which the persona folders of one dataset are read. The original run
# took the cluster's filesystem order, which was this one in every folder. The
# GDV does not depend on the order, but the MDS initialisation does, so fixing it
# here makes compute_projections.py reproduce the archived projections exactly.
LOAD_ORDER = ["baseline", "persona_1", "persona_2", "persona_4", "persona_5", "persona_3", "no_persona"]


def _load_order(p_dir: Path):
    name = p_dir.name
    return (LOAD_ORDER.index(name), "") if name in LOAD_ORDER else (len(LOAD_ORDER), name)


def resolve_dataset(name: str):
    """Resolve a dataset name into (configs, share_labels, exclude_baseline).
    Single-config datasets pass through unchanged; combined datasets expand
    via the dispatch."""
    return COMBINED_DATASETS.get(name, ([name], False, False))


def load_persona_dataset(
    dataset_name,
    analysis_root,
    text_outputs_root,
    ci_scores_csv=None,
):
    """Load embeddings + per-image labels for a (possibly combined) dataset.

    Mirrors the loading loop used by the GDV computation so the projection
    overview script can plot the same point cloud the GDV is summarising.

    Returns dict with keys:
        all_image_ids (list[str]), all_personas (list[str]),
        all_conditions (list[str]), all_ratings (list[str]),
        all_ci_scores (np.ndarray), llm_embeddings_by_layer (dict[str, list]),
        llm_layer_names (list[str]).
    """
    configs, share_labels, exclude_baseline = resolve_dataset(dataset_name)
    all_image_ids = []
    all_personas = []
    all_conditions = []
    all_ratings = []
    all_ci_scores = []
    llm_embeddings_by_layer = {}
    handled_personas = set()
    llm_layer_names = []

    for config in configs:
        config_condition = _condition_from_config(config)
        persona_set = _persona_set_from_config(config)
        csv_path = Path(text_outputs_root) / config / "persona_text_outputs.csv"
        if not csv_path.exists():
            print(f"Warning: text outputs missing at {csv_path}")
            continue
        df_text = pd.read_csv(csv_path)

        analysis_dir = Path(analysis_root) / config
        if not analysis_dir.exists():
            print(f"Warning: analysis dir missing at {analysis_dir}")
            continue

        for p_dir in sorted(analysis_dir.iterdir(), key=_load_order):
            if not p_dir.is_dir():
                continue
            persona_id = p_dir.name

            if exclude_baseline and persona_id == "baseline":
                continue

            h5_path = None
            for model_dir in p_dir.iterdir():
                if model_dir.is_dir() and "embeddings.h5" in os.listdir(model_dir):
                    h5_path = model_dir / "embeddings.h5"
                    break
            if h5_path is None:
                continue

            if persona_id == "baseline":
                dedup_key = "baseline"
                display_persona_id = "baseline"
                effective_condition = "original_baseline"
            else:
                dedup_key = f"{config}::{persona_id}"
                if share_labels:
                    display_persona_id = f"{persona_set}_{persona_id}"
                else:
                    display_persona_id = f"{config}_{persona_id}"
                effective_condition = config_condition

            if dedup_key in handled_personas:
                continue

            try:
                with h5py.File(h5_path, "r") as f:
                    if "image_ids" not in f:
                        continue
                    ids = [d.decode() if isinstance(d, bytes) else d for d in f["image_ids"][:]]
                    ci_vals = f["ci_scores"][:] if "ci_scores" in f else [np.nan] * len(ids)
                    if "layers" not in f:
                        continue
                    if not llm_layer_names:
                        llm_layer_names = sorted(f["layers"].keys(), key=lambda x: int(x.split("_")[1]))
                    for l_name in llm_layer_names:
                        if l_name not in f["layers"]:
                            continue
                        llm_embeddings_by_layer.setdefault(l_name, []).extend(f["layers"][l_name][:])
            except Exception as e:
                print(f"Error reading {h5_path}: {e}")
                continue

            if persona_id == "baseline":
                # Base hidden states sit in the S1 folder, its ratings in the
                # S2 folder (<schema>_P1), as in the figures.
                df_base = pd.read_csv(Path(text_outputs_root) / f"{config}_P1" / "persona_text_outputs.csv")
                df_persona = df_base[df_base["persona_id"] == persona_id]
            else:
                df_persona = df_text[df_text["persona_id"] == persona_id]
            rating_map = {}
            for _, row in df_persona.iterrows():
                rating_map[str(row["image_id"])] = parse_rating(row["raw_response"])

            for i, img_id in enumerate(ids):
                all_image_ids.append(img_id)
                all_personas.append(display_persona_id)
                all_conditions.append(effective_condition)
                all_ci_scores.append(ci_vals[i])
                r = rating_map.get(img_id, "Unknown")
                if not isinstance(r, str):
                    r = "Unknown"
                else:
                    r = r.strip().title()
                all_ratings.append(r)

            handled_personas.add(dedup_key)
            print(
                f"Loaded {len(ids)} images for {config}/{persona_id} "
                f"(label={display_persona_id}, condition={effective_condition})"
            )

    # The persona embeddings.h5 files carry NaN CI scores; take them from the
    # CI table of the image set instead.
    ci_scores_arr = np.array(all_ci_scores)
    if ci_scores_csv and np.all(np.isnan(ci_scores_arr)) and os.path.exists(ci_scores_csv):
        df_ci = pd.read_csv(ci_scores_csv, dtype={"image_id": str})
        id_to_ci = dict(zip(df_ci["image_id"], df_ci["ci_score"]))
        ci_scores_arr = np.array([id_to_ci.get(img_id, np.nan) for img_id in all_image_ids])

    return {
        "all_image_ids": all_image_ids,
        "all_personas": all_personas,
        "all_conditions": all_conditions,
        "all_ratings": all_ratings,
        "all_ci_scores": ci_scores_arr,
        "llm_embeddings_by_layer": llm_embeddings_by_layer,
        "llm_layer_names": llm_layer_names,
    }


def parse_rating(raw_response):
    if not isinstance(raw_response, str) or pd.isna(raw_response):
        return "Unknown"
    try:
        if "{" in raw_response and "}" in raw_response:
            json_str = raw_response[raw_response.find("{") : raw_response.rfind("}") + 1]
            data = json.loads(json_str)
            return data.get("interestingness", "Unknown")
        else:
            return "Unknown"
    except json.JSONDecodeError:
        return "Unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, 
                        help="e.g. personas_prototypical, combined_prototypical")
    parser.add_argument("--analysis-root", type=str, default=str(HIDDEN_STATES),
                        help="Folder with <dataset>/<persona>/<model>/embeddings.h5")
    parser.add_argument("--text-outputs-root", type=str, default=str(RATINGS),
                        help="Folder with <dataset>/persona_text_outputs.csv")
    parser.add_argument("--ci-scores", type=str, default=str(CI_SCORES),
                        help="CSV with image_id, ci_score (the persona embeddings carry no CI)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: <PERSONA_OUTPUT>/gdv/<dataset>")
    args = parser.parse_args()

    data = load_persona_dataset(
        args.dataset,
        analysis_root=args.analysis_root,
        text_outputs_root=args.text_outputs_root,
        ci_scores_csv=args.ci_scores,
    )
    all_image_ids = data["all_image_ids"]
    all_personas = data["all_personas"]
    all_conditions = data["all_conditions"]
    all_ratings = data["all_ratings"]
    ci_scores_arr = data["all_ci_scores"]
    llm_embeddings_by_layer = data["llm_embeddings_by_layer"]
    llm_layer_names = data["llm_layer_names"]

    if not llm_embeddings_by_layer:
        print("No data found!")
        return

    image_labels = LabelEncoder().fit_transform(all_image_ids)
    persona_labels = LabelEncoder().fit_transform(all_personas)
    rating_labels = LabelEncoder().fit_transform(all_ratings)
    condition_labels = LabelEncoder().fit_transform(all_conditions)
    n_unique_conditions = len(set(all_conditions))

    # Bin CI scores into 5 percentile bins if we have them
    if not np.all(np.isnan(ci_scores_arr)):
        valid_ci = ci_scores_arr[~np.isnan(ci_scores_arr)]
        if len(valid_ci) > 0:
            bins = np.percentile(valid_ci, [20, 40, 60, 80])
            ci_labels = np.digitize(ci_scores_arr, bins)
            print(f"Binned CI scores into 5 buckets using non-NaN values (range: {valid_ci.min():.4f} - {valid_ci.max():.4f})")
        else:
            ci_labels = np.zeros_like(ci_scores_arr)
    else:
        print("Warning: All CI scores are NaN even after fallback.")
        ci_labels = np.zeros_like(ci_scores_arr)

    labels_list = [("Image", image_labels), ("Persona", persona_labels), ("Rating", rating_labels), ("CI_Bin", ci_labels)]
    if n_unique_conditions > 1:
        labels_list.append(("PromptStyle", condition_labels))

    results = []
    for l_name in llm_layer_names:
        print(f"--- Processing LLM {l_name} ---")
        layer_embeddings = np.array(llm_embeddings_by_layer[l_name])
        for label_name, label_arr in labels_list:
            # safe_cmpGDV emits NaN itself if fewer than 2 classes
            # survive (e.g. Defaults_D1/D2's single-persona case, or
            # a rating level with only 1 response).
            intra, inter, gdv_val = safe_cmpGDV(layer_embeddings, label_arr)
            results.append({"Dataset":args.dataset, "Layer_Type":"LLM", "Layer":l_name, "Space":"Original", "Grouping":label_name, "Intra_Mean":intra, "Inter_Mean":inter,"GDV":gdv_val})

    df_results = pd.DataFrame(results)
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT / "gdv" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "gdv_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
