"""Read the projections.npz caches written by pipeline/compute_projections.py.

Copied from compute_projections.py so that the figure scripts do not import
scikit-learn.
"""
from pathlib import Path

import numpy as np


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


def _layer_sort_key(name: str):
    return int(name.split("_")[-1])


def _short_layer_label(name: str) -> str:
    if name.startswith("layer_"):
        return f"L{name.split('_')[-1]}"
    return name
