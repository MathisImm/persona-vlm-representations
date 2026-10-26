"""Layer-wise GDV numbers behind the Results (Fig. 4, Fig. B1, Fig. C1).

Prints, for each schema and prompting style, the GDV by image, persona and
rating in the original hidden-state space at every decoder layer, the layer
of strongest clustering, and the pooled-style comparison of clustering by
prompting style versus persona. More negative = stronger clustering.

    python analysis/gdv_claims.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import GDV  # noqa: E402

SCHEMAS = {"personas_big5_prototypical": "Big Five", "personas_prototypical": "socio-demographic"}
# S1 uses the *_no_baseline run so that every style has the same five personas.
STYLES = {"S1": "_no_baseline", "S2": "_P1", "S3": "_P2", "S4": "_P3", "S5": "_P4"}


def load(dataset):
    df = pd.read_csv(GDV / dataset / "gdv_results.csv")
    df = df[(df.Layer_Type == "LLM") & (df.Space == "Original")].copy()
    df["L"] = df.Layer.str.extract(r"(\d+)$")[0].astype(int)
    return df.pivot_table(index="L", columns="Grouping", values="GDV")


pd.set_option("display.width", 200)
for schema, name in SCHEMAS.items():
    print(f"\n=== {name} ({schema})")
    per_style = {s: load(schema + sfx) for s, sfx in STYLES.items()}
    for g in ["Image", "Persona", "Rating"]:
        tab = pd.DataFrame({s: d[g] for s, d in per_style.items()})
        print(f"\n-- GDV by {g}, per layer and style")
        print(tab.round(3).T.to_string())
        print("   layer of strongest clustering:", tab.idxmin().to_dict())
        print("   mean L0-15 / L20-36:", {s: (round(float(tab[s].loc[0:15].mean()), 3), round(float(tab[s].loc[20:36].mean()), 3))
                                           for s in tab})

    pooled = load(schema + "_all_styles")
    print(f"\n-- All five styles pooled ({schema}_all_styles, 2,500 points per layer)")
    print(pooled[["PromptStyle", "Persona", "Image", "Rating"]].round(3).T.to_string())
    print("   style clustering strongest at layer", int(pooled.PromptStyle.idxmin()),
          "| ratio style/persona GDV at layers 30-36:",
          round(float((pooled.PromptStyle / pooled.Persona).loc[30:36].mean()), 1))
