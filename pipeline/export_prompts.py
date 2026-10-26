"""Write the exact text of all 53 prompts to a CSV (prompts.csv in the archive).

One row per prompt: the paper's labels (condition, schema, persona), the
folders and persona_id under which its ratings and hidden states are stored,
and the prompt text exactly as built by prompt_conditions.py. The image is
passed before this text in the same user turn, and the model's chat template
is applied on top.

    python pipeline/export_prompts.py --output prompts.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CONFIGS, DATA_ROOT  # noqa: E402
from pipeline.prompt_conditions import CONDITIONS  # noqa: E402

COLUMNS = ["condition", "schema", "persona", "ratings_dataset",
           "hidden_states_dataset", "persona_id", "prompt"]
SCHEMAS = {"personas_prototypical": "socio-demographic",
           "personas_big5_prototypical": "Big Five"}
# Paper style label -> (registry condition, dataset suffix)
STYLES = {"S1": ("original_persona", ""), "S2": ("P1", "_P1"), "S3": ("P2", "_P2"),
          "S4": ("P3", "_P3"), "S5": ("P4", "_P4")}


def prompt_rows(config_dir=CONFIGS):
    """The 53 prompts: Base, D1, D2, then schema x style x persona."""
    both = ";".join(SCHEMAS)
    # Base: hidden states in the S1 folders, ratings in the S2 folders (same in both schemas).
    rows = [{"condition": "Base", "schema": "", "persona": "",
             "ratings_dataset": ";".join(f"{s}_P1" for s in SCHEMAS),
             "hidden_states_dataset": both, "persona_id": "baseline",
             "prompt": CONDITIONS["original_baseline"].build()}]
    for d in ("D1", "D2"):
        rows.append({"condition": d, "schema": "", "persona": "",
                     "ratings_dataset": f"defaults_{d}", "hidden_states_dataset": f"defaults_{d}",
                     "persona_id": "no_persona", "prompt": CONDITIONS[d].build()})
    for schema, schema_label in SCHEMAS.items():
        personas = json.loads((Path(config_dir) / f"{schema}.json").read_text())
        for style, (cond, suffix) in STYLES.items():
            for k in range(1, 6):
                desc = personas[f"persona_{k}"]["description"]
                rows.append({"condition": style, "schema": schema_label, "persona": f"P{k}",
                             "ratings_dataset": schema + suffix,
                             "hidden_states_dataset": schema + suffix,
                             "persona_id": f"persona_{k}",
                             "prompt": CONDITIONS[cond].build(desc)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "prompts.csv")
    args = parser.parse_args()
    rows = prompt_rows()
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} prompts to {args.output}")


if __name__ == "__main__":
    main()
