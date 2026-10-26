#!/usr/bin/env python3
"""Extract layer-wise hidden states for every persona of one persona set.

Calls pipeline/extract_embeddings.py once per entry of the personas JSON and
writes <output-dir>/<persona_id>/<model_folder>/embeddings.h5, the layout the
GDV and projection scripts read.

Usage (one prompt style of the Big Five schema, see cluster/hidden_states.slurm):
    python pipeline/run_persona_analysis.py \
        --output-dir data/hidden_states/personas_big5_prototypical_P1 \
        --personas-config configs/personas_big5_prototypical.json \
        --prompt-condition P1 --image-ids-file configs/image_ids_100_with_hidden_states.txt
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from paths import IMAGE_DIR, IMAGE_IDS_100  # noqa: E402
from pipeline.prompt_conditions import CONDITIONS  # noqa: E402


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("persona_analysis")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "persona_analysis.log")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def check_embeddings_exist(output_dir: Path, persona_name: str) -> bool:
    """True if a complete (> 10 MB) embeddings.h5 already exists for this persona."""
    persona_dir = output_dir / persona_name
    if persona_dir.exists():
        for subdir in persona_dir.iterdir():
            f = subdir / "embeddings.h5"
            if subdir.is_dir() and f.exists() and f.stat().st_size > 10 * 1024 * 1024:
                return True
    return False


def extract_persona_embeddings(
    persona_name: str,
    persona_config: Dict[str, str],
    output_dir: Path,
    model: str,
    image_dir: Path,
    device: str,
    pooling: str,
    extract_layers: str,
    image_ids_file: Path,
    prompt_condition: str,
    limit: Optional[int],
    logger: logging.Logger,
) -> bool:
    persona_output = output_dir / persona_name
    persona_output.mkdir(parents=True, exist_ok=True)
    if check_embeddings_exist(output_dir, persona_name):
        logger.info(f"  Embeddings already exist for {persona_name}. Skipping.")
        return True

    cmd = [
        sys.executable, str(REPO_ROOT / "pipeline" / "extract_embeddings.py"),
        "--model", model,
        "--image-dir", str(image_dir),
        "--output-dir", str(persona_output),
        "--device", device,
        "--extract-layers", extract_layers,
        "--pooling", pooling,
        "--image-ids-file", str(image_ids_file),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    # Same rule as extract_persona_text.py: the 'baseline' entry (empty
    # description) always gets the task-only prompt, whatever the run's style.
    desc = persona_config.get("description", "")
    condition = CONDITIONS[prompt_condition]
    if persona_name == "baseline" or (condition.requires_persona and not desc):
        cmd.extend(["--prompt-condition", "original_baseline", "--use-baseline-prompt"])
    elif condition.requires_persona:
        cmd.extend(["--persona", desc, "--prompt-condition", prompt_condition])
    else:
        # No-persona condition (D1, D2)
        cmd.extend(["--prompt-condition", prompt_condition])

    logger.info(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        logger.error(f"  Extraction failed for {persona_name}:\n{result.stderr}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Dataset folder, e.g. <hidden_states>/personas_big5_prototypical_P1")
    parser.add_argument("--personas-config", type=Path, required=True,
                        help="configs/personas_*.json (personas_defaults.json for D1/D2)")
    parser.add_argument("--prompt-condition", default="original_persona", choices=sorted(CONDITIONS.keys()),
                        help="original_persona = S1, P1..P4 = S2..S5, D1/D2 with personas_defaults.json")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--image-ids-file", type=Path, default=IMAGE_IDS_100)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--pooling", default="last", choices=["mean", "last"])
    parser.add_argument("--extract-layers", default="all")
    parser.add_argument("--limit", type=int, default=None, help="First N images only (for testing)")
    parser.add_argument("--personas", default=None, help="Comma-separated subset of persona ids")
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    with open(args.personas_config) as f:
        personas = json.load(f)
    if args.personas:
        keep = {p.strip() for p in args.personas.split(",")}
        personas = {k: v for k, v in personas.items() if k in keep}
    if not args.image_dir.exists():
        logger.error(f"Image directory not found: {args.image_dir}")
        return 1

    n_ok = 0
    for name, cfg in personas.items():
        logger.info(f"Extracting: {cfg.get('name', name)}")
        n_ok += extract_persona_embeddings(
            name, cfg, args.output_dir, args.model, args.image_dir, args.device, args.pooling,
            args.extract_layers, args.image_ids_file, args.prompt_condition, args.limit, logger,
        )
    logger.info(f"Extraction complete: {n_ok}/{len(personas)} personas")
    return 0 if n_ok == len(personas) else 1


if __name__ == "__main__":
    sys.exit(main())
