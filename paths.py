"""Default locations for all scripts.

Point PERSONA_DATA at the unpacked Zenodo archive (default: ./data) and,
for inference only, PERSONA_IMAGES at the folder holding the Flickr images.
Every default below can also be overridden on the command line.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIGS = REPO_ROOT / "configs"
IMAGE_IDS_100 = CONFIGS / "image_ids_100_with_hidden_states.txt"

DATA_ROOT = Path(os.environ.get("PERSONA_DATA", REPO_ROOT / "data"))
RATINGS = DATA_ROOT / "ratings"
CI_SCORES = DATA_ROOT / "ci_scores_100_uniform.csv"
GDV = DATA_ROOT / "gdv"
PROJECTIONS = DATA_ROOT / "projections"
HIDDEN_STATES = DATA_ROOT / "hidden_states"
IMAGE_DIR = Path(os.environ.get("PERSONA_IMAGES", DATA_ROOT / "images"))

OUTPUT = Path(os.environ.get("PERSONA_OUTPUT", REPO_ROOT / "output"))
