"""Every module imports (no stale paths or missing dependencies)."""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))

ANALYSIS_MODULES = [
    "paths",
    "src.gdv",
    "pipeline.prompt_conditions",
    "pipeline.calculate_persona_gdv",
    "pipeline.compute_projections",
    "pipeline.export_prompts",
]
FIGURE_MODULES = [p.stem for p in (ROOT / "figures").glob("*.py")]
INFERENCE_MODULES = [
    "src.vlm_wrapper",
    "pipeline.extract_persona_text",
    "pipeline.extract_embeddings",
    "pipeline.run_persona_analysis",
]


@pytest.mark.parametrize("name", ANALYSIS_MODULES + FIGURE_MODULES)
def test_import(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", INFERENCE_MODULES)
def test_import_inference(name):
    pytest.importorskip("torch")
    importlib.import_module(name)
