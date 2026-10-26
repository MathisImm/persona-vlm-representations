# Persona and prompt style in a multimodal transformer

Code for *The Geometry of Simulated Perspectives: Dissociating Representation and Behaviour in Multimodal Language Models*
(Immertreu, Abdullahu, Kinfe, Grabner, Krauss and Schilling).

Qwen3-VL-8B-Instruct rates the interestingness of 100 Flickr images under 53 prompts: a task-only baseline (Base), two minimal identity statements (D1 "You are an AI.", D2 "You are a human being.") and five socio-demographic and five Big Five personas, each in five prompting styles (S1–S5). We record the rating and the final-token hidden state of every decoder layer. We then analyse the hidden states with the generalised discrimination value (GDV) and MDS, and the ratings with a variance decomposition, mixed-effects models and correlations between conditions.

The data (ratings, hidden states, GDV tables, projections) are archived on Zenodo: [10.5281/zenodo.23664820](https://doi.org/10.5281/zenodo.23664820).

## Layout

| Folder | Contents |
|---|---|
| `pipeline/` | `prompt_conditions.py` holds the exact wording of every prompt (the single source of truth). `export_prompts.py` writes all 53 prompts to `prompts.csv` in the archive. `extract_persona_text.py` produces the ratings and `run_persona_analysis.py` → `extract_embeddings.py` the hidden states (GPU). `calculate_persona_gdv.py` computes the GDV and `compute_projections.py` the MDS caches (CPU). `run_gdv_and_projections.sh` runs all GDV/projection jobs of the paper. |
| `figures/` | One script per figure. `render_paper_figures.sh` renders all of them and saves Table C1. |
| `analysis/` | `paper_numbers.py`: rating numbers in the text, Table C1 (independent re-computation) and Table C2. `gdv_claims.py`: the layer-wise GDV statements of the Results. |
| `cluster/` | SLURM jobs for the GPU steps (`ratings.slurm`, `hidden_states.slurm`, settings in `env.sh`). |
| `configs/` | Persona definitions and the image list. |
| `src/` | GDV implementation (`gdv.py`) and the Qwen3-VL wrapper (`vlm_wrapper.py`). |
| `tests/` | Byte-level checks of the prompt wording, the `prompts.csv` export and imports of every module. |

## Naming in code and data versus the paper

| Code / data | Paper |
|---|---|
| `<schema>` (no suffix), `original_persona` | style S1 (first person) |
| `<schema>_P1` … `<schema>_P4`, conditions `P1` … `P4` | styles S2 … S5 |
| `persona_1` … `persona_5` | personas P1 … P5 |
| `baseline`, `original_baseline` | Base (task-only prompt) |
| `defaults_D1`, `defaults_D2` | D1, D2 |
| `personas_prototypical` (also "descriptive") | socio-demographic schema |
| `personas_big5_prototypical` (also "Big-5+") | Big Five schema |

A dataset folder is the schema plus the style suffix, e.g. `personas_big5_prototypical_P2` for the Big Five personas in S3. The suffixes `_P1` … `_P4` mark styles, not personas.

GDV datasets: `<schema>_no_baseline` is S1 without Base. `<schema>_P1..P4` are S2–S5, which also contain no Base. `<schema>_all_styles` pools S1–S5. `*_all_conditions_with_defaults` and `combined_prototypical_all_conditions` add Base, D1 and D2 (see `COMBINED_DATASETS` in `pipeline/calculate_persona_gdv.py`).

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # analysis, GDV, projections, figures
pip install -r requirements-inference.txt    # only for re-running the model (GPU, >= 24 GB in fp16)
```

Download `persona-vlm-representations_data.zip` from Zenodo and unpack it in the repository as `./data` (`unzip persona-vlm-representations_data.zip && mv persona-vlm-representations_data data`), or point `PERSONA_DATA` to the unpacked folder. The figures use Arial, as on the paper's machine (`ttf-mscorefonts-installer` on Debian/Ubuntu). Without it, matplotlib falls back to DejaVu Sans and the layout shifts slightly.

The images are not redistributed. They come from the Flickr subset used by Abdullahu and Grabner (2024, *Commonly interesting images*). The 100 identifiers are in `configs/image_ids_100_with_hidden_states.txt`. Point `PERSONA_IMAGES` to a folder holding them as `<image_id>.jpg`. This is needed only for inference and for the example photo in Fig. 1.

## Reproducing the paper

```bash
export PERSONA_DATA=$PWD/data PERSONA_IMAGES=/path/to/images
bash figures/render_paper_figures.sh output/figures   # all 10 figures + Table C1 (about 1 min)
python analysis/paper_numbers.py                       # counts, Base/D1/D2 agreement, Table C1, Table C2 (about 30 s)
python analysis/gdv_claims.py                          # GDV statements of the Results
JOBS=24 bash pipeline/run_gdv_and_projections.sh output   # recompute GDV and projections from the hidden states
pytest                                                 # prompt wording, prompts.csv, imports
```

| Paper | Produced by | Reads |
|---|---|---|
| Fig. 1 (setup) | `figures/plot_setup_infographic.py` | `configs/`, `pipeline/prompt_conditions.py`, one example image |
| Fig. 2 (all 53 prompts, MDS) | `figures/plot_figure1_combined_scatter.py` | `projections/combined_prototypical_all_conditions` |
| Fig. 3 (persona prompts, MDS) | `figures/plot_figure2_schema_split_scatter.py` | `projections/combined_prototypical_all_conditions__schema_split` |
| Fig. 4 / B1 (GDV per layer and style) | `figures/plot_f2_per_condition_curves.py --persona-set big5 / descriptive` | `gdv/<schema>_no_baseline`, `gdv/<schema>_P1..P4`, `projections/<schema>_all_conditions_with_defaults__style_*` |
| Fig. 5 / B2, Table C1 (ratings, variance, LRTs) | `figures/plot_rating_analysis_publication.py` | `ratings/` |
| Fig. 6 / B3 (correlations between conditions) | `figures/plot_rating_condition_correlations.py` | `ratings/` |
| Fig. C1 (pooled styles) | `figures/plot_app_style_gdv.py` | `gdv/<schema>_all_styles` |
| Table C2, cell counts, Base/D1/D2 | `analysis/paper_numbers.py` | `ratings/` |

`render_paper_figures.sh` reproduces the figures of the paper from the archive. `run_gdv_and_projections.sh` recomputes the archived GDV tables and projection caches from the hidden states, up to floating-point rounding.

### Re-running the model

The GPU steps ran on NVIDIA A40 GPUs (NHR@FAU Alex) in fp16 with greedy decoding, model revision `0c351dd`:

```bash
sbatch --export=ALL,RUN=5 cluster/ratings.slurm         # RUN 0..11, see ALL_RUNS in cluster/env.sh
sbatch --export=ALL,RUN=5 cluster/hidden_states.slurm
```

On the A40 GPUs used for the paper, a short test run of the release code reproduced the archived ratings and hidden states. On other hardware or library versions, small numerical differences in the hidden states are expected, and they can occasionally change a response.

Inside SLURM jobs, transformers 4.57 contacts the Hugging Face Hub while loading the model, even when the model is already cached. Setting `HF_HUB_OFFLINE=1` makes loading fail, so the compute node needs web access (see `cluster/env.sh`).

## Notes

- **Image list.** The list was drawn as about ten images from each of ten equal-width bins of the common interestingness (CI) score. It is shipped as data, not regenerated.
- **CI scores.** The persona hidden-state files carry no CI scores. The GDV and projection scripts take them from `ci_scores_100_uniform.csv`. The CI bins define the CI-stratified 20-image projection subset (4 per CI quintile, seed 42).
- **Mixed-effects models.** The models are fitted with three optimisers (lbfgs, powell, Nelder–Mead), keeping the fit with the highest likelihood. The LRT p-values are plain χ² p-values.
- **Base ratings.** The figures, the projections and `paper_numbers.py` take Base from the S2 folders (`<schema>_P1`). Base was rated in the S2–S5 folders with the task-only prompt of `pipeline/prompt_conditions.py` and gave identical responses in all eight, and the release code reproduces them exactly. The S1 folders hold the Base hidden states but no Base ratings.
- **Base hidden states.** The Base hidden states (`hidden_states/<schema>/baseline`, used only in Fig. 2) were extracted with the task-only prompt by `run_persona_analysis.py` and are the same file in both schemas.
- **Load order.** The MDS solution depends on the order of the input points. `calculate_persona_gdv.py` loads the persona folders in a fixed order (`LOAD_ORDER`), the directory order of the original runs, so that the archived projections are reproduced exactly. GDV does not depend on the order.

## Licence

Code: MIT (see `LICENSE`). Data on Zenodo: CC BY 4.0.
