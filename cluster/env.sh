# Settings shared by the SLURM jobs in this folder. Adjust to your cluster.
#   REPO            checkout of this repository
#   VENV            Python environment with requirements.txt + requirements-inference.txt
#   PERSONA_IMAGES  folder with the Flickr images (<image_id>.jpg)
#   OUT             where ratings/ and hidden_states/ are written
REPO="${REPO:-$HOME/persona-vlm-representations}"
VENV="${VENV:-$REPO/.venv}"
export PERSONA_IMAGES="${PERSONA_IMAGES:-$REPO/data/images}"
OUT="${OUT:-$REPO/output}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
# transformers 4.57 queries the Hugging Face Hub while loading even when the model is
# cached (HF_HUB_OFFLINE=1 makes loading fail), so compute nodes need web access.
# On NHR@FAU: export http_proxy=http://proxy:80 https_proxy=http://proxy:80

source "$VENV/bin/activate"
cd "$REPO"

# Paper runs: persona set (JSON in configs/) x prompt condition -> dataset folder.
#   original_persona = S1 (no suffix), P1..P4 = S2..S5, D1/D2 = minimal identity statements.
#   The 'baseline' entry of each persona JSON (Base) always uses the task-only prompt.
dataset_name() {  # dataset_name <personas_json_stem> <condition>
    case "$2" in
        original_persona) echo "$1" ;;
        D1|D2)            echo "defaults_$2" ;;
        *)                echo "$1_$2" ;;
    esac
}
ALL_RUNS=(
    "personas_prototypical original_persona" "personas_prototypical P1" "personas_prototypical P2"
    "personas_prototypical P3" "personas_prototypical P4"
    "personas_big5_prototypical original_persona" "personas_big5_prototypical P1" "personas_big5_prototypical P2"
    "personas_big5_prototypical P3" "personas_big5_prototypical P4"
    "personas_defaults D1" "personas_defaults D2"
)
