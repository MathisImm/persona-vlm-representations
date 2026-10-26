#!/bin/bash
#SBATCH --job-name=persona_hidden_states
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=08:00:00
# Final-token hidden states of every decoder layer for one run (RUN=0..11, see
# ALL_RUNS in env.sh) or for all runs (RUN unset), on the 100 images.
# Submit from the repository root:
#   sbatch --export=ALL,RUN=5 cluster/hidden_states.slurm
set -e
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
runs=("${ALL_RUNS[@]}"); [[ -n "$RUN" ]] && runs=("${ALL_RUNS[$RUN]}")
for r in "${runs[@]}"; do
    set -- $r
    python pipeline/run_persona_analysis.py --model "$MODEL_ID" \
        --personas-config "configs/$1.json" --prompt-condition "$2" \
        --image-ids-file "${IMAGE_IDS_FILE:-configs/image_ids_100_with_hidden_states.txt}" \
        --image-dir "$PERSONA_IMAGES" --output-dir "$OUT/hidden_states/$(dataset_name "$1" "$2")" \
        --device cuda --pooling last --extract-layers all ${PERSONAS:+--personas "$PERSONAS"}
done
