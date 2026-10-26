#!/bin/bash
#SBATCH --job-name=persona_ratings
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=08:00:00
# Interestingness ratings for one run (RUN=0..11, see ALL_RUNS in env.sh) or for
# all runs (RUN unset). The paper ran on NHR@FAU Alex (NVIDIA A40), fp16, greedy.
# Submit from the repository root:
#   sbatch --export=ALL,RUN=5 cluster/ratings.slurm
set -e
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
runs=("${ALL_RUNS[@]}"); [[ -n "$RUN" ]] && runs=("${ALL_RUNS[$RUN]}")
for r in "${runs[@]}"; do
    set -- $r
    python pipeline/extract_persona_text.py --model "$MODEL_ID" \
        --personas-config "configs/$1.json" --prompt-condition "$2" \
        --image-ids-file "${IMAGE_IDS_FILE:-configs/image_ids_100_with_hidden_states.txt}" \
        --image-dir "$PERSONA_IMAGES" --output-dir "$OUT/ratings/$(dataset_name "$1" "$2")" \
        --device cuda --dtype float16 --max-new-tokens 128
done
