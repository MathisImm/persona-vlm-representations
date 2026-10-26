#!/bin/bash
# Recompute every GDV table and MDS projection cache the paper uses from the
# hidden states in $PERSONA_DATA/hidden_states (CPU only).
#   JOBS=12 pipeline/run_gdv_and_projections.sh [OUT_DIR]      (default: ./output)
# Results go to OUT_DIR/gdv/<dataset>/gdv_results.csv and
# OUT_DIR/projections/<cache>/projections.npz, laid out like the archive.
# The 24 runs are independent; JOBS runs them in parallel (default 1, which
# takes several hours, mostly the GDV of the 2,500-point *_all_styles sets).
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=$(realpath -m "${1:-output}")
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

{
# GDV in the original hidden-state space, decoder layers (Fig. 4, Fig. B1, Fig. C1).
# S1 = *_no_baseline, S2..S5 = *_P1.._P4 (five personas each); *_all_styles pools S1..S5.
for schema in personas_big5_prototypical personas_prototypical; do
  for ds in ${schema}_all_styles ${schema}_no_baseline ${schema}_P1 ${schema}_P2 ${schema}_P3 ${schema}_P4; do
    echo "gdv_$ds pipeline/calculate_persona_gdv.py --dataset $ds --output-dir $OUT/gdv/$ds"
  done
done
# MDS caches on the CI-stratified 20-image subset (seed 42), decoder layers only.
P="pipeline/compute_projections.py --max-images 20 --cache-dir $OUT/projections --output-dir $OUT/projection_overviews"
echo "proj_fig2 $P --dataset combined_prototypical_all_conditions --colorings promptstyle"
echo "proj_fig3 $P --dataset combined_prototypical_all_conditions --persona-schemas-only --colorings schema --cmap Set1"
for schema in personas_big5_prototypical personas_prototypical; do       # Fig. 4 / B1 projection panels
  for style in original_persona P1 P2 P3 P4; do
    echo "proj_${schema}_$style $P --dataset ${schema}_all_conditions_with_defaults --colorings promptstyle --promptstyle-only $style"
  done
done
} | xargs -P "${JOBS:-1}" -L 1 sh -c 'name=$0; '"$PY"' "$@" > '"$OUT"'/logs/$name.log 2>&1 || { echo "FAILED: $name (see logs/$name.log)"; exit 255; }'
echo "GDV and projections written to $OUT"
