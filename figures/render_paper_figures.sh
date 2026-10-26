#!/bin/bash
# Renders every figure of the persona paper into OUT_DIR.
#   figures/render_paper_figures.sh OUT_DIR
# Reads the data archive at $PERSONA_DATA (default ./data) and, for Fig. 1 only,
# the example image in $PERSONA_IMAGES. Table C1 (variance components and LRTs)
# is printed by plot_rating_analysis_publication.py and saved as OUT_DIR/table_C1_<schema>.txt.
# Both layouts set figures at \linewidth (preprint 5.16 in, Proc. R. Soc. A 5.33 in).
# All figures are drawn at the same width W, so the shared style sizes print at
# the same size in every figure (W = 6.2 in prints at ~84%: 9 pt -> 7.5 pt).
# 6.2 in is the narrowest width at which the setup infographic, the 5 x 5
# projection grid and the 28 x 28 correlation matrix keep all their content.
set -e
OUT=$(realpath -m "${1:?output directory}")
W=6.2
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
P=figures
IDS=configs/image_ids_100_with_hidden_states.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
height() { python3 -c "print(round($1 * $W / 7.0, 3))"; }   # keep each figure's 7-in aspect ratio
mkdir -p "$OUT"
$PY $P/plot_setup_infographic.py --output-dir "$OUT" --width $W                     # Fig. 1
$PY $P/plot_figure1_combined_scatter.py --output-dir "$OUT/_tmp" --fig-width $W     # Fig. 2
$PY $P/plot_figure2_schema_split_scatter.py --output-dir "$OUT/_tmp" --fig-width $W # Fig. 3
cp "$OUT"/_tmp/*/figure1_combined_scatter.pdf "$OUT"/_tmp/*/figure2_schema_split_scatter.pdf "$OUT"/
rm -rf "$OUT/_tmp"
for s in big5 descriptive; do                                                     # Fig. 4 / B1
  $PY $P/plot_f2_per_condition_curves.py --persona-set $s --fig-width $W --output-dir "$OUT"
done
for schema in personas_big5_prototypical personas_prototypical; do
  $PY $P/plot_rating_analysis_publication.py --schema $schema --image-ids-file $IDS \
      --figsize $W $(height 8.3) --output-path "$OUT" | tee "$OUT/table_C1_$schema.txt" # Fig. 5 / B2
  $PY $P/plot_rating_condition_correlations.py --schema $schema --image-ids-file $IDS \
      --figsize $W $(height 7.6) --annot-size 7 --output-path "$OUT"                # Fig. 6 / B3
done
$PY $P/plot_app_style_gdv.py --width $W --output "$OUT/figure_app_style_gdv.pdf"    # Fig. C1
rm -f "$OUT"/*.png
echo "Rendered at $W in -> $OUT"
