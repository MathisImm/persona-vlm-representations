#!/usr/bin/env python3
"""
Publication Plot: Rating Analysis (100 Images, uniform sample) with PromptStyle Factor

Layout:
- Panel a (Top): Rating Distributions (Personas + Ensemble), grouped by PromptStyle; Defaults (Base / D1 / D2) prepended
- Panel b (Bottom, full width): full 3-way variance decomposition (Image / Persona / PromptStyle,
  their three two-way interactions, and the three-way interaction as residual)

Uses the ratings/ folders of the archive (100 images shared across conditions),
default schema = personas_prototypical (Descriptive).
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import argparse
import re
from pathlib import Path
from scipy import stats
from tqdm import tqdm

# Repo root (paths.py); style_guide sits next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CI_SCORES, CONFIGS, OUTPUT, RATINGS
try:
    from style_guide import set_ccn_style, STYLE_COLORS, CONDITION_STYLES, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND
except ImportError:
    def set_ccn_style():
        sns.set_theme(style="whitegrid")
    STYLE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#000000"]
    CONDITION_STYLES = {}
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 10, 9, 8, 8

# Default ordering for promptstyles in plots (persona conditions). Defaults
# (D1, D2) are handled separately in panel (a).
PROMPTSTYLE_ORDER = ['original_persona', 'P1', 'P2', 'P3', 'P4']

# S1..S5 promptstyle display convention shared with Figures 1/2/5
# (figures/_scatter_common.py: PROMPTSTYLE_DISPLAY / _DESCR /
# promptstyle_label). Defaults (Base/D1/D2) keep their own
# CONDITION_STYLES labels and are not part of this S-code scheme.
PROMPTSTYLE_CODE = {s: f'S{i + 1}' for i, s in enumerate(PROMPTSTYLE_ORDER)}
PROMPTSTYLE_DESCR = {
    'original_persona': 'first-person',
    'P1': 'third-person',
    'P2': 'persona after task',
    'P3': 'explicit override',
    'P4': 'statistical framing',
}

# Default root of the per-style ratings folders
UNIFORM_RATINGS_ROOT = RATINGS
UNIFORM_CI_PATH = CI_SCORES

# Mapping of interestingness text to numeric values (1-5)
RATING_MAP = {
    "Not Interesting": 1,
    "Slightly Interesting": 2,
    "Moderately Interesting": 3,
    "Very Interesting": 4,
    "Extremely Interesting": 5
}

def extract_rating_from_json(raw_text):
    """Parse JSON and extract numeric rating."""
    try:
        # Clean up possible Markdown code blocks
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_text)
        text = data.get("interestingness", "")
        # Remove punctuation for matching
        text_clean = re.sub(r"[^a-zA-Z ]", "", text).strip()
        return RATING_MAP.get(text_clean, np.nan)
    except:
        return np.nan


def get_ci_map(ci_path=UNIFORM_CI_PATH):
    """Load the CI scores of the 100 images."""
    df = pd.read_csv(ci_path)
    mapping = dict(zip(df['image_id'], df['ci_score']))
    print(f"Loaded CI mapping for {len(mapping)} unique images from {ci_path.name}")
    return mapping


def _folder_to_schema_style(folder_name):
    """Map a ratings sub-folder name to (schema, promptstyle).

    Returns (schema, persona_style). For persona conditions, persona_style is
    what persona_id != 'baseline' rows get assigned. Baseline rows are always
    mapped to 'original_baseline' downstream.
    """
    if folder_name == 'defaults_D1':
        return 'defaults', 'D1'
    if folder_name == 'defaults_D2':
        return 'defaults', 'D2'
    m = re.match(r'^(.+?)(?:_P([1-4]))?$', folder_name)
    if not m:
        return folder_name, 'original_persona'
    schema, p_num = m.group(1), m.group(2)
    if p_num is None:
        return schema, 'original_persona'
    return schema, f'P{p_num}'


def filter_image_ids(df, ids_file):
    """Keep only the images listed (one id per line) in ids_file, if given.

    Used to restrict the rating analyses to the images for which hidden
    states exist, so ratings and representations share one image set."""
    if not ids_file:
        return df
    ids = {l.strip() for l in open(ids_file) if l.strip()}
    out = df[df['image_id'].astype(str).isin(ids)].copy()
    print(f"Image filter: {df['image_id'].nunique()} -> {out['image_id'].nunique()} images")
    return out


def load_uniform_dataset(schema, ratings_root=UNIFORM_RATINGS_ROOT,
                          include_defaults=True, config_dir=None):
    """Load all promptstyle variants for a single schema from ratings/.

    Returns a long-format DataFrame with columns:
        image_id, persona_id, persona_name, raw_response, rating,
        ci_score, schema, promptstyle, display_name

    Baseline rows are taken from the S2 folder (<schema>_P1) only. Base was
    rated in the S2-S5 folders, with identical responses.
    Defaults D1/D2 are appended with schema='defaults' and promptstyle in
    {'D1', 'D2'} if include_defaults=True.
    """
    ratings_root = Path(ratings_root)
    ci_map = get_ci_map()

    # Find all folders matching this schema (no suffix + _P1..P4).
    candidates = [schema] + [f'{schema}_P{i}' for i in range(1, 5)]
    folders = [f for f in candidates if (ratings_root / f).exists()]
    if not folders:
        raise FileNotFoundError(f"No folders for schema {schema} in {ratings_root}")

    # Load persona descriptions for display labels.
    descriptions = {}
    if config_dir is None:
        config_dir = CONFIGS
    cfg_path = Path(config_dir) / f'{schema}.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
            for p_id, info in cfg.items():
                descriptions[p_id] = info.get('description', '')

    frames = []
    for folder in folders:
        sch, style = _folder_to_schema_style(folder)
        csv_path = ratings_root / folder / 'persona_text_outputs.csv'
        df = pd.read_csv(csv_path)
        df['rating'] = df['raw_response'].apply(extract_rating_from_json)
        df = df.dropna(subset=['rating'])
        df['schema'] = sch
        # Baseline rows always map to original_baseline and come from the
        # S2 folder alone (see docstring); the other folders' copies are dropped.
        baseline_mask = df['persona_id'] == 'baseline'
        if folder == f'{schema}_P1':
            df.loc[baseline_mask, 'promptstyle'] = 'original_baseline'
            df.loc[~baseline_mask, 'promptstyle'] = style
        else:
            df = df.loc[~baseline_mask].copy()
            df['promptstyle'] = style
        frames.append(df)

    if include_defaults:
        for d in ('defaults_D1', 'defaults_D2'):
            d_path = ratings_root / d / 'persona_text_outputs.csv'
            if not d_path.exists():
                continue
            ddf = pd.read_csv(d_path)
            ddf['rating'] = ddf['raw_response'].apply(extract_rating_from_json)
            ddf = ddf.dropna(subset=['rating'])
            sch, style = _folder_to_schema_style(d)
            ddf['schema'] = sch
            ddf['promptstyle'] = style
            frames.append(ddf)

    out = pd.concat(frames, ignore_index=True)
    out['ci_score'] = out['image_id'].map(ci_map)
    out = out.dropna(subset=['ci_score'])
    out['display_name'] = out.apply(
        lambda r: get_persona_label(r['persona_id'], r['persona_name'],
                                    descriptions.get(r['persona_id'], ''),
                                    schema), axis=1)
    print(f"Loaded {len(out)} rows: schema={schema}, "
          f"{out['image_id'].nunique()} images, "
          f"{out['persona_id'].nunique()} personas, "
          f"{out['promptstyle'].nunique()} styles "
          f"({sorted(out['promptstyle'].unique())})")
    return out

def get_persona_label(p_id, p_name, description, dataset_name):
    """Determine label (number or age) for a persona."""
    if p_id == 'baseline':
        return 'Base'
    
    # User feedback: Use age for variation/quantitative datasets
    if 'age_variation' in dataset_name or 'quantitative' in dataset_name:
        age_match = re.search(r"Age:\s*(\d+)", description)
        if age_match:
            return age_match.group(1)
            
    # Prototypical schemas: "P1".."P5", matching the paper's notation (personas
    # are P1-P5 and prompt styles are S1-S5, see _scatter_common.PROMPTSTYLE_DISPLAY).
    # A bare digit here collided with the style codes in Figure 4a's x-axis and
    # Figure 5b's tick labels.
    match = re.search(r"persona_(\d+)", p_id)
    if not match:
        match = re.search(r"Persona\s+(\d+)", p_name)
    if match:
        return f"P{match.group(1)}"


    return p_name


VAR_COMPONENTS = ['Image', 'Persona', 'PromptStyle',
                  'Persona x PromptStyle', 'Image x Persona',
                  'Image x PromptStyle', 'Residual']


def compute_variance_decomposition(df):
    """Full 3-factor variance decomposition by sum of squares.

    Restricted to rows where persona_id != 'baseline' and promptstyle is in
    the persona-condition set (original_persona + P1..P4). Decomposes total
    SS into the three main effects (Image, Persona, PromptStyle), the three
    two-way interactions, and Residual. With one observation per
    (image, persona, style) cell the residual is the three-way interaction
    (plus any LLM stochasticity, negligible under greedy decoding).
    """
    persona_styles = ['original_persona', 'P1', 'P2', 'P3', 'P4']
    sub = df[(df['persona_id'] != 'baseline') &
             (df['promptstyle'].isin(persona_styles))]
    if len(sub) == 0:
        return None

    y = sub['rating'].to_numpy(dtype=float)
    grand = y.mean()
    total_ss = ((y - grand) ** 2).sum()
    if total_ss == 0:
        return None

    def _main_ss(col):
        means = sub.groupby(col)['rating'].mean()
        counts = sub.groupby(col)['rating'].count()
        return float(((means - grand) ** 2 * counts).sum())

    image_ss = _main_ss('image_id')
    persona_ss = _main_ss('persona_id')
    style_ss = _main_ss('promptstyle')

    # Persona x PromptStyle interaction:
    # SS(P,S) - SS(P) - SS(S), where SS(P,S) is SS of cell means.
    def _cells_ss(cols):
        cell_means = sub.groupby(cols)['rating'].mean()
        cell_n = sub.groupby(cols)['rating'].count()
        return float(((cell_means - grand) ** 2 * cell_n).sum())

    interaction_ss = max(0.0, _cells_ss(['persona_id', 'promptstyle'])
                         - persona_ss - style_ss)
    img_persona_ss = max(0.0, _cells_ss(['image_id', 'persona_id'])
                         - image_ss - persona_ss)
    img_style_ss = max(0.0, _cells_ss(['image_id', 'promptstyle'])
                       - image_ss - style_ss)

    residual_ss = max(0.0, total_ss - image_ss - persona_ss - style_ss
                      - interaction_ss - img_persona_ss - img_style_ss)

    return {
        'Image':                  image_ss / total_ss,
        'Persona':                persona_ss / total_ss,
        'PromptStyle':            style_ss / total_ss,
        'Persona x PromptStyle':  interaction_ss / total_ss,
        'Image x Persona':        img_persona_ss / total_ss,
        'Image x PromptStyle':    img_style_ss / total_ss,
        'Residual':               residual_ss / total_ss,
    }


def compute_all_stats(df):
    """Variance decomposition (per-style correlations are computed
    separately, by plot_rating_analysis_consolidated.py, and no longer
    rendered by this script)."""
    return compute_variance_decomposition(df)

def compute_mixed_effects(df):
    """Mixed-effects model:
    rating ~ C(persona) * C(promptstyle)
             + (1|image_id) + (1|image_id:persona) + (1|image_id:promptstyle).

    The image x persona and image x style random intercepts carry the
    image-specific persona/style effects, so the fixed-effect tests are not
    inflated by them. Fits the full model and nested submodels and computes
    likelihood-ratio tests for the Persona, PromptStyle and
    Persona x PromptStyle terms, plus tests for the two image interaction
    variances (plain chi2(1), i.e. conservative at the zero boundary).
    Returns a dict of {term: {df, chi2, p}} plus the random-effect variance.
    Restricted to persona-condition rows (excludes baseline + defaults).
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("[warn] statsmodels not installed; skipping mixed-effects fit")
        return None

    persona_styles = ['original_persona', 'P1', 'P2', 'P3', 'P4']
    sub = df[(df['persona_id'] != 'baseline') &
             (df['promptstyle'].isin(persona_styles))].copy()
    if len(sub) < 50:
        return None
    sub['persona'] = sub['persona_id'].astype('category')
    sub['style'] = sub['promptstyle'].astype('category')

    import warnings
    vc_all = {'img_persona': '0 + C(persona)', 'img_style': '0 + C(style)'}

    def _fit(formula, vc=vc_all):
        m = smf.mixedlm(formula, data=sub, groups=sub['image_id'],
                        re_formula='1', vc_formula=vc)
        # Run several optimizers and keep the best likelihood: a single
        # optimizer can stop early without raising (lbfgs did so for the
        # additive Big-5 model), which would bias the likelihood-ratio tests.
        best = None
        for opt in ('lbfgs', 'powell', 'nm'):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r = m.fit(reml=False, method=opt, disp=False, maxiter=5000)
            except (np.linalg.LinAlgError, Exception):
                continue
            if best is None or r.llf > best.llf:
                best = r
        return best

    full = _fit("rating ~ C(persona) * C(style)")
    no_int = _fit("rating ~ C(persona) + C(style)")
    no_style = _fit("rating ~ C(persona)")
    no_persona = _fit("rating ~ C(style)")
    no_ixp = _fit("rating ~ C(persona) * C(style)",
                  {'img_style': vc_all['img_style']})
    no_ixs = _fit("rating ~ C(persona) * C(style)",
                  {'img_persona': vc_all['img_persona']})
    if any(m is None for m in (full, no_int, no_style, no_persona,
                               no_ixp, no_ixs)):
        print("[warn] One or more mixed-effects fits failed; skipping LRT.")
        return None

    n_p = sub['persona'].nunique()
    n_s = sub['style'].nunique()

    def _lrt(full_m, reduced_m, df_diff):
        chi2 = 2 * (full_m.llf - reduced_m.llf)
        p = 1 - stats.chi2.cdf(chi2, df_diff)
        return float(chi2), float(p), int(df_diff)

    chi2_i, p_i, d_i = _lrt(full, no_int, (n_p - 1) * (n_s - 1))
    # Style main effect = improvement from adding style to a persona-only model:
    chi2_s, p_s, d_s = _lrt(no_int, no_style, n_s - 1)
    # Persona main effect = improvement from adding persona to a style-only model:
    chi2_p, p_p, d_p = _lrt(no_int, no_persona, n_p - 1)
    # Variance components: plain chi2(1) p-values, conservative because the
    # null (variance = 0) lies on the boundary; all are far below 0.001.
    chi2_ip, p_ip, d_ip = _lrt(full, no_ixp, 1)
    chi2_is, p_is, d_is = _lrt(full, no_ixs, 1)

    re_var = float(full.cov_re.iloc[0, 0]) if full.cov_re is not None else float('nan')
    sigma2 = float(full.scale)

    return {
        'Persona':              {'df': d_p, 'chi2': chi2_p, 'p': p_p},
        'PromptStyle':          {'df': d_s, 'chi2': chi2_s, 'p': p_s},
        'Persona x PromptStyle':{'df': d_i, 'chi2': chi2_i, 'p': p_i},
        'Image x Persona':      {'df': d_ip, 'chi2': chi2_ip, 'p': p_ip},
        'Image x PromptStyle':  {'df': d_is, 'chi2': chi2_is, 'p': p_is},
        'image_var': re_var,
        'residual_var': sigma2,
    }


def run_bootstrapping(df, n_iterations=500, ci_level=0.95, seed=42):
    """Image-level bootstrap for variance decomposition and per-style correlations.

    `seed` makes the reported CIs reproducible: the resampling used to draw the
    global RNG, so two runs on identical data gave different intervals.
    """
    print(f"Running bootstrapping ({n_iterations} iterations, CI={ci_level}, seed={seed})...")
    rng = np.random.default_rng(seed)
    unique_images = df['image_id'].unique()
    grouped_df = {img_id: group for img_id, group in df.groupby('image_id')}

    var_samples = []

    for j_iter in tqdm(range(n_iterations)):
        sampled_img_ids = rng.choice(unique_images,
                                     size=len(unique_images), replace=True)
        sample_pieces = []
        for j, img_id in enumerate(sampled_img_ids):
            piece = grouped_df[img_id].copy()
            # Rename so each draw is treated as its own image in groupbys
            piece['image_id'] = f"img_{j}_{img_id}"
            sample_pieces.append(piece)
        sample_df = pd.concat(sample_pieces, ignore_index=True)

        v = compute_all_stats(sample_df)
        if v is not None:
            var_samples.append(v)

    # Variance components CI
    var_summary = {}
    for key in VAR_COMPONENTS:
        vals = [s[key] for s in var_samples]
        var_summary[key] = {
            'mean': float(np.mean(vals)),
            'low': float(np.percentile(vals, (1 - ci_level) / 2 * 100)),
            'high': float(np.percentile(vals, (1 + ci_level) / 2 * 100)),
        }

    return var_summary

def _persona_order(df):
    """Return persona display names in numeric order (baseline/no-persona excluded)."""
    persona_df = df[~df['persona_id'].isin(['baseline', 'no_persona'])]
    rows = (persona_df.groupby('persona_id')['display_name']
            .first().reset_index())
    def key(p_id):
        m = re.search(r'(\d+)', p_id)
        return int(m.group(1)) if m else 999
    rows['sort'] = rows['persona_id'].apply(key)
    rows = rows.sort_values('sort')
    return rows['display_name'].tolist(), rows['persona_id'].tolist()


def plot_publication_figure(df, var_summary, me_results, args):
    """Render the 2-panel rating-analysis figure with promptstyle as a hue."""
    set_ccn_style()

    persona_display, persona_ids = _persona_order(df)
    persona_styles = PROMPTSTYLE_ORDER

    fig = plt.figure(figsize=args.figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1])

    # --- Panel a: Rating distributions split by promptstyle ---
    ax_dist = fig.add_subplot(gs[0])

    # Build long-format dataframe with three groups:
    # Defaults (Base / D1 / D2), the persona rows, and Ensemble.
    dist_rows = []
    # Defaults: baseline ratings (style=original_baseline) + D1 + D2.
    base_styles = ['original_baseline', 'D1', 'D2']
    base_x_labels = ['Base', 'D1', 'D2']
    for x_lab, style in zip(base_x_labels, base_styles):
        sub = df[df['promptstyle'] == style]
        for r in sub['rating']:
            dist_rows.append({'Source': x_lab, 'Style': style, 'Rating': r})

    # Per-persona ratings, hue=promptstyle.
    for p_lab, p_id in zip(persona_display, persona_ids):
        for style in persona_styles:
            sub = df[(df['persona_id'] == p_id) & (df['promptstyle'] == style)]
            for r in sub['rating']:
                dist_rows.append({'Source': p_lab, 'Style': style, 'Rating': r})

    # Ensemble: mean rating per (image, style).
    persona_df = df[(df['persona_id'] != 'baseline') &
                    (df['promptstyle'].isin(persona_styles))]
    for style in persona_styles:
        ens = (persona_df[persona_df['promptstyle'] == style]
               .groupby('image_id')['rating'].mean())
        for r in ens:
            dist_rows.append({'Source': 'Ensemble', 'Style': style, 'Rating': r})

    dist_df = pd.DataFrame(dist_rows)
    source_order = base_x_labels + persona_display + ['Ensemble']
    dist_df['Source'] = pd.Categorical(dist_df['Source'],
                                       categories=source_order, ordered=True)
    # Style palette: defaults use their own colors; persona styles share palette.
    all_styles_in_panel = base_styles + persona_styles
    style_to_color = {s: CONDITION_STYLES.get(s, {}).get('color', STYLE_COLORS[i % 7])
                      for i, s in enumerate(all_styles_in_panel)}

    # Two violin layers: defaults (no hue, color per source) + persona/ensemble (hue=style).
    # Simpler: plot all in one pass with hue=Style ordered by all_styles_in_panel.
    dist_df['Style'] = pd.Categorical(dist_df['Style'],
                                      categories=all_styles_in_panel, ordered=True)
    palette = [style_to_color[s] for s in all_styles_in_panel]
    sns.violinplot(data=dist_df, x='Source', y='Rating', hue='Style',
                   ax=ax_dist, inner=None, linewidth=0.6, palette=palette,
                   dodge=True, cut=0, density_norm='width',
                   hue_order=all_styles_in_panel, order=source_order)
    # Legend handling: keep only persona styles + defaults (concise).
    # Promptstyles (original_persona/P1-P4) use the S1-S5 display convention
    # shared with Figures 1/2/5; Base/D1/D2 keep their own labels.
    handles, labels_l = ax_dist.get_legend_handles_labels()
    keep_labels = base_styles + persona_styles
    pretty = {s: CONDITION_STYLES.get(s, {}).get('label', s)
              for s in base_styles}
    pretty.update({s: f"{PROMPTSTYLE_CODE[s]} ({PROMPTSTYLE_DESCR[s]})"
                  for s in persona_styles})
    new_handles = [h for h, l in zip(handles, labels_l) if l in keep_labels]
    new_labels = [pretty[l] for l in labels_l if l in keep_labels]
    legend = ax_dist.legend(new_handles, new_labels, fontsize=FS_LEGEND - 1,
                            ncol=4, loc='upper center',
                            bbox_to_anchor=(0.5, args.legend_y), frameon=True)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(0.8)
    ax_dist.set_ylabel("Rating", fontsize=FS_LABEL)
    ax_dist.set_xlabel("Persona", fontsize=FS_LABEL)
    ax_dist.set_ylim(0.5, 5.5)
    # Visual separator between Defaults group, Personas, and Ensemble.
    ax_dist.axvline(len(base_x_labels) - 0.5, color='gray', lw=0.6, ls=':', alpha=0.6)
    ax_dist.axvline(len(base_x_labels) + len(persona_display) - 0.5,
                    color='gray', lw=0.6, ls=':', alpha=0.6)

    # --- Panel b: 7-component variance decomposition (full figure width) ---
    ax_var = fig.add_subplot(gs[1])

    labels_v = VAR_COMPONENTS
    means_v = [var_summary[k]['mean'] * 100 for k in labels_v]
    lows_v = [var_summary[k]['low'] * 100 for k in labels_v]
    highs_v = [var_summary[k]['high'] * 100 for k in labels_v]
    yerr_v = [[means_v[i] - lows_v[i] for i in range(len(labels_v))],
              [highs_v[i] - means_v[i] for i in range(len(labels_v))]]
    # Main effects and P×S keep their colours; the two image interactions
    # take muted tones, and the three-way residual stays light grey.
    bar_colors = [STYLE_COLORS[1], STYLE_COLORS[0], STYLE_COLORS[2],
                  STYLE_COLORS[4], STYLE_COLORS[5], '#a6a6a6', 'lightgray']
    short_labels = ['Image', 'Persona', 'Style', 'P×S', 'I×P', 'I×S', 'I×P×S']
    bars = ax_var.bar(short_labels, means_v, yerr=yerr_v, color=bar_colors,
                      width=0.5,
                      error_kw={'capsize': 3, 'elinewidth': 1.0,
                                'ecolor': 'black'})
    ax_var.set_ylabel("Variance Explained (%)", fontsize=FS_LABEL)
    # Same fixed range for both schemas so Fig. 5B and C2B compare directly;
    # widened only if a whisker would otherwise be clipped.
    ax_var.set_ylim(0, max(args.var_ymax, max(highs_v) + 8))
    ax_var.xaxis.grid(False)
    ax_var.yaxis.grid(True, alpha=0.3)
    ax_var.tick_params(axis='x', labelsize=FS_TICK + 1)
    for i, bar in enumerate(bars):
        ax_var.text(bar.get_x() + bar.get_width() / 2, highs_v[i] + 1,
                    f'{means_v[i]:.1f}%', ha='center', va='bottom',
                    fontsize=FS_TICK)

    # Mixed-effects summary line. The chi2 values live in the appendix table;
    # the panel only states whether every testable bar (Persona/Style/P×S
    # fixed, I×P/I×S variance) is significant, and lists any that is not.
    if me_results is not None:
        terms = [('Persona', 'Persona'), ('Style', 'PromptStyle'),
                 ('P×S', 'Persona x PromptStyle'), ('I×P', 'Image x Persona'),
                 ('I×S', 'Image x PromptStyle')]
        not_sig = [f"{term} p={me_results[key]['p']:.2g}"
                   for term, key in terms if me_results[key]['p'] >= 0.001]
        text = "Mixed-effects LRT: all testable terms p < 0.001"
        if not_sig:
            text = ("Mixed-effects LRT: p < 0.001 except "
                    + ", ".join(not_sig))
        ax_var.text(0.97, 0.96, text,
                    transform=ax_var.transAxes, fontsize=FS_ANNOT_FALLBACK + 1,
                    va='top', ha='right',
                    bbox=dict(facecolor='white', edgecolor='gray',
                              boxstyle='round,pad=0.3', alpha=0.9))

    plt.subplots_adjust(left=args.left, right=args.right, top=args.top,
                        bottom=args.bottom, hspace=args.hspace)

    # Shift panel b) right (narrower, same right edge) and add panel labels
    # in figure coordinates so a) and b) line up on the same x position
    # regardless of each axes' own left margin.
    pos_var = ax_var.get_position()
    ax_var.set_position([pos_var.x0 + args.panel_b_shift, pos_var.y0,
                         pos_var.width - args.panel_b_shift, pos_var.height])

    pos_dist = ax_dist.get_position()
    pos_var = ax_var.get_position()
    label_x = args.label_x_fig
    fig.text(label_x, pos_dist.y1 + 0.015, "A", fontsize=FS_TITLE + 1,
             fontweight='bold', va='bottom', ha='left')
    fig.text(label_x, pos_var.y1 + 0.015, "B", fontsize=FS_TITLE + 1,
             fontweight='bold', va='bottom', ha='left')

    plt.savefig(f"{args.output_path}.png", dpi=300)
    plt.savefig(f"{args.output_path}.pdf")
    print(f"Saved publication plot to {args.output_path}.png/pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Rating Analysis Publication Plot (100 images, '
                    'with promptstyle factor)')
    parser.add_argument('--schema', type=str, default='personas_prototypical',
                        help='Schema folder name in ratings/ '
                             '(default: personas_prototypical)')
    parser.add_argument('--ratings-root', type=Path,
                        default=UNIFORM_RATINGS_ROOT)
    parser.add_argument('--include-defaults', action='store_true', default=True)
    parser.add_argument('--no-defaults', dest='include_defaults',
                        action='store_false')
    parser.add_argument('--output-path', type=Path,
                        default=OUTPUT / "figures")
    parser.add_argument('--image-ids-file', type=Path, default=None,
                        help='Restrict all analyses to these image ids (one per line).')
    parser.add_argument('--n-bootstrap', type=int, default=500)
    parser.add_argument('--ci-level', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the image-level bootstrap resampling')
    parser.add_argument('--skip-mixed-effects', action='store_true')

    # Layout parameters
    parser.add_argument('--label-x-fig', type=float, default=0.01,
                        help='Shared figure-fraction x for a)/b) panel labels')
    parser.add_argument('--legend-y', type=float, default=-0.17,
                        help='bbox_to_anchor y for the panel-a legend '
                             '(more negative = further below the x-label)')
    parser.add_argument('--panel-b-shift', type=float, default=0.05,
                        help='Figure-fraction amount to shift panel b) '
                             'right (its right edge stays fixed)')
    parser.add_argument('--hspace', type=float, default=0.5)
    parser.add_argument('--var-ymax', type=float, default=60,
                        help='Upper y limit (%%) of the variance panel.')
    parser.add_argument('--figsize', type=float, nargs=2, default=[7, 8.3],
                        help='Paper figures use a 7-inch standard width '
                             '(see age_vc.py FIG_WIDTH / '
                             'plot_persona_agreement_master.py)')
    parser.add_argument('--left', type=float, default=0.09)
    parser.add_argument('--right', type=float, default=0.99)
    parser.add_argument('--top', type=float, default=0.96)
    parser.add_argument('--bottom', type=float, default=0.13)

    args = parser.parse_args()

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_path = out_dir / f"figure4_rating_analysis_publication_{args.schema}"

    df = load_uniform_dataset(args.schema, ratings_root=args.ratings_root,
                              include_defaults=args.include_defaults)
    df = filter_image_ids(df, args.image_ids_file)

    var_summary = run_bootstrapping(
        df, n_iterations=args.n_bootstrap, ci_level=args.ci_level, seed=args.seed)
    for key in VAR_COMPONENTS:
        s = var_summary[key]
        print(f"  {key:<24} {100 * s['mean']:5.1f} "
              f"[{100 * s['low']:.1f}, {100 * s['high']:.1f}]")

    if args.skip_mixed_effects:
        me_results = None
    else:
        print("Fitting mixed-effects model...")
        me_results = compute_mixed_effects(df)
        if me_results is not None:
            for key in ('Persona', 'PromptStyle', 'Persona x PromptStyle',
                        'Image x Persona', 'Image x PromptStyle'):
                r = me_results[key]
                print(f"  {key:<22} χ²({r['df']})={r['chi2']:.2f}, p={r['p']:.3g}")

    plot_publication_figure(df, var_summary, me_results, args)


# Small font-size fallback for the mixed-effects inset (named distinctly so
# style_guide changes don't silently break this caption).
try:
    from style_guide import FS_ANNOT as FS_ANNOT_FALLBACK
except ImportError:
    FS_ANNOT_FALLBACK = 7


if __name__ == "__main__":
    main()
