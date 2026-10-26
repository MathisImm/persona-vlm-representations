import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------
# Standard Styles
# ----------------------

# Font Sizes
FS_TITLE = 9
FS_LABEL = 9
FS_TICK = 8
FS_ANNOT = 7
FS_LEGEND = 8

# Colors
STYLE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#000000"]

# Method Styles (consistent across plots)
METHOD_STYLES = {
    'diff_mean':   {'label': 'Diff Mean',   'color': STYLE_COLORS[0]},
    'pca':         {'label': 'PCA (1st)',   'color': STYLE_COLORS[1]},
    'pca_corr':    {'label': 'PCA (Best)',  'color': STYLE_COLORS[5]},
    'probe_clf':   {'label': 'Probe (Clf)', 'color': STYLE_COLORS[2]},
    'probe_reg':   {'label': 'Probe (Reg)', 'color': STYLE_COLORS[3]},
    'sparse_dict': {'label': 'Sparse Dict', 'color': STYLE_COLORS[4]},
}

# Prompt-condition styles. Keys must match
# pipeline/prompt_conditions.CONDITIONS.
# Conditions without a persona (Base/D1/D2) use neutral colours (black, grey,
# olive) and dashed/dotted lines so they read as no-persona at a glance; the
# five prompt styles (S1-S5) each take a colour from the standard palette.
CONDITION_STYLES = {
    'original_baseline': {'label': 'Base (task only)',               'color': '#000000', 'linestyle': ':',  'marker': 'x'},
    'D1':                {'label': 'D1 ("You are an AI")',            'color': '#7f7f7f', 'linestyle': '--', 'marker': 's'},
    'D2':                {'label': 'D2 ("You are a human being")',    'color': '#bcbd22', 'linestyle': '--', 'marker': 'D'},
    'original_persona':  {'label': 'S1 (first-person)',             'color': STYLE_COLORS[1], 'linestyle': '-', 'marker': 'o'},
    'P1':                {'label': 'S2 (third-person)',              'color': STYLE_COLORS[0], 'linestyle': '-', 'marker': 'o'},
    'P2':                {'label': 'S3 (persona after task)',         'color': STYLE_COLORS[2], 'linestyle': '-', 'marker': 'o'},
    'P3':                {'label': 'S4 (explicit override)',          'color': STYLE_COLORS[4], 'linestyle': '-', 'marker': 'o'},
    'P4':                {'label': 'S5 (statistical framing)',        'color': STYLE_COLORS[3], 'linestyle': '-', 'marker': 'o'},
}

# Colormaps
CMAP_CORR = 'YlGnBu' # Diverging/sequential for correlations (unified)
CMAP_DIST = 'viridis' # Sequential for distances/scores

def set_ccn_style():
    """
    Matplotlib style for Cognitive Computational Neuroscience (CCN) conference.
    Optimized for on-screen reading, PDFs, and posters.
    """

    mpl.rcParams.update({

        # ----------------------
        # Font & text
        # ----------------------
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS_LABEL,
        # Math text ($r$, $n$, ...) in the same face as the body text.
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",

        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "axes.labelpad": 3,

        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,

        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,

        # Embed fonts as TrueType (Type 42), not Type 3: journals such as
        # Proc. R. Soc. A reject Type 3 fonts in figures.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # ----------------------
        # Lines & markers
        # ----------------------
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.8,

        # ----------------------
        # Axes
        # ----------------------
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,

        "xtick.direction": "out",
        "ytick.direction": "out",

        # ----------------------
        # Grid (light, readable)
        # ----------------------
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,

        # ----------------------
        # Figure
        # ----------------------
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        # ----------------------
        # Colorblind-safe palette
        # ----------------------
        "axes.prop_cycle": mpl.cycler(
            color=STYLE_COLORS
        ),
    })
