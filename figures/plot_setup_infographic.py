#!/usr/bin/env python3
"""Figure 1 (setup overview, file figure6_setup_overview.pdf) for the persona paper.

Draws the full design space of the experiment:

  1. both persona schemas (Descriptive + Big-5), 5 personas each (P1..P5),
  2. a colour key defining the three prompt blocks, then the three no-persona
     default styles (Base/D1/D2) and the five persona styles (S1..S5) as
     block sequences, so the *structural* difference between them (where the
     persona sits relative to the task) is visible at a glance. The blocks the
     rows share are defined once in the key: the {Persona} field template, and
     the task text with its one persona-only clause marked inline rather than
     the whole ~90-word block printed twice,
  3. the Qwen3-VL pipeline with the LLM layers the analyses read from,
  4. the three read-outs.

Sized for a 7 inch (full text-width) paper figure.

All content is pulled from the sources the experiment itself uses, so the
figure cannot drift from the code:

  configs/personas_prototypical.json                        (schema A; also
      supplies the {Persona} field template)
  configs/personas_big5_prototypical.json                   (schema B)
  pipeline/prompt_conditions.py   (prompt styles; the
      task text is *taken* from the registry and the two variants are diffed
      at render time, never retyped)
  figures/_scatter_common.py                 (S1..S5 / P1..P5
      display convention, persona hue palette, extraction layers)
  figures/style_guide.py                                   (condition colours)

Usage::

    python figures/plot_setup_infographic.py
    python figures/plot_setup_infographic.py --output-dir some/dir
"""
from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.append(str(Path(__file__).resolve().parent))          # figures/
sys.path.append(str(Path(__file__).resolve().parents[1]))      # repo root

from style_guide import CONDITION_STYLES, set_ccn_style  # noqa: E402
from _scatter_common import (  # noqa: E402
    DEFAULT_LAYERS,
    PROMPTSTYLE_DESCR,
    PROMPTSTYLE_DISPLAY,
    persona_hue_palette,
)
from pipeline.prompt_conditions import CONDITIONS  # noqa: E402
from paths import CONFIGS, IMAGE_DIR, OUTPUT  # noqa: E402

# Mirrors _common.DEFAULT_OUTPUT_DIR (drawn from the JSON configs alone).
DEFAULT_OUTPUT_DIR = OUTPUT / "figures"

PERSONAS_A_JSON = CONFIGS / "personas_prototypical.json"
PERSONAS_B_JSON = CONFIGS / "personas_big5_prototypical.json"
# Example Flickr image of the image set (not redistributed; see README).
DEFAULT_IMAGE = IMAGE_DIR / "10006564535_60aec85fde.jpg"

PERSONA_KEYS = ["persona_1", "persona_2", "persona_3", "persona_4", "persona_5"]

# Architecture (Qwen3-VL): 27 ViT blocks, 37 LLM hidden states (0..36).
# Persona embeddings are extracted `llm_only` (pipeline/run_persona_analysis.py), so
# only LLM layers are highlighted; the ViT is drawn but carries no read-out.
N_VIT, N_LLM = 27, 37
# GDV is computed over EVERY LLM layer (gdv_results.csv holds all 37). The
# subset below is only the layers the MDS/scatter figures project - conflating
# the two would misstate what the curves are computed from.
PROJECTION_LAYERS = list(DEFAULT_LAYERS)    # [2, 10, 22, 28, 36]

COL_VIT = "#3a5a99"
COL_LLM = "#c25400"
COL_RATE = "#555555"
COL_FRAME = "#333333"
COL_ARROW = "#444444"
COL_MUTED = "#6b6b6b"

# Block fills for the prompt-composition rows.
COL_BLK_TASK_FC, COL_BLK_TASK_EC = "#eceff3", "#8b93a1"
COL_BLK_PRE_FC, COL_BLK_PRE_EC = "#e3ecf7", COL_VIT
COL_BLK_PER_FC, COL_BLK_PER_EC = "#fbe6d4", COL_LLM

# --- font sizes (points, chosen for 7 in reproduction) ----------------------
# 7 pt is the floor for body text: FS_BLOCK carries the colour key and every
# condition row, i.e. most of the figure's words. It sat at 5 pt, which is
# below what is legible in print. The extra height is paid for out of the
# whitespace constants in _plan_layout rather than by shrinking text again.
FS_SECTION = 10.0
FS_SECSUB = 7.4
FS_BOXTITLE = 8.4
FS_HEAD = 7.2
FS_CELL = 7.2
FS_CHIP = 7.4
FS_DESC = 7.0
FS_BLOCK = 7.0        # colour-key body + preamble text in the condition rows
FS_ARCH = 8.4
FS_LAYER = 7.0
FS_OUT = 7.6

# --- vertical layout --------------------------------------------------------
# Section anchors are NOT hardcoded: the condition rows and the worked prompts
# wrap to a measured number of lines, so their height depends on the text. A
# measuring pass (_plan_layout) adds the required heights up in inches, derives
# FIG_H from the total, and fills in the fractions below. Hand-tuned constants
# drifted every time a prompt was reworded.
AXES_RECT = [0.006, 0.005, 0.988, 0.99]

FIG_W = 7.0            # full text width of the paper
FIG_H = 9.0            # replaced by _plan_layout()
L = SimpleNamespace()  # computed fraction anchors


def _line_h(fontsize: float) -> float:
    """Height of one text line, in inches."""
    return fontsize * 1.28 / 72.0


def _fy(inches: float) -> float:
    """Inches -> axes-fraction height (valid once FIG_H is planned)."""
    return inches / FIG_H


def _fx(inches: float) -> float:
    """Inches -> axes-fraction width."""
    return inches / _axes_w_in()


def _axes_w_in() -> float:
    return FIG_W * AXES_RECT[2]


def _plan_layout() -> None:
    """Measure the text, derive FIG_H, and fill in the section anchors.

    Everything that wraps (condition-row preambles, the two verbatim task
    blocks, the {Persona} field template) has a height that depends on the
    wording, so the vertical layout is computed rather than hand-tuned:
    reword a prompt and the figure regrows instead of overlapping.
    """
    global FIG_H

    tmp = plt.figure(figsize=(FIG_W, 10.0))
    tax = tmp.add_axes(AXES_RECT)
    tax.set_xlim(0, 1); tax.set_ylim(0, 1); tax.axis("off")
    cv = Canvas(tax)

    # Each row is only as tall as its own wrapped text. Sizing every row to
    # the tallest one wasted ~1.3 in of page height. _size_row is the *same*
    # call the drawing code makes, so the reserved height and the drawn height
    # cannot disagree.
    row_h = {row["id"]: _size_row(cv, row)[1] for row in DEFAULT_ROWS + PERSONA_ROWS}
    leg_h = _legend_height(cv)
    plt.close(tmp)

    # Whitespace budget. Deliberately tight: the 7 pt body text costs ~1.2 in
    # of page height over the old 5 pt, and it is paid for here rather than by
    # shrinking the type back down.
    # Second pass on the same budget, to land the page inside a LaTeX float.
    # At 8.95 in the figure did not fit a two-column text block at full
    # \textwidth: 7.0 x 8.95 in plus a caption exceeds \textheight, and the
    # largest width that compiled was 0.91\textwidth, i.e. 7 pt type printing
    # at 6.4 pt. Scaling down defeats the whole point of the font work, so the
    # ~0.3 in came out of inter-element gaps instead. Nothing that separates
    # *text* from its own border was touched -- in particular the schema-table
    # pads (_table_metrics) are unchanged.
    sec_h = _line_h(FS_SECTION) + 0.03          # was +0.05
    tbl_h = _table_metrics()["height"]
    hdr_h = _line_h(FS_BOXTITLE) + 0.02         # was +0.03
    row_gap, group_gap, pad = 0.026, 0.08, 0.038
    def _grp(rows):
        return sum(row_h[r["id"]] + row_gap for r in rows)
    rows_h = (pad + hdr_h + leg_h + group_gap
              + hdr_h + _grp(DEFAULT_ROWS) + group_gap
              + hdr_h + _grp(PERSONA_ROWS) - row_gap + pad)
    # The prompt box is drawn straddling the top edge of the stacks, not up in
    # the caption band, so the band only has to hold one caption line. `sub_gap`
    # is the space between the captions and the stacks: enough for the per-stack
    # sub-caption ("27 layers") and for the prompt box's upper half.
    prompt_h = 2 * _line_h(FS_ARCH - 1.6) + 0.08
    cap_h = _line_h(FS_ARCH) + 0.02
    sub_gap = max(_line_h(FS_DESC - 0.4) + 0.03, prompt_h / 2 + 0.05)
    # The §4 panels are schematic previews of the real figures, not readable
    # plots, so they only need to be big enough to recognise. lab_h + trunk_h
    # must stay clear of the GDV cross-bar at 0.39 in below the stack.
    bar_h, lab_h, trunk_h, thumb_h = 0.38, _line_h(FS_LAYER) + 0.03, 0.38, 0.70

    gap = 0.075                                 # was 0.10
    y = 0.04                                    # was 0.05
    plan = {}
    plan["sec1"] = y + sec_h / 2;      y += sec_h
    plan["pb1"] = y;                   y += tbl_h;   plan["pb0"] = y
    y += gap
    plan["sec2"] = y + sec_h / 2;      y += sec_h
    plan["cb1"] = y;                   y += rows_h;  plan["cb0"] = y
    y += gap
    plan["sec3"] = y + sec_h / 2;      y += sec_h
    y += 0.05
    plan["band"] = y + cap_h;          y += cap_h + sub_gap
    plan["pipe1"] = y;                 y += bar_h;   plan["pipe0"] = y
    y += lab_h
    plan["trunk"] = y;                 y += trunk_h
    plan["sec4"] = y + sec_h / 2;      y += sec_h
    plan["out1"] = y;                  y += thumb_h; plan["out0"] = y
    FIG_H = y + 0.06

    def top(v):   # inches from top -> axes fraction
        return 1.0 - v / FIG_H

    L.SEC1_TITLE_Y = top(plan["sec1"]); L.PB_Y1 = top(plan["pb1"]); L.PB_Y0 = top(plan["pb0"])
    L.SEC2_TITLE_Y = top(plan["sec2"]); L.CB_Y1 = top(plan["cb1"]); L.CB_Y0 = top(plan["cb0"])
    L.SEC3_TITLE_Y = top(plan["sec3"]); L.ARCH_TITLE_Y = top(plan["band"])
    L.PROMPT_H = _fy(prompt_h)
    L.PIPE_Y = top(plan["pipe0"]);      L.PIPE_H = _fy(bar_h)
    L.TRUNK_Y = top(plan["trunk"])
    L.SEC4_TITLE_Y = top(plan["sec4"])
    L.OUT_Y = top(plan["out0"]);        L.OUT_H = _fy(thumb_h)
    L.ROW_H_IN = row_h
    L.ROW_GAP_IN = row_gap
    # Inch offsets shared with the drawing code so the boxes always contain
    # exactly what the plan reserved space for.
    L.PAD_IN = pad
    L.HDR_H_IN = hdr_h
    L.GROUP_GAP_IN = group_gap
    L.LEG_H_IN = leg_h


# ---------------------------------------------------------------------------
# Content: prompt-condition block structure
# ---------------------------------------------------------------------------
# Mirrors pipeline/prompt_conditions.py. Each row is
# the ordered sequence of prompt blocks; `kind` drives the block colour:
#   "pre"     style preamble         "persona"  the {Persona} slot
#   "task"    task + scale + JSON
# The short tags below are abbreviated with "…", and every literal fragment is
# verified against the real prompt at render time (_verify_against_registry),
# so they cannot silently drift.
#
# Rows stay compact: {Persona} and TASK appear as tags only, and the legend at
# the top of the section holds their contents. Spelling them out inside the
# rows made the two rows that carried them ~10x taller than their neighbours
# and still left it unclear which box the text belonged to.
_IMAGINE = "Imagine you are a person with the following specific details:"

DEFAULT_ROWS: List[dict] = [
    {"id": "original_baseline", "chip": "Base", "desc": "task only",
     "blocks": [("task", None)]},
    {"id": "D1", "chip": "D1", "desc": "AI identity",
     "blocks": [("pre", "You are an AI."), ("task", None)]},
    {"id": "D2", "chip": "D2", "desc": "human identity",
     "blocks": [("pre", "You are a human being."), ("task", None)]},
]

PERSONA_ROWS: List[dict] = [
    {"id": "original_persona",
     "blocks": [("pre", _IMAGINE), ("persona", None), ("task", None)]},
    {"id": "P1",
     "blocks": [("pre", "A person with the following specific details is "
                        "observing the image:"),
                ("persona", None), ("task", None)]},
    # P2 appends the *whole* persona block after the task, i.e. the "Imagine
    # you are a person..." preamble comes along with it (_build_p2 composes
    # _PERSONA_BLOCK, not the bare description). The row used to drop that
    # sentence, which made P2 look like the only style whose persona arrives
    # unannounced.
    {"id": "P2",
     "blocks": [("task", None),
                ("pre", "Note: the viewer has the following background:"),
                ("pre", _IMAGINE), ("persona", None)]},
    {"id": "P3",
     "blocks": [("pre", _IMAGINE), ("persona", None),
                ("pre", "Answer only as this persona would, not as a general "
                        "AI assistant."),
                ("task", None)]},
    {"id": "P4",
     "blocks": [("pre", "Research shows that people with the following profile "
                        "tend to perceive images distinctively. Following that "
                        "pattern:"),
                ("persona", None), ("task", None)]},
]
for _r in PERSONA_ROWS:                        # S1..S5 + descriptions from the
    _r["chip"] = PROMPTSTYLE_DISPLAY[_r["id"]]        # shared paper convention
    _r["desc"] = PROMPTSTYLE_DESCR[_r["id"]]

TASK_TAG = "TASK"
PERSONA_TAG = "{Persona}"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_fields(desc: str) -> Dict[str, str]:
    """Parse a persona description into {lowercased key: value}.

    Handles both schemas, including the Big-5 variant whose first field is
    prefixed with a group label ("Big 5 Percentiles: Openness: 15, ...").
    """
    if not desc or not desc.strip():
        return {}
    text = desc.replace("\n", " ").strip()
    for prefix in ("Big 5 Percentiles:", "Big-5 Percentiles:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    fields: Dict[str, str] = {}
    for part in text.split(","):
        part = part.strip().rstrip(".")
        if ":" in part:
            key, val = part.split(":", 1)
            fields[key.strip().lower()] = val.strip()
    return fields


def _short_gender(value: str) -> str:
    return {"male": "M", "female": "F", "diverse": "D"}.get(
        value.lower(), (value[:1].upper() if value else "-")
    )


def _nickname(name: str) -> str:
    """'Persona 3 (The Undercontrolled)' -> 'The Undercontrolled'."""
    if "(" in name and ")" in name:
        return name[name.index("(") + 1: name.rindex(")")].strip()
    return "-"


def _load_schemas() -> Tuple[List[List[str]], List[List[str]]]:
    """Return (rows_A, rows_B) as formatted table cells."""
    with open(PERSONAS_A_JSON) as fh:
        set_a = json.load(fh)
    with open(PERSONAS_B_JSON) as fh:
        set_b = json.load(fh)

    rows_a, rows_b = [], []
    for idx, key in enumerate(PERSONA_KEYS, start=1):
        fa = _parse_fields(set_a[key].get("description", ""))
        rows_a.append([
            f"P{idx}", fa.get("origin", "-"), fa.get("age", "-"),
            _short_gender(fa.get("gender", "")),
            fa.get("professional background", "-"),
            fa.get("mental workload", "-"), fa.get("emotion", "-"),
        ])
        fb = _parse_fields(set_b[key].get("description", ""))
        rows_b.append([
            f"P{idx}", _nickname(set_b[key].get("name", "")),
            fb.get("openness", "-"), fb.get("conscientiousness", "-"),
            fb.get("extraversion", "-"), fb.get("agreeableness", "-"),
            fb.get("neuroticism", "-"), fb.get("age", "-"),
            _short_gender(fb.get("gender", "")),
        ])
    return rows_a, rows_b


def _slot_templates() -> List[Tuple[str, str]]:
    """The {Persona} field template per schema, read off the real descriptions.

    Keys and their order come from the JSON itself (values replaced by "…"),
    so adding or renaming a persona field updates the figure automatically.
    Split on the *last* colon of each comma-part: schema B's first part is
    "Big 5 Percentiles: Openness: 15", whose key spans both labels.
    """
    out = []
    for tag, path in (("A", PERSONAS_A_JSON), ("B", PERSONAS_B_JSON)):
        with open(path) as fh:
            desc = json.load(fh)[PERSONA_KEYS[0]]["description"].strip().rstrip(".")
        keys = [p.strip().rsplit(":", 1)[0].strip() for p in desc.split(",") if ":" in p]
        out.append((tag, ", ".join(f"{k}: …" for k in keys)))
    return out


def _task_texts() -> Dict[str, str]:
    """The two verbatim task blocks, lifted out of the registry's own output.

    Not retyped: the persona variant is whatever `original_persona` emits after
    the persona body, the no-persona variant is `original_baseline` entire.
    """
    sentinel = "\x00PERSONA\x00"
    persona = CONDITIONS["original_persona"].build(sentinel).partition(sentinel)[2]
    return {
        "persona": " ".join(persona.split()),
        "default": " ".join(CONDITIONS["original_baseline"].build(None).split()),
    }


def _verify_against_registry() -> None:
    """Fail loudly if a block tag or ordering no longer matches the real prompt.

    Row tags are abbreviated with "…" to fit; every literal fragment either
    side of an ellipsis must still occur, in order, in the prompt the registry
    actually builds.
    """
    sentinel = "\x00PERSONA\x00"
    for row in DEFAULT_ROWS + PERSONA_ROWS:
        cond = CONDITIONS[row["id"]]
        real = cond.build(sentinel if cond.requires_persona else None)
        flat = " ".join(real.replace("\n", " ").split()).lower()
        kinds = [k for k, _ in row["blocks"]]
        assert ("persona" in kinds) == cond.requires_persona, row["id"]
        if "persona" in kinds:
            assert (real.index(sentinel) < real.index("You are observing")) == (
                kinds.index("persona") < kinds.index("task")
            ), f"{row['id']}: block order disagrees with the registry"
        for kind, text in row["blocks"]:
            if kind != "pre":
                continue
            pos = 0
            for frag in text.strip('"').split("…"):
                frag = " ".join(frag.split()).lower().strip()
                if not frag:
                    continue
                found = flat.find(frag, pos)
                assert found >= 0, f"{row['id']}: {frag!r} not in the real prompt"
                pos = found + len(frag)


def _verify_tables_complete() -> None:
    """Every persona field that *varies* must appear in the figure.

    IQ is in the Big-5 spec but is 100 for all five personas, so it is reported
    in the subtitle rather than given a column. If it (or any other constant)
    ever becomes a manipulated variable, this fails instead of quietly hiding
    a factor the model was conditioned on.
    """
    shown = {
        PERSONAS_A_JSON: {"origin", "age", "gender", "professional background",
                          "mental workload", "emotion"},
        PERSONAS_B_JSON: {"openness", "conscientiousness", "extraversion",
                          "agreeableness", "neuroticism", "age", "gender"},
    }
    for path, cols in shown.items():
        with open(path) as fh:
            data = json.load(fh)
        rows = [_parse_fields(data[k]["description"]) for k in PERSONA_KEYS]
        for field in rows[0]:
            varies = len({r.get(field) for r in rows}) > 1
            assert not (varies and field not in cols), (
                f"{path.name}: '{field}' varies across personas but is not "
                f"shown in the figure")


def _task_runs() -> List[Tuple[str, str]]:
    """Word-level diff of the two task blocks, as (text, role) runs.

    role is "" for text both variants share, "persona" for text only the
    persona conditions get, "default" for text only the no-persona ones get.
    The two blocks are ~90 words that differ in one clause, so diffing them
    beats printing both: the figure shows the task once and marks the fork.
    """
    a = _task_texts()["default"].split()
    b = _task_texts()["persona"].split()
    runs: List[Tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            runs.append((" ".join(a[i1:i2]), ""))
            continue
        if j2 > j1:
            runs.append((" ".join(b[j1:j2]), "persona"))
        if i2 > i1:
            runs.append((" ".join(a[i1:i2]), "default"))
    return runs


def _verify_task_variants() -> None:
    """The two task blocks must differ, in exactly one place.

    The inline fork in the legend only reads correctly if there is a single
    divergence. If the registry ever unified the blocks, or spread the
    difference across the text, the figure would be claiming something false -
    so fail the build instead of drawing it.
    """
    tasks = _task_texts()
    assert tasks["persona"] != tasks["default"], (
        "the persona and no-persona task blocks are identical - "
        "the legend's fork no longer says anything")
    assert "based on your details" in tasks["persona"].lower()
    assert "based on your details" not in tasks["default"].lower()

    forks = [i for i, (_, role) in enumerate(_task_runs()) if role]
    contiguous = forks == list(range(forks[0], forks[0] + len(forks)))
    assert forks and contiguous and len(forks) <= 2, (
        "the task blocks now diverge in more than one place; a single inline "
        "fork would misrepresent them")


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
class Canvas:
    """Thin wrapper giving axes-fraction text measurement + common patches."""

    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
        self.fig.canvas.draw()  # ensure a renderer exists
        self._renderer = self.fig.canvas.get_renderer()
        self._inv = ax.transAxes.inverted()
        self._cache: Dict[tuple, Tuple[float, float]] = {}

    def measure(self, text: str, fontsize: float, weight: str = "normal",
                style: str = "normal") -> Tuple[float, float]:
        """Width/height of `text` in axes fraction.

        Cached: laying out the legend paragraphs measures a growing prefix per
        word, which is hundreds of calls over a handful of distinct strings.
        """
        key = (text, fontsize, weight, style)
        if key not in self._cache:
            t = self.ax.text(0, 0, text, fontsize=fontsize, fontweight=weight,
                             style=style, transform=self.ax.transAxes)
            bb = t.get_window_extent(renderer=self._renderer)
            t.remove()
            (x0, y0), (x1, y1) = self._inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            self._cache[key] = (x1 - x0, y1 - y0)
        return self._cache[key]

    def box(self, x, y, w, h, *, fc="white", ec=COL_FRAME, lw=0.6, ls="-",
            radius=0.006, z=1):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
            transform=self.ax.transAxes, zorder=z))

    def arrow(self, verts: Sequence[Tuple[float, float]], color=COL_ARROW,
              lw=0.9, scale=8):
        kw = dict(arrowstyle="->", color=color, linewidth=lw, mutation_scale=scale,
                  shrinkA=0, shrinkB=1.5, capstyle="round", joinstyle="round", zorder=5)
        if len(verts) == 2:
            patch = mpatches.FancyArrowPatch(verts[0], verts[1],
                                             transform=self.ax.transAxes, **kw)
        else:
            path = mpath.Path(list(verts),
                              [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(verts) - 1))
            patch = mpatches.FancyArrowPatch(path=path, transform=self.ax.transAxes, **kw)
        self.ax.add_patch(patch)


def _section_title(cv: Canvas, x: float, y: float, number: str, text: str, sub: str = ""):
    """'1' in a filled circle, then the section heading, then a muted suffix."""
    cv.ax.text(x, y, number, fontsize=FS_SECTION - 2.0, fontweight="bold", color="white",
               ha="center", va="center", transform=cv.ax.transAxes, zorder=4,
               bbox=dict(boxstyle="circle,pad=0.30", facecolor=COL_FRAME, edgecolor="none"))
    tx = x + 0.016
    cv.ax.text(tx, y, text, fontsize=FS_SECTION, fontweight="bold", color=COL_FRAME,
               ha="left", va="center", transform=cv.ax.transAxes)
    if sub:
        w, _ = cv.measure(text, FS_SECTION, "bold")
        cv.ax.text(tx + w + 0.010, y, sub, fontsize=FS_SECSUB, color=COL_MUTED,
                   ha="left", va="center", transform=cv.ax.transAxes)


# ---------------------------------------------------------------------------
# Section 1 - persona schemas
# ---------------------------------------------------------------------------
TBL_COL_PAD = 0.008     # gap between table columns
TBL_MARGIN = 0.018      # slack _draw_table keeps inside (x1 - x0)
PANEL_INSET_L, PANEL_INSET_R = 0.010, 0.006   # panel border -> table text
PANEL_L, PANEL_R, PANEL_GAP = 0.012, 0.988, 0.016

# Vertical metrics of a schema table, in inches below the panel top.
TBL_PAD_TOP = 0.045
TBL_BRACKET = 0.13      # room above the headers for the Big-5 group bracket
TBL_HEAD_GAP = 0.035    # header baseline -> first data row
TBL_ROW_PAD = 0.032     # added to the cell line height for the row pitch
TBL_PAD_BOT = 0.060     # last row's descender -> panel bottom


def _table_metrics() -> Dict[str, float]:
    """Offsets of every table element below the panel top, plus panel height.

    Shared by the layout planner and the drawing code. They used to carry
    separate copies of these numbers and had drifted apart: the plan reserved
    1.432 in while the last row was drawn down to 1.455 in, i.e. the bottom
    row sat *outside* its own panel.
    """
    title = TBL_PAD_TOP + _line_h(FS_BOXTITLE) / 2
    sub = title + _line_h(FS_BOXTITLE) / 2 + _line_h(FS_DESC)
    head = sub + _line_h(FS_DESC) + TBL_BRACKET
    row0 = head + _line_h(FS_HEAD) + TBL_HEAD_GAP
    step = _line_h(FS_CELL) + TBL_ROW_PAD
    n_rows = len(PERSONA_KEYS)
    return {
        "title": title, "sub": sub, "head": head, "row0": row0, "step": step,
        "height": row0 + (n_rows - 1) * step + _line_h(FS_CELL) / 2 + TBL_PAD_BOT,
    }


def _table_cols(cv: Canvas, headers, rows) -> Tuple[List[float], float]:
    """Natural column widths and their total, in axes fraction."""
    widths = []
    for ci in range(len(headers)):
        w = cv.measure(headers[ci], FS_HEAD, "bold")[0]
        for r in rows:
            w = max(w, cv.measure(str(r[ci]), FS_CELL)[0])
        widths.append(w)
    return widths, sum(widths) + TBL_COL_PAD * (len(headers) - 1)


def _panel_split(cv, headers_a, rows_a, headers_b, rows_b):
    """Divide the row between the two schema panels by what each table needs.

    A 50/50-ish hardcoded split had Schema A overflowing its border by 0.07 in
    once the cell font grew. Each panel now gets its natural width plus a
    proportional share of the slack, so the divider follows the content.
    """
    need = [_table_cols(cv, h, r)[1] + TBL_MARGIN + PANEL_INSET_L + PANEL_INSET_R
            for h, r in ((headers_a, rows_a), (headers_b, rows_b))]
    span = (PANEL_R - PANEL_L) - PANEL_GAP
    slack = span - sum(need)
    assert slack >= 0, (
        f"the two schema tables need {sum(need) * _axes_w_in():.2f} in but the "
        f"page gives {span * _axes_w_in():.2f} in; lower FS_CELL")
    w_a = need[0] + slack * need[0] / sum(need)
    return (PANEL_L, PANEL_L + w_a), (PANEL_L + w_a + PANEL_GAP, PANEL_R)


def _draw_table(cv: Canvas, x0, x1, y1, title, subtitle, headers, rows, colors,
                group_span: Optional[Tuple[int, int, str]] = None) -> None:
    """Draw one persona schema table; columns are auto-sized from content."""
    tm = _table_metrics()
    row_step = _fy(tm["step"])
    title_y = y1 - _fy(tm["title"])
    sub_y = y1 - _fy(tm["sub"])
    head_y = y1 - _fy(tm["head"])
    row_y0 = y1 - _fy(tm["row0"])

    cv.ax.text(x0, title_y, title, fontsize=FS_BOXTITLE, fontweight="bold",
               color=COL_FRAME, ha="left", va="center", transform=cv.ax.transAxes)
    cv.ax.text(x0, sub_y, subtitle, fontsize=FS_DESC, color=COL_MUTED, style="italic",
               ha="left", va="center", transform=cv.ax.transAxes)

    n_col = len(headers)
    widths, total = _table_cols(cv, headers, rows)
    pad = TBL_COL_PAD
    avail = (x1 - x0) - TBL_MARGIN
    # Columns are natural-width, so an over-wide table silently ran past x1
    # into the panel border instead of failing. Larger cell fonts made this a
    # live risk, so it is now an error.
    assert total <= avail + 1e-6, (
        f"{title}: columns need {total * _axes_w_in():.2f} in but the panel "
        f"gives {avail * _axes_w_in():.2f} in; lower FS_CELL or widen the panel")
    extra = max(avail - total, 0.0) / (n_col - 1) if n_col > 1 else 0.0

    xs, cur = [], x0 + 0.016
    for ci in range(n_col):
        xs.append(cur)
        cur += widths[ci] + pad + extra

    if group_span is not None:
        gi, gj, glabel = group_span
        gx0, gx1 = xs[gi] - 0.003, xs[gj] + widths[gj] + 0.003
        gy = head_y + _fy(0.075)
        cv.ax.plot([gx0, gx1], [gy, gy], color=COL_MUTED, lw=0.6,
                   transform=cv.ax.transAxes, clip_on=False)
        for gx in (gx0, gx1):
            cv.ax.plot([gx, gx], [gy - _fy(0.03), gy], color=COL_MUTED, lw=0.6,
                       transform=cv.ax.transAxes, clip_on=False)
        cv.ax.text((gx0 + gx1) / 2, gy + _fy(0.012), glabel, fontsize=FS_DESC - 0.4,
                   color=COL_MUTED, ha="center", va="bottom", transform=cv.ax.transAxes)

    for x, h in zip(xs, headers):
        cv.ax.text(x, head_y, h, fontsize=FS_HEAD, fontweight="bold", color="#666666",
                   ha="left", va="center", transform=cv.ax.transAxes)
    cv.ax.plot([x0 + 0.004, x1], [head_y - 0.009, head_y - 0.009],
               color="#cccccc", lw=0.6, transform=cv.ax.transAxes)

    for ri, row in enumerate(rows):
        y = row_y0 - ri * row_step
        cv.ax.scatter([x0 + 0.007], [y], c=[colors[ri]], s=13, edgecolor="black",
                      linewidth=0.3, transform=cv.ax.transAxes, zorder=3)
        for x, val in zip(xs, row):
            cv.ax.text(x, y, str(val), fontsize=FS_CELL, color="#1a1a1a",
                       ha="left", va="center", transform=cv.ax.transAxes)


def draw_personas(cv: Canvas, rows_a, rows_b, colors_a, colors_b) -> None:
    _section_title(cv, 0.020, L.SEC1_TITLE_Y, "1", "Personas",
                   sub="two schemas × five prototypical personas (P1–P5)")
    head_a = ["ID", "Origin", "Age", "Sex", "Profession", "Workload", "Emotion"]
    head_b = ["ID", "Type", "O", "C", "E", "A", "N", "Age", "Sex"]
    (a0, a1), (b0, b1) = _panel_split(cv, head_a, rows_a, head_b, rows_b)

    h = L.PB_Y1 - L.PB_Y0
    cv.box(a0, L.PB_Y0, a1 - a0, h, fc="#fefefe", ec="#aaaaaa", ls="--", lw=0.6)
    cv.box(b0, L.PB_Y0, b1 - b0, h, fc="#fefefe", ec="#aaaaaa", ls="--", lw=0.6)

    _draw_table(cv, a0 + PANEL_INSET_L, a1 - PANEL_INSET_R, L.PB_Y1,
                "Socio-demographic schema", "verbal profile",
                head_a, rows_a, colors_a)
    _draw_table(cv, b0 + PANEL_INSET_L, b1 - PANEL_INSET_R, L.PB_Y1,
                "Big Five schema", "trait percentiles · IQ fixed at 100",
                head_b, rows_b, colors_b, group_span=(2, 6, "Big Five percentiles"))


# ---------------------------------------------------------------------------
# Section 2 - prompt styles
# ---------------------------------------------------------------------------
CHIP_X, CHIP_W = 0.022, 0.040
DESC_X = 0.062        # floor; _desc_x() pushes it past the widest chip
CHIP_PAD = 0.26       # bbox pad, in fractions of the chip font size
BLOCK_X0_MIN, BLOCK_X1 = 0.178, 0.980
BLOCK_WRAP_MAX_IN = 3.70            # widest a single preamble block may get
BLOCK_GAP = 0.006
BLOCK_PAD_IN = 0.05                 # block box padding above+below its text

# Legend: the two blocks the rows only reference by tag get their contents
# spelled out once, next to a chip drawn exactly like the in-row block, so it
# is unambiguous which box the text belongs to.
LEG_X0, LEG_X1 = 0.022, 0.980
LEG_GAP = 0.014          # chip -> text column
LEG_ENTRY_GAP = 0.042    # inches between legend entries
PRE_TAG = "preamble"

# Roles in the task diff. Orange matches the {Persona} block, so "this bit is
# only there when a persona is present" needs no further explanation.
ROLE_COLOR = {"": "#1a1a1a", "persona": COL_LLM, "default": "#3f3f3f"}
COL_FORK = "#9098a4"     # brackets and the "/" separator


def _block_style(kind: str):
    return {"task": (COL_BLK_TASK_FC, COL_BLK_TASK_EC),
            "pre": (COL_BLK_PRE_FC, COL_BLK_PRE_EC),
            "persona": (COL_BLK_PER_FC, COL_BLK_PER_EC)}[kind]


def _desc_x(cv: Canvas) -> float:
    """Where the italic descriptions start: clear of the *widest* chip.

    "Base" is a good deal wider than "D1"/"S1", so a fixed DESC_X let its chip
    collide with its own description. Measured once over every row, so all the
    descriptions stay left-aligned with each other.
    """
    widest = max(cv.measure(r["chip"], FS_CHIP, "bold")[0]
                 for r in DEFAULT_ROWS + PERSONA_ROWS)
    chip_edge = CHIP_X + CHIP_W / 2 + widest / 2 + _fx(CHIP_PAD * FS_CHIP / 72.0)
    return max(DESC_X, chip_edge + 0.006)


def _block_x0(cv: Canvas) -> float:
    """Where the block sequence starts: clear of the longest description.

    Also measured rather than fixed - at 7 pt "persona after task" is wide
    enough to run into a hardcoded column start.
    """
    widest = max(cv.measure(r["desc"], FS_DESC, style="italic")[0]
                 for r in DEFAULT_ROWS + PERSONA_ROWS)
    return max(BLOCK_X0_MIN, _desc_x(cv) + widest + 0.010)


def _wrap_to_width(cv: Canvas, text: str, fontsize: float, max_w: float,
                   pad: float = 0.012) -> Tuple[List[str], float]:
    """Wrap `text` so the drawn block is at most `max_w` wide.

    Returns (lines, block_width). Every candidate line is *measured*, not
    estimated from a nominal character width: sizing by the width of "n"
    overestimated the average glyph by ~30% (0.063 in vs 0.048 in for mixed
    text in DejaVu Sans), so every block wrapped a line or two earlier than it
    needed to. Width comes from the longest measured line, so the box always
    contains its text.
    """
    limit = max_w - pad
    lines: List[str] = []
    cur: List[str] = []
    for word in text.split():
        if cur and cv.measure(" ".join(cur + [word]), fontsize)[0] > limit:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    lines = lines or [""]
    return lines, max(cv.measure(ln, fontsize)[0] for ln in lines) + pad


# --- multi-coloured paragraphs ---------------------------------------------
def _wrap_tokens(cv: Canvas, tokens: List[Tuple[str, str]], fontsize: float,
                 width: float, style: str = "normal") -> List[List[Tuple[str, str, float]]]:
    """Wrap coloured word tokens; returns lines of (word, colour, x offset).

    A plain text wrapper can't be used here: each word carries its own colour. Word
    positions come from measuring the *prefix* of the line rather than summing
    per-word widths, so inter-word kerning is accounted for and the coloured
    run sits exactly where it would in a plain string.
    """
    space = max(cv.measure("a a", fontsize, style=style)[0]
                - cv.measure("aa", fontsize, style=style)[0], 0.001)

    lines: List[List[Tuple[str, str]]] = []
    cur: List[Tuple[str, str]] = []
    for word, role in tokens:
        trial = " ".join(w for w, _ in cur + [(word, role)])
        if cur and cv.measure(trial, fontsize, style=style)[0] > width:
            lines.append(cur)
            cur = []
        cur.append((word, role))
    if cur:
        lines.append(cur)

    placed = []
    for line in lines:
        words = [w for w, _ in line]
        placed.append([
            (word, colour,
             0.0 if i == 0 else cv.measure(" ".join(words[:i]), fontsize, style=style)[0] + space)
            for i, (word, colour) in enumerate(line)
        ])
    return placed


def _draw_tokens(cv: Canvas, placed, x0: float, y_first: float, fontsize: float,
                 style: str = "normal") -> None:
    line_h = _fy(_line_h(fontsize))
    for i, line in enumerate(placed):
        for word, colour, dx in line:
            cv.ax.text(x0 + dx, y_first - i * line_h, word, fontsize=fontsize,
                       color=colour, style=style, ha="left", va="center",
                       transform=cv.ax.transAxes, zorder=3)


def _task_tokens() -> List[Tuple[str, str]]:
    """The task text as (word, colour) tokens, with the fork marked inline.

    Shared text is black; at the one point the variants diverge both readings
    are shown as "[persona wording / default wording]".
    """
    tokens: List[Tuple[str, str]] = []
    runs = _task_runs()
    i = 0
    while i < len(runs):
        text, role = runs[i]
        if not role:
            tokens += [(w, ROLE_COLOR[""]) for w in text.split()]
            i += 1
            continue
        alts = []
        while i < len(runs) and runs[i][1]:
            alts.append(runs[i])
            i += 1
        for k, (alt_text, alt_role) in enumerate(alts):
            words = alt_text.split()
            for j, w in enumerate(words):
                w = ("[" + w) if (k == 0 and j == 0) else w
                if k == len(alts) - 1 and j == len(words) - 1:
                    w += "]"
                tokens.append((w, ROLE_COLOR[alt_role]))
            if k < len(alts) - 1:
                tokens.append(("/", COL_FORK))
    return tokens


def _legend_entries(cv: Canvas) -> List[dict]:
    """The three block kinds, with the contents the rows abbreviate."""
    schema_name = {"A": "Socio-demographic", "B": "Big Five"}
    schema_paras = [([(w, ROLE_COLOR[""]) for w in f"{schema_name[tag]}: {template}".split()],
                     "normal")
                    for tag, template in _slot_templates()]
    entries = [
        {"kind": "pre", "tag": PRE_TAG, "paras": [
            ([(w, ROLE_COLOR[""]) for w in
              "the sentence that differs between styles; written out in full in every row below.".split()],
             "normal")]},
        {"kind": "persona", "tag": PERSONA_TAG, "paras": schema_paras},
        {"kind": "task", "tag": TASK_TAG, "paras": [
            (_task_tokens(), "normal"),
            ([(w, COL_MUTED) for w in
              ("orange text appears only in the prompt styles (S1–S5); "
               "Base, D1 and D2 take the grey alternative.").split()],
             "italic")]},
    ]

    chip_w = max(cv.measure(e["tag"], FS_BLOCK, "bold")[0] for e in entries) + 0.016
    text_x = LEG_X0 + chip_w + LEG_GAP
    for e in entries:
        e["chip_w"] = chip_w
        e["text_x"] = text_x
        e["lines"] = []
        for tokens, style in e["paras"]:
            for line in _wrap_tokens(cv, tokens, FS_BLOCK, LEG_X1 - text_x, style):
                e["lines"].append((line, style))
        e["h"] = len(e["lines"]) * _line_h(FS_BLOCK) + BLOCK_PAD_IN
    return entries


def _legend_height(cv: Canvas) -> float:
    entries = _legend_entries(cv)
    return sum(e["h"] for e in entries) + LEG_ENTRY_GAP * (len(entries) - 1)


def _draw_legend(cv: Canvas, y_top: float) -> None:
    entries = _legend_entries(cv)
    chip_h = _line_h(FS_BLOCK) + BLOCK_PAD_IN
    d = 0.0
    for e in entries:
        fc, ec = _block_style(e["kind"])
        # The chip is one line tall whatever the entry's height, so it is the
        # same object the rows show rather than a stretched version of it.
        cv.box(LEG_X0, y_top - _fy(d + chip_h), e["chip_w"], _fy(chip_h),
               fc=fc, ec=ec, lw=0.7, radius=0.004, z=2)
        cv.ax.text(LEG_X0 + e["chip_w"] / 2, y_top - _fy(d + chip_h / 2), e["tag"],
                   fontsize=FS_BLOCK, fontweight="bold", color="#1a1a1a",
                   ha="center", va="center", transform=cv.ax.transAxes, zorder=3)

        y = y_top - _fy(d + BLOCK_PAD_IN / 2 + _line_h(FS_BLOCK) / 2)
        for i, (line, style) in enumerate(e["lines"]):
            _draw_tokens(cv, [line], e["text_x"],
                         y - i * _fy(_line_h(FS_BLOCK)), FS_BLOCK, style)
        d += e["h"] + LEG_ENTRY_GAP


def _size_row(cv: Canvas, row: dict) -> Tuple[List[dict], float]:
    """Lay a condition row out horizontally; return (blocks, row height in inches).

    Called by both the layout planner and the drawing code, so the height the
    plan reserves is by construction the height that gets drawn. Widths are
    measured, never assumed, so a block can't come out narrower than the label
    inside it (the {Persona} chip used to overflow a hardcoded width).
    """
    avail = BLOCK_X1 - _block_x0(cv)
    gaps = BLOCK_GAP * (len(row["blocks"]) - 1)

    # Preambles share whatever the fixed-width tags leave in *this* row, so a
    # row with one preamble gets a wide box (one line) while S4, which has two,
    # gets narrower ones. A single global wrap width forced the roomy rows to
    # wrap to 3 lines for no reason and cost ~0.4 in of page height.
    tag_w = {"task": cv.measure(TASK_TAG, FS_BLOCK)[0] + 0.014,
             "persona": cv.measure(PERSONA_TAG, FS_BLOCK, "bold")[0] + 0.016}
    fixed = sum(tag_w[k] for k, _ in row["blocks"] if k in tag_w)
    n_pre = sum(1 for k, _ in row["blocks"] if k == "pre")
    budget = min((avail - gaps - fixed) / max(n_pre, 1), _fx(BLOCK_WRAP_MAX_IN))

    blocks: List[dict] = []
    for kind, text in row["blocks"]:
        if kind in tag_w:
            tag = TASK_TAG if kind == "task" else PERSONA_TAG
            blocks.append({"kind": kind, "lines": [tag], "w": tag_w[kind]})
        else:
            lines, w = _wrap_to_width(cv, text, FS_BLOCK, budget)
            blocks.append({"kind": kind, "lines": lines, "w": w})

    used = sum(b["w"] for b in blocks) + gaps
    assert used <= avail + 1e-6, (
        f"{row['id']}: blocks need {used * _axes_w_in():.2f} in but only "
        f"{avail * _axes_w_in():.2f} in is available")

    return blocks, (max(len(b["lines"]) for b in blocks) * _line_h(FS_BLOCK)
                    + BLOCK_PAD_IN)


def _draw_condition_row(cv: Canvas, row: dict, yc: float, block_h: float) -> None:
    color = CONDITION_STYLES.get(row["id"], {"color": "#000000"})["color"]
    cv.ax.text(CHIP_X + CHIP_W / 2, yc, row["chip"], fontsize=FS_CHIP, fontweight="bold",
               color="white", ha="center", va="center", transform=cv.ax.transAxes, zorder=3,
               bbox=dict(boxstyle=f"round,pad={CHIP_PAD}", facecolor=color, edgecolor="none"))
    cv.ax.text(_desc_x(cv), yc, row["desc"], fontsize=FS_DESC, color=COL_MUTED,
               style="italic", ha="left", va="center", transform=cv.ax.transAxes)

    x = _block_x0(cv)
    for b in _size_row(cv, row)[0]:
        fc, ec = _block_style(b["kind"])
        w = b["w"]
        cv.box(x, yc - block_h / 2, w, block_h, fc=fc, ec=ec, lw=0.7, radius=0.004, z=2)
        cv.ax.text(x + w / 2, yc, "\n".join(b["lines"]), fontsize=FS_BLOCK,
                   fontweight="bold" if b["kind"] == "persona" else "normal",
                   color="#1a1a1a", ha="center", va="center", linespacing=1.30,
                   multialignment="left",  # centred block, left-aligned lines
                   transform=cv.ax.transAxes, zorder=3)
        x += w + BLOCK_GAP


def draw_conditions(cv: Canvas) -> None:
    # "Prompt style" is reserved for S1-S5 (the PromptStyle factor in the
    # rating ANOVA); the section as a whole covers the defaults too, so it is
    # named for the act of assembling a prompt rather than for one factor.
    _section_title(cv, 0.020, L.SEC2_TITLE_Y, "2", "Prompt Creation",
                   sub="3 reference conditions + 5 prompt styles (S1–S5) "
                       "→ 53 prompt variants per image")
    cv.box(0.012, L.CB_Y0, 0.976, L.CB_Y1 - L.CB_Y0, fc="#fcfcfc", ec="#aaaaaa",
           ls="--", lw=0.6)

    def _group_header(y, text, count):
        cv.ax.text(CHIP_X, y, text, fontsize=FS_BOXTITLE - 0.8, fontweight="bold",
                   color=COL_FRAME, ha="left", va="center", transform=cv.ax.transAxes)
        w = cv.measure(text, FS_BOXTITLE - 0.8, "bold")[0]
        cv.ax.text(CHIP_X + w + 0.008, y, count, fontsize=FS_DESC, color=COL_MUTED,
                   ha="left", va="center", transform=cv.ax.transAxes)

    def at(inches_down):
        return L.CB_Y1 - _fy(inches_down)

    def _group(d, title, count, rows):
        _group_header(at(d + L.HDR_H_IN / 2), title, count)
        d += L.HDR_H_IN
        for row in rows:
            h = L.ROW_H_IN[row["id"]]
            _draw_condition_row(cv, row, at(d + h / 2), _fy(h))
            d += h + L.ROW_GAP_IN
        return d - L.ROW_GAP_IN

    d = L.PAD_IN
    _group_header(at(d + L.HDR_H_IN / 2), "Prompt blocks", "")
    d += L.HDR_H_IN
    _draw_legend(cv, at(d))
    d += L.LEG_H_IN + L.GROUP_GAP_IN

    d = _group(d, "Reference conditions", "(no persona, 3)", DEFAULT_ROWS)
    _group(d + L.GROUP_GAP_IN, "Prompt styles",
           "(5 × 2 schemas × 5 personas = 50)", PERSONA_ROWS)


# ---------------------------------------------------------------------------
# Section 3 - pipeline
# ---------------------------------------------------------------------------
# LLM + rating shifted right to open a gap after the ViT: the prompt box sits
# in that gap, above the ViT->LLM arrow, so both inputs enter the decoder at
# its left edge (layer 0) rather than partway along the stack.
IMG_X, IMG_W = 0.014, 0.052
VIT_X, VIT_W = 0.098, 0.120
LLM_X, LLM_W = 0.446, 0.300
RATE_X, RATE_W = 0.792, 0.196
# Centred in the ViT->LLM gap rather than placed by hand, so it stays centred
# if either stack is resized.
PROMPT_W = 0.164
PROMPT_X = (VIT_X + VIT_W + LLM_X) / 2 - PROMPT_W / 2


def _draw_stack(cv: Canvas, x0, w, n_layers, highlights, col, label, sub,
                read_all: bool = False) -> List[float]:
    cv.box(x0, L.PIPE_Y, w, L.PIPE_H, fc="white", ec=col, lw=1.0, radius=0.004, z=2)
    # `read_all` stacks show every layer tick in full colour: GDV is computed
    # from all of them, not only the labelled subset.
    tick = col + ("88" if read_all else "33")
    for li in range(n_layers):
        x = x0 + (li / max(n_layers - 1, 1)) * w
        cv.ax.plot([x, x], [L.PIPE_Y + 0.003, L.PIPE_Y + L.PIPE_H - 0.003],
                   color=tick, lw=0.55 if read_all else 0.3,
                   transform=cv.ax.transAxes, zorder=3)

    hi_w = max(w / n_layers * 0.95, 0.003)
    centers = []
    for li in highlights:
        xc = x0 + (li / max(n_layers - 1, 1)) * w
        cv.ax.add_patch(FancyBboxPatch(
            (xc - hi_w / 2, L.PIPE_Y + 0.002), hi_w, L.PIPE_H - 0.004,
            boxstyle="square,pad=0", facecolor=col, edgecolor=col, linewidth=0.4,
            transform=cv.ax.transAxes, zorder=4))
        # Offset right of the tick, not centred on it: the read-out lines drop
        # from exactly these x positions and would otherwise strike the label.
        cv.ax.text(xc + _fx(0.03), L.PIPE_Y - _fy(0.045), f"L{li}", fontsize=FS_LAYER,
                   ha="left", va="top", color=col, fontweight="bold",
                   transform=cv.ax.transAxes)
        centers.append(xc)

    # Caption on the band, layer count on its own line just above the box: a
    # single long caption overran the narrow ViT box into its neighbours.
    cv.ax.text(x0 + w / 2, L.ARCH_TITLE_Y, label, fontsize=FS_ARCH, fontweight="bold",
               color=col, ha="center", va="bottom", transform=cv.ax.transAxes)
    cv.ax.text(x0 + w / 2, L.PIPE_Y + L.PIPE_H + _fy(0.025), sub,
               fontsize=FS_DESC - 0.4, color=col, ha="center", va="bottom",
               transform=cv.ax.transAxes)
    return centers


def draw_pipeline(cv: Canvas, image_path: Optional[Path]) -> List[float]:
    ax = cv.ax
    _section_title(cv, 0.020, L.SEC3_TITLE_Y, "3", "Model",
                   sub="Qwen3-VL · LLM-stream embeddings, last-token pooled")
    if image_path and Path(image_path).exists():
        try:
            from PIL import Image as PILImage

            thumb = ax.inset_axes([IMG_X, L.PIPE_Y, IMG_W, L.PIPE_H])
            thumb.imshow(PILImage.open(image_path).convert("RGB"))
            thumb.set_xticks([]); thumb.set_yticks([]); thumb.grid(False)
            for sp in thumb.spines.values():
                sp.set_visible(True); sp.set_edgecolor(COL_FRAME); sp.set_linewidth(0.6)
        except Exception as exc:  # pragma: no cover - cosmetic only
            print(f"  Warning: could not embed thumbnail ({exc})")
    ax.text(IMG_X + IMG_W / 2, L.ARCH_TITLE_Y, "Image", ha="center", va="bottom",
            fontsize=FS_ARCH, fontweight="bold", color=COL_FRAME, transform=ax.transAxes)

    # The ViT carries no highlight: persona embeddings are llm_only.
    _draw_stack(cv, VIT_X, VIT_W, N_VIT, [], COL_VIT, "ViT Encoder",
                f"{N_VIT} layers")
    llm = _draw_stack(cv, LLM_X, LLM_W, N_LLM, PROJECTION_LAYERS, COL_LLM,
                      "LLM Decoder", f"{N_LLM - 1} layers · every layer read",
                      read_all=True)

    # The rating is a block of the same height as the stacks, so the arrow into
    # it starts and ends on a box edge exactly like Image->ViT and ViT->LLM
    # (it used to stop short in whitespace ahead of a floating text bbox).
    cv.box(RATE_X, L.PIPE_Y, RATE_W, L.PIPE_H, fc="#f5f5f5", ec=COL_RATE,
           lw=1.0, radius=0.004, z=2)
    ax.text(RATE_X + RATE_W / 2, L.ARCH_TITLE_Y, "Rating", ha="center", va="bottom",
            fontsize=FS_ARCH, fontweight="bold", color=COL_RATE, transform=ax.transAxes)
    ax.text(RATE_X + RATE_W / 2, L.PIPE_Y + L.PIPE_H / 2, '"Moderately\nInteresting"',
            ha="center", va="center", fontsize=FS_ARCH - 1.4, fontweight="bold",
            color=COL_RATE, linespacing=1.25, transform=ax.transAxes, zorder=3)

    mid = L.PIPE_Y + L.PIPE_H / 2
    cv.arrow([(IMG_X + IMG_W, mid), (VIT_X, mid)])
    cv.arrow([(LLM_X + LLM_W, mid), (RATE_X, mid)])

    # Prompt input: its own box in the ViT->LLM gap, straddling the top edge of
    # the stacks and clearing the image arrow that runs beneath it. Both arrows
    # land on the decoder's left edge = layer 0, where the prompt tokens
    # actually enter.
    prompt_y = L.PIPE_Y + L.PIPE_H
    cv.box(PROMPT_X, prompt_y - L.PROMPT_H / 2, PROMPT_W, L.PROMPT_H,
           fc=COL_BLK_PER_FC, ec=COL_LLM, lw=0.8, radius=0.004, z=2)
    ax.text(PROMPT_X + PROMPT_W / 2, prompt_y, "Assembled Prompt\n(§2)",
            fontsize=FS_ARCH - 1.6, fontweight="bold", color=COL_LLM,
            ha="center", va="center", linespacing=1.25,
            transform=ax.transAxes, zorder=3)

    entry_y = L.PIPE_Y + L.PIPE_H * 0.78
    cv.arrow([(PROMPT_X + PROMPT_W, prompt_y), (LLM_X - 0.014, prompt_y),
              (LLM_X - 0.014, entry_y), (LLM_X, entry_y)], color=COL_LLM)
    cv.arrow([(VIT_X + VIT_W, mid), (LLM_X, mid)])

    return llm


# ---------------------------------------------------------------------------
# Section 4 - read-outs
# ---------------------------------------------------------------------------
MDS_X, MDS_W = 0.060, 0.260
GDV_X, GDV_W = 0.370, 0.260
BEH_X, BEH_W = 0.720, 0.260


def _thumb_axes(cv: Canvas, x, w, title):
    ax = cv.ax.inset_axes([x, L.OUT_Y, w, L.OUT_H])
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_edgecolor("#555555")
    ax.text(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top",
            fontsize=FS_OUT, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9))
    return ax


def draw_outputs(cv: Canvas, colors_a, colors_b, llm_centers):
    _section_title(cv, 0.020, L.SEC4_TITLE_Y, "4", "Read-outs")

    ax_mds = _thumb_axes(cv, MDS_X, MDS_W, "MDS projections")
    rng = np.random.RandomState(42)
    for ci, c in enumerate(list(colors_a) + list(colors_b)):
        ax_mds.scatter(rng.normal((ci % 5) * 0.5 + (ci // 5) * 0.7, 0.30, 9),
                       rng.normal((ci // 5) * 1.0, 0.40, 9),
                       c=[c], s=3.5, alpha=0.85, edgecolor="none")
    ax_mds.set_xlim(-1.2, 3.7); ax_mds.set_ylim(-1.6, 3.0)

    ax_gdv = _thumb_axes(cv, GDV_X, GDV_W, "GDV by layer")
    xc = np.linspace(0, 1, 200)
    ax_gdv.plot(xc, 0.05 - 0.55 * np.exp(-((xc - 0.62) ** 2) / 0.05), color=COL_LLM, lw=0.9)
    ax_gdv.plot(xc, 0.05 - 0.34 * np.exp(-((xc - 0.30) ** 2) / 0.06), color="#8172B3", lw=0.9)
    ax_gdv.axhline(0, color="gray", linestyle=":", lw=0.5, alpha=0.6)
    ax_gdv.set_ylim(-0.78, 0.42)

    ax_beh = _thumb_axes(cv, BEH_X, BEH_W, "Rating behaviour")
    ax_beh.axvline(0.46, color="#cccccc", lw=0.5)
    bottom = 0.0
    for share, c in zip([0.52, 0.23, 0.25], [COL_VIT, COL_LLM, "#999999"]):
        ax_beh.bar(0.20, share, bottom=bottom, width=0.20, color=c, edgecolor="none")
        bottom += share
    rs = [0.42, 0.50, 0.55, 0.48, 0.45]
    ax_beh.bar(np.linspace(0.56, 0.92, len(rs)), rs, width=0.055,
               color=list(colors_a)[: len(rs)], edgecolor="none")
    ax_beh.set_xlim(0, 1); ax_beh.set_ylim(0, 1.30)

    # Three read-out paths, deliberately not merged into one trunk - they do
    # not draw on the same thing:
    #   GDV  <- every LLM layer          (orange, from the middle of the stack)
    #   MDS  <- the projection layers    (grey, from the marked ticks only)
    #   behaviour <- the generated rating
    # Both cross-bars clear the layer labels, which hang 0.045+1 line below the
    # stack; the drop lines pass to the left of each label.
    y_mds = L.PIPE_Y - _fy(0.21)
    y_gdv = L.PIPE_Y - _fy(0.39)
    gdv_cx, mds_cx = GDV_X + GDV_W / 2, MDS_X + MDS_W / 2
    gdv_from = LLM_X + LLM_W / 2

    for x in llm_centers:
        cv.ax.plot([x, x], [L.PIPE_Y - 0.002, y_mds], color=COL_ARROW, lw=0.5,
                   transform=cv.ax.transAxes)
    if llm_centers:
        cv.ax.plot([mds_cx, max(llm_centers)], [y_mds, y_mds], color=COL_ARROW,
                   lw=0.8, transform=cv.ax.transAxes)
        cv.arrow([(mds_cx, y_mds), (mds_cx, L.OUT_Y + L.OUT_H)])
        cv.ax.text(mds_cx + _fx(0.10), y_mds + _fy(0.03),
                   f"{len(llm_centers)} projection layers", fontsize=FS_DESC,
                   color=COL_ARROW, ha="left", va="bottom", transform=cv.ax.transAxes)

    cv.ax.plot([gdv_from, gdv_from], [L.PIPE_Y - 0.002, y_gdv], color=COL_LLM,
               lw=0.9, transform=cv.ax.transAxes)
    cv.ax.plot([gdv_cx, gdv_from], [y_gdv, y_gdv], color=COL_LLM, lw=0.9,
               transform=cv.ax.transAxes)
    cv.arrow([(gdv_cx, y_gdv), (gdv_cx, L.OUT_Y + L.OUT_H)], color=COL_LLM)
    cv.ax.text((gdv_cx + gdv_from) / 2, y_gdv - _fy(0.03), "all layers",
               fontsize=FS_DESC, color=COL_LLM, ha="center", va="top",
               transform=cv.ax.transAxes)

    cv.arrow([(RATE_X + RATE_W / 2, L.PIPE_Y - 0.002), (RATE_X + RATE_W / 2, L.OUT_Y + L.OUT_H)])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_figure(image_path: Optional[Path]) -> plt.Figure:
    _verify_against_registry()
    _verify_tables_complete()
    _verify_task_variants()
    # Style first: the planner measures text, and it must measure it in the
    # font the figure is drawn in (Arial), not matplotlib's wider default.
    set_ccn_style()
    _plan_layout()
    rows_a, rows_b = _load_schemas()

    # Schema-aware persona hues, identical to the scatter figures.
    pal = persona_hue_palette(len(PERSONA_KEYS))
    colors_a = [pal[f"personas_prototypical_persona_{i}"] for i in range(1, 6)]
    colors_b = [pal[f"personas_big5_prototypical_persona_{i}"] for i in range(1, 6)]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(AXES_RECT)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # Canvas measures text in axes fraction, so the axes must already have its
    # final size — never resize the axes after this point.
    cv = Canvas(ax)

    draw_personas(cv, rows_a, rows_b, colors_a, colors_b)
    draw_conditions(cv)
    llm_centers = draw_pipeline(cv, image_path)
    draw_outputs(cv, colors_a, colors_b, llm_centers)
    return fig


def main():
    global FIG_W
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="figure6_setup_overview")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help="Example stimulus shown entering the pipeline.")
    parser.add_argument("--width", type=float, default=FIG_W,
                        help="Figure width in inches (the text width of the target layout).")
    args = parser.parse_args()
    FIG_W = args.width

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(args.image)

    # savefig.bbox must be overridden, not just left unset: set_ccn_style sets
    # it to "tight", and passing bbox_inches=None falls back to that rcParam.
    # Tight-cropping re-derives the page from the drawn content, so the width
    # becomes an outcome (6.976 in) instead of the 7.0 in this figure is laid
    # out for. Forcing "standard" keeps the page exactly FIG_W x FIG_H, which
    # is what makes the point sizes above true point sizes on the page.
    with plt.rc_context({"savefig.bbox": "standard"}):
        for ext, dpi in (("pdf", None), ("png", 400)):
            out = args.output_dir / f"{args.name}.{ext}"
            fig.savefig(out, dpi=dpi, facecolor="white")
            print(f"Saved {out}")

    saved = args.output_dir / f"{args.name}.pdf"
    assert abs(fig.get_size_inches()[0] - FIG_W) < 1e-6, "figure width drifted"
    print(f"Page: {FIG_W} x {FIG_H} in (fits the {FIG_W} in text width) -> {saved}")
    plt.close(fig)


if __name__ == "__main__":
    main()
