"""Rating numbers of the paper on the 100-image set (the images with hidden states).

Prints the cell counts, the Base re-run and Base/D1/D2 agreement, Table C1
(variance components with bootstrap intervals, mixed-model LRTs) as an
independent re-computation, and Table C2 (mean correlation between the 25
persona x style conditions by what a pair shares). Own loader, own
decomposition, own mixed-model fits; does not import the figure scripts.

    python analysis/paper_numbers.py
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CI_SCORES, IMAGE_IDS_100, RATINGS as R  # noqa: E402

IDS = {l.strip() for l in open(IMAGE_IDS_100) if l.strip()}
MAP = {"Not Interesting": 1, "Slightly Interesting": 2, "Moderately Interesting": 3,
       "Very Interesting": 4, "Extremely Interesting": 5}
SCHEMAS = {"personas_prototypical": "socio", "personas_big5_prototypical": "big5"}
STYLES = {"": "S1", "_P1": "S2", "_P2": "S3", "_P3": "S4", "_P4": "S5"}


def parse(raw):
    t = re.sub(r"^```(json)?|```$", "", str(raw).strip(), flags=re.M).strip()
    try:
        v = json.loads(t).get("interestingness", "")
    except Exception:
        return np.nan
    return MAP.get(re.sub(r"[^a-zA-Z ]", "", v).strip(), np.nan)


def load(folder):
    d = pd.read_csv(R / folder / "persona_text_outputs.csv", dtype={"image_id": str})
    d["r"] = d.raw_response.map(parse)
    return d[d.image_id.isin(IDS)]


print("== Image set")
ci = pd.read_csv(CI_SCORES, dtype={"image_id": str})
print("ids:", len(IDS), "| all in CI file:", IDS <= set(ci.image_id))

# ---- load everything
base_runs, pers = {}, []
for sch, s_name in SCHEMAS.items():
    for sfx, sty in STYLES.items():
        d = load(sch + sfx)
        b = d[d.persona_id == "baseline"]
        if len(b):
            base_runs[f"{s_name}_{sty}"] = b.set_index("image_id").r
        p = d[d.persona_id != "baseline"].assign(schema=s_name, style=sty)
        pers.append(p)
pers = pd.concat(pers, ignore_index=True)
d1, d2 = (load(f"defaults_D{k}").set_index("image_id").r for k in (1, 2))
# Base: the S2 run (identical in all S2-S5 runs, see the Base re-run section).
base = base_runs["big5_S2"]

print("\n== Cells")
n_pers = len(pers.drop_duplicates(["image_id", "schema", "persona_id", "style"]))
print("persona cells:", n_pers, "| Base", base.size, "D1", d1.size, "D2", d2.size,
      "| total", n_pers + base.size + d1.size + d2.size, "(paper 5,300)")
allr = pd.concat([pers.r, base, d1, d2])
print("unparseable:", int(allr.isna().sum()))
vc = allr.value_counts().sort_index()
print("distribution:", vc.to_dict())
print(f"extremes 1 or 5: {allr.isin([1, 5]).mean():.2%} | share 2-4: {allr.isin([2, 3, 4]).mean():.1%}")
print("prompts per image:", 3 + 5 * 2 * 5)

print("\n== Base re-runs (S2-S5 schema x style folders)")
B = pd.DataFrame(base_runs)
nu = B.nunique(axis=1)
print("runs:", B.shape[1], "| identical in all runs:", int((nu == 1).sum()), "of", len(B))
for img, row in B[nu > 1].iterrows():
    print("  varying:", img, row.value_counts().to_dict())

print("\n== Base / D1 / D2")
ref = pd.DataFrame({"Base": base, "D1": d1, "D2": d2})
print("D1==D2:", int((ref.D1 == ref.D2).sum()), "| Base==D1:", int((ref.Base == ref.D1).sum()),
      "| Base==D2:", int((ref.Base == ref.D2).sum()), "| all three equal:",
      int(((ref.Base == ref.D1) & (ref.D1 == ref.D2)).sum()))
print("r:", ref.corr().round(3).to_dict())

# ---- variance decomposition (own implementation)
def decomp(s):
    y = s.r.to_numpy(float); g = y.mean(); tot = ((y - g) ** 2).sum()
    def ss(cols):
        m = s.groupby(cols).r.agg(["mean", "count"])
        return float((m["count"] * (m["mean"] - g) ** 2).sum())
    I, P, S = ss("image_id"), ss("persona_id"), ss("style")
    PS = ss(["persona_id", "style"]) - P - S
    IP = ss(["image_id", "persona_id"]) - I - P
    IS = ss(["image_id", "style"]) - I - S
    res = tot - I - P - S - PS - IP - IS
    return {k: 100 * v / tot for k, v in
            dict(Image=I, Persona=P, Style=S, PxS=PS, IxP=IP, IxS=IS, IxPxS=res).items()}


def boot(s, n=500, seed=42):
    rng = np.random.default_rng(seed)
    imgs = s.image_id.unique(); g = {i: x for i, x in s.groupby("image_id")}
    out = []
    for _ in range(n):
        pick = rng.choice(imgs, len(imgs), replace=True)
        out.append(decomp(pd.concat([g[i].assign(image_id=f"{k}_{i}") for k, i in enumerate(pick)])))
    b = pd.DataFrame(out)
    return {c: (round(b[c].mean(), 1), round(np.percentile(b[c], 2.5), 1), round(np.percentile(b[c], 97.5), 1)) for c in b}


def mixed(s):
    import statsmodels.formula.api as smf
    s = s.assign(persona=s.persona_id.astype("category"), sty=s["style"].astype("category"))
    vc = {"ip": "0 + C(persona)", "is": "0 + C(sty)"}
    def fit(f, v):
        best, llfs = None, {}
        for opt in ("lbfgs", "powell", "nm"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = smf.mixedlm(f, s, groups=s.image_id, re_formula="1", vc_formula=v).fit(
                        reml=False, method=opt, maxiter=5000)
                llfs[opt] = round(r.llf, 2)
                if best is None or r.llf > best.llf:
                    best = r
            except Exception as e:
                llfs[opt] = type(e).__name__
        return best, llfs
    F, lf = fit("r ~ C(persona) * C(sty)", vc)
    A, la = fit("r ~ C(persona) + C(sty)", vc)
    Po, _ = fit("r ~ C(persona)", vc)
    So, _ = fit("r ~ C(sty)", vc)
    nIP, _ = fit("r ~ C(persona) * C(sty)", {"is": vc["is"]})
    nIS, _ = fit("r ~ C(persona) * C(sty)", {"ip": vc["ip"]})
    print("  llf per optimiser, full:", lf, "| additive:", la)
    # Plain chi2 p-values, as in the paper (no halving for the variance terms).
    for name, a, b, df in [("Persona", A, So, 4), ("Style", A, Po, 4), ("PxS", F, A, 16),
                           ("IxP var", F, nIP, 1), ("IxS var", F, nIS, 1)]:
        c = 2 * (a.llf - b.llf)
        print(f"  {name:8s} chi2({df}) = {c:.1f}  p = {stats.chi2.sf(c, df):.2e}")


for s_name in ("socio", "big5"):
    s = pers[pers.schema == s_name].dropna(subset=["r"])
    print(f"\n== {s_name}: n = {len(s)}, cells per persona x style:",
          s.groupby(["persona_id", "style"]).size().unique().tolist())
    print("  point:", {k: round(v, 1) for k, v in decomp(s).items()})
    print("  bootstrap mean [2.5, 97.5]:", boot(s))
    print("  persona means:", s.groupby("persona_id").r.mean().round(2).to_dict())
    print("  style means:", s.groupby("style").r.mean().round(2).to_dict())
    # correlation structure over 25 persona x style conditions
    W = s.pivot_table(index="image_id", columns=["persona_id", "style"], values="r")
    Cm = W.corr(); keys = list(Cm.columns)
    sp, ss_, ne = [], [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (p1, s1), (p2, s2) = keys[i], keys[j]
            (sp if p1 == p2 else ss_ if s1 == s2 else ne).append(Cm.iat[i, j])
    print(f"  Table C2, mean r: same persona {np.mean(sp):.3f} | same style {np.mean(ss_):.3f} | neither {np.mean(ne):.3f}")
    blocks = {p: np.mean([Cm.loc[(p, a), (p, b)] for a in STYLES.values() for b in STYLES.values() if a < b])
              for p in sorted(s.persona_id.unique())}
    print("  within-persona block mean r:", {k: round(v, 2) for k, v in blocks.items()})
    mixed(s)
