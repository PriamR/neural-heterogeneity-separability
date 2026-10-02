"""F4_20: the separability bridge.

  F4_20_separability_bridge  -- two panels across all forty networks:
                                (a) shattering dimensionality (twenty conditions) against task accuracy
                                (b) manifold capacity alpha_M (twenty conditions) against task accuracy
                                Both separability readings are independent of the trained task, so the step
                                from geometry to accuracy is observed on label-free measures.

Pulls per-network shattering from shattering_n20_summary.json, capacity from capacity_corr_n20_summary.json,
and recomputes per-network accuracy from the representation cache (pred == y). Writes the correlations to
separability_bridge.json so the numbers quoted in the mediation section regenerate like every other.

Run: python code/analysis_suite/make_separability_bridge.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
import matplotlib.pyplot as plt

REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "representation_clean"
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
SH = json.load(open(OUT / "shattering_n20_summary.json"))
CAP = json.load(open(OUT / "capacity_corr_n20_summary.json"))

acc = {"het": {}, "hom": {}}
for arch in ("het", "hom"):
    for s in SEEDS:
        d = np.load(CACHE / f"repr_seed{s}_{arch}.npz")
        acc[arch][s] = float((d["pred"] == d["y"]).mean())

sh20 = SH["per_network"]["B_20cond"]
sh8 = SH["per_network"]["A_8cond"]
cap20 = CAP["per_network"]["B_20cond"]


def series(src, field=None):
    xs, ys = [], []
    for arch in ("het", "hom"):
        for s in SEEDS:
            v = src[arch][str(s)]
            if field is not None:
                v = v[field]
            elif isinstance(v, dict):
                v = v["SD"]
            xs.append(float(v)); ys.append(acc[arch][s])
    return np.array(xs), np.array(ys)


panels = [
    ("sh20", *series(sh20), "Shattering dimensionality, twenty conditions",
     "Shattering dimensionality against accuracy"),
    ("cap20", *series(cap20, "alpha_M"), "Manifold capacity, $\\alpha_M$", "Manifold capacity against accuracy"),
]
x8, y8 = series(sh8)
r8, p8 = pearsonr(x8, y8)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
fig.subplots_adjust(wspace=0.26)
stats = {"n": len(SEEDS) * 2, "shattering_8cond": {"r": float(r8), "p": float(p8)}}
for i, (ax, (key, xs, ys, xlab, title)) in enumerate(zip(axes, panels)):
    r, p = pearsonr(xs, ys)
    stats[{"sh20": "shattering_20cond", "cap20": "capacity_20cond"}[key]] = {"r": float(r), "p": float(p)}
    for arch in ("het", "hom"):
        m = xs[:20] if arch == "het" else xs[20:]
        a = ys[:20] if arch == "het" else ys[20:]
        S.dots(ax, m, a, S.COL[arch], label=S.LABEL[arch], s=40)
    b1, b0 = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xx, b1 * xx + b0, color=S.ACCENT, lw=1.4, ls=(0, (4, 2)), zorder=2,
            label=f"Linear fit, $r$ = {r:.2f}, $p$ = {S.fmt_p(p)}")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Task accuracy")
    S.panel_title(ax, "ab"[i], title)
    S.legend(ax, loc="lower right")
S.save(fig, FIGS / "F4_20_separability_bridge.png")

json.dump(stats, open(OUT / "separability_bridge.json", "w"), indent=1)
for k, v in stats.items():
    if isinstance(v, dict):
        print(f"{k}: r = {v['r']:.3f}, p = {v['p']:.2e}")
print("wrote F4_20_separability_bridge.png, separability_bridge.json")
