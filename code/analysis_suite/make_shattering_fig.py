"""Standalone shattering figure (decoupled from CCGP). Reads the per-network decodability values already in
geometry_n20_summary.json and plots the three balanced dichotomies for het vs hom, with no CCGP panel.

Run: python code/analysis_suite/make_shattering_fig.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
import matplotlib.pyplot as plt

REPO = HERE.parent.parent
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
S.apply()

G = json.load(open(OUT / "geometry_n20_summary.json"))
seeds = sorted(G["per_network"]["het"], key=int)
cats = ["Language", "Parity", "XOR"]


def vals(arch, field):
    return np.array([G["per_network"][arch][s][field] for s in seeds], float)


fig, ax = plt.subplots(figsize=(7, 4.8))
xp = np.arange(3)
for off, arch, col in [(-0.19, "het", S.HET), (0.19, "hom", S.HOM)]:
    m = [vals(arch, f"dec_{c}").mean() for c in cats]
    sd = [vals(arch, f"dec_{c}").std(ddof=1) for c in cats]
    ax.bar(xp + off, m, 0.36, yerr=sd, color=col, label=S.LABEL[arch], capsize=3)
ax.axhline(0.5, color=S.CHANCE, ls=":", lw=1.2, label="chance (binary)")
ax.set_xticks(xp); ax.set_xticklabels(cats)
ax.set_ylabel("cross-validated decodability")
ax.set_title("Shattering: cross-validated decodability by dichotomy")
ax.set_ylim(0.45, 1.0)
ax.legend(loc="upper right", fontsize=9)
S.save(fig, FIGS / "F4_6b_shattering.png")
print("wrote F4_6b_shattering.png (het shatter %.3f vs hom %.3f)"
      % (G["paired"]["shatter"]["het_mean"], G["paired"]["shatter"]["hom_mean"]))
