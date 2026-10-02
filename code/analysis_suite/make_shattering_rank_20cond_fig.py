"""Redraw the shattering-against-rank figure from shattering_vs_rank_20cond_n20_summary.json (no recompute).

Companion to make_shattering_rank_fig.py, which draws the eight-condition version. Identical layout and style;
only the source JSON and the output filename differ. The panel title takes its condition count from the JSON,
so it labels itself as the twenty-condition set.

(a) Shattering dimensionality on the twenty-condition set recomputed on rank-m truncations of the neural code,
    both types of network, mean with a 95% confidence band over the twenty seeds.
(b) The paired difference at each rank with its 95% confidence band; the dotted line marks the rank at which
    the difference first reaches 90% of its maximum.

Run: python code/analysis_suite/make_shattering_rank_20cond_fig.py
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

D = json.load(open(OUT / "shattering_vs_rank_20cond_n20_summary.json"))
RANKS = D["ranks"]
seeds = sorted(int(s) for s in D["per_network"]["het"])
M = {a: np.stack([D["per_network"][a][str(s)] for s in seeds]) for a in ("het", "hom")}
delta = M["het"] - M["hom"]
dmean = delta.mean(0)
n = len(seeds)
ci = lambda A: 1.96 * A.std(0, ddof=1) / np.sqrt(n)

thr = 0.9 * dmean.max()
sat = next(m for m, d in zip(RANKS, dmean) if d >= thr)
ncond = D["condition_set"]["n_conditions"]

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
fig.subplots_adjust(wspace=0.28)

ax = axes[0]
for arch in ("het", "hom"):
    S.band(ax, RANKS, M[arch].mean(0), ci(M[arch]), S.COL[arch], label=S.LABEL[arch], marker="o", ms=3.5)
S.chance_line(ax, 0.5, label="Chance")
ax.set_xscale("log"); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
ax.set_xlabel("Retained principal components, $m$")
ax.set_ylabel("Shattering dimensionality")
S.panel_title(ax, "a", f"Shattering against retained rank, {ncond} conditions")
S.legend(ax, loc="upper left")

ax = axes[1]
S.band(ax, RANKS, dmean, ci(delta), S.ACCENT, marker="o", ms=3.5, label="Paired difference")
ax.axhline(0, color=S.CHANCE, lw=1.0)
S.ref_line(ax, x=sat, ls=(0, (1, 2)), label=f"90% of the maximum difference, $m$ = {sat}")
ax.set_xscale("log"); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
ax.set_xlabel("Retained principal components, $m$")
ax.set_ylabel("Shattering difference\n(heterogeneous $-$ homogeneous)")
S.panel_title(ax, "b", "Paired difference against retained rank")
S.legend(ax, loc="upper left")

S.save(fig, FIGS / "F4_25_shattering_vs_rank_20cond.png")
print(f"advantage at m=1: {dmean[0]:+.4f}; max {dmean.max():+.4f} at m={RANKS[int(np.argmax(dmean))]}; "
      f"90% of max first reached at m={sat}")
