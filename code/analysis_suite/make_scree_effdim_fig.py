"""The dimensional result as one figure: the variance spectrum of the two learned networks beside the
cumulative variance that reads off their effective dimension, with the four fixed-tau rungs for comparison.

(a) Explained-variance ratio per component on a log axis, the two learned networks with 95% confidence
    bands over the twenty seeds, the four fixed-tau rungs dashed (ten seeds each).
(b) Cumulative explained variance, same series, with the 90% line; the legend carries each series' n90.

Reads the per-network records in data_cache/network_json, writes docs/analysis_suite/figs/F4_7_scree_effdim.png.

Run: python code/analysis_suite/make_scree_effdim_fig.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S

REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "network_json"
FIGS = REPO / "docs" / "analysis_suite" / "figs"

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
TEN = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
CONFOUNDED = {202, 210, 340, 440, 550, 660, 710, 820, 930}
RUNGS = [2.0, 3.5, 15.0, 35.0]
TAG = {2.0: "fixed2ms", 3.5: "fixed3p5ms", 15.0: "fixed15ms", 35.0: "fixed35ms"}

R = {}
for f in CACHE.glob("*.json"):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if isinstance(r, dict) and "name" in r:
        R[r["name"]] = r


def hom_name(s):
    nowd = f"4class_seed{s}_hom_geommean_nowd_lognormal"
    return nowd if (s in CONFOUNDED and nowd in R) else f"4class_seed{s}_hom_geommean_lognormal"


het = [R[f"4class_seed{s}_het_lognormal"] for s in SEEDS]
hom = [R[hom_name(s)] for s in SEEDS]
fixed = {t: [R[f"4class_seed{s}_hom_{TAG[t]}_lognormal"] for s in TEN
             if f"4class_seed{s}_hom_{TAG[t]}_lognormal" in R] for t in RUNGS}


def evr(recs):
    return np.array([r["evr_1ms"] for r in recs], dtype=float)


def main():
    S.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    fig.subplots_adjust(wspace=0.28)

    # ---- (a) variance spectrum, log axis
    ax = axes[0]
    for recs, k in ((het, "het"), (hom, "hom")):
        e = evr(recs)
        m = e.mean(0)
        h = 1.96 * e.std(0, ddof=1) / np.sqrt(e.shape[0])
        x = np.arange(1, e.shape[1] + 1)
        ax.plot(x, m, color=S.COL[k], lw=2.0, zorder=4, label=S.LABEL[k])
        ax.fill_between(x, np.clip(m - h, 1e-7, None), m + h, color=S.COL[k], alpha=0.14, lw=0)
    for t in RUNGS:
        e = evr(fixed[t])
        x = np.arange(1, e.shape[1] + 1)
        ax.plot(x, e.mean(0), color=S.FIXED_RAMP[t], lw=1.3, ls=(0, (4, 2)), zorder=2, label=S.FIXED_LABEL[t])
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlim(1, 64)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    S.panel_title(ax, "a", "Variance spectrum")
    S.legend(ax, loc="upper right", fontsize=8)

    # ---- (b) cumulative variance with n90 in the legend
    ax = axes[1]
    order = [("het", het), ("hom", hom)] + [(t, fixed[t]) for t in RUNGS]
    for key, recs in order:
        cum = np.cumsum(evr(recs), axis=1)
        m = cum.mean(0)
        x = np.arange(1, len(m) + 1)
        n90 = np.mean([r["n90_1ms"] for r in recs])     # mean of per-network n90, as quoted in the text
        if key in ("het", "hom"):
            h = 1.96 * cum.std(0, ddof=1) / np.sqrt(cum.shape[0])
            ax.plot(x, m, color=S.COL[key], lw=2.0, zorder=4, label=f"{S.LABEL[key]}, $n_{{90}}$ = {n90:.1f}")
            ax.fill_between(x, m - h, m + h, color=S.COL[key], alpha=0.14, lw=0)
        else:
            ax.plot(x, m, color=S.FIXED_RAMP[key], lw=1.3, ls=(0, (4, 2)), zorder=2,
                    label=f"{S.FIXED_LABEL[key]}, $n_{{90}}$ = {n90:.1f}")
    S.ref_line(ax, y=0.90, ls=(0, (1, 2)))
    ax.set_xlim(1, 64)
    ax.set_ylim(0.2, 1.02)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance")
    S.panel_title(ax, "b", "Cumulative variance")
    S.legend(ax, loc="lower right", fontsize=8)

    FIGS.mkdir(parents=True, exist_ok=True)
    S.save(fig, FIGS / "F4_7_scree_effdim.png")


if __name__ == "__main__":
    main()
