"""Redraw the Part I paired figures as grouped bars, from the stored summary JSON.

Five figures in the meeting document drew twenty connected het-hom slopes. That idiom appears nowhere in the
package this work replies to, and twenty crossing lines are hard to read: the eye follows individual threads
instead of the comparison. The information the slopes carried, whether the seeds agree, is carried better by
the win count printed on each bar and by the per-seed dots.

Two of the capacity figures were already bars, but with standard-deviation whiskers, while every other figure
in the document uses a 95% confidence interval on the mean. Redrawing all three from one place removes that
inconsistency: the whisker means the same thing everywhere.

Nothing is recomputed. Every number comes from the summary JSON the original analyses wrote, so these figures
cannot drift from the tables.

Run: python code/analysis_suite/redraw_paired_figs.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
import matplotlib.pyplot as plt

S.apply()
DOCS = HERE.parent.parent / "docs" / "analysis_suite"
FIGS = DOCS / "figs"
SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930,
         1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
ARCHS = ("het", "hom")

CAP_PANELS = [("alpha_M", "capacity $\\alpha_M$", True),
              ("R_M", "radius $R_M$", False),
              ("D_M", "dimension $D_M$", False),
              ("center_corr", "centre correlation", False)]


def _load(name):
    return json.load(open(DOCS / name))


# ---------------------------------------------------------------- manifold capacity, all three sets
def fig_capacity(set_name, tag, nice, corrected=True):
    D = _load("capacity_corr_n20_summary.json")
    per, pa = D["per_network"][set_name], D["paired"][set_name]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.4))
    for ax, (key, ylab, higher_better) in zip(axes, CAP_PANELS):
        V = {a: np.array([[per[a][str(s)][key] for s in SEEDS]], float) for a in ARCHS}
        # paired_bars expects wins = het-higher count; the summary JSON stores predicted-direction
        # wins (het-lower seeds) for the lower-is-better panels, so convert before annotating.
        r = dict(pa[key])
        if not higher_better:
            r["wins"] = r["n"] - r["wins"]
        S.paired_bars(ax, [""], V, [r])
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab}\n({'higher' if higher_better else 'lower'} = more separable)", fontsize=10)
    S.arch_legend(fig)
    est = "correlation-corrected" if corrected else "uncorrected"
    fig.suptitle(f"Manifold capacity, {est} estimator ({nice}, n = 20; whiskers 95% CI, "
                 f"dots individual networks)", fontweight="bold", y=1.03)
    S.save(fig, FIGS / tag)


# ---------------------------------------------------------------- shattering dimensionality
def fig_shattering_paired():
    D = _load("shattering_n20_summary.json")
    sets = [("A_8cond", "8 conditions\n(language x parity x magnitude)"),
            ("B_20cond", "20 conditions\n(digit x language)")]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    V = {a: np.array([[D["per_network"][s][a][str(sd)]["SD"] for sd in SEEDS] for s, _ in sets], float)
         for a in ARCHS}
    S.paired_bars(ax, [lbl for _, lbl in sets], V, [D["paired"][s] for s, _ in sets])
    ax.set_ylabel("shattering dimensionality")
    ax.set_ylim(0.5, ax.get_ylim()[1])
    S.arch_legend(fig)
    fig.suptitle("Shattering dimensionality on both condition sets\n"
                 "(whiskers 95% CI, dots individual networks)", fontweight="bold", y=1.04)
    S.save(fig, FIGS / "F4_11b_shattering_paired.png")


# ---------------------------------------------------------------- VC (Cover) shattering
def fig_vc():
    D = _load("vc_shattering_n20_summary.json")
    P = D["loads"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ax = axes[0]
    for arch in ARCHS:
        M = np.array([[D["per_network"][arch][str(s)].get("frac_curve", [])] for s in SEEDS], float) \
            if "frac_curve" in D["per_network"][arch][str(SEEDS[0])] else None
        curve = np.array(D["by_arch"][arch]["frac_curve_mean"], float)
        ax.plot(P, curve, "-o", color=S.COL[arch], ms=3.5, label=S.LABEL[arch])
        if M is not None:
            m = M[:, 0, :]
            h = np.array([S.ci95_mean(m[:, j]) for j in range(m.shape[1])])
            ax.fill_between(P, curve - h, curve + h, color=S.COL[arch], alpha=0.18, lw=0)
    ax.axhline(0.5, color=S.CHANCE, ls=":", lw=1.2, label="50% crossing")
    ax.axvline(2 * D["n_features"], color=S.ACCENT, ls="--", lw=1.0, alpha=0.6)
    ax.annotate("Cover bound\nfor general position\n$P = 2N$", xy=(2 * D["n_features"], 0.72),
                xytext=(2 * D["n_features"] * 0.55, 0.82), fontsize=8, color=S.ACCENT,
                arrowprops=dict(arrowstyle="->", color=S.ACCENT, lw=0.9))
    ax.set_xlabel("number of points $P$ given random labels")
    ax.set_ylabel("fraction linearly separable")
    ax.set_title("Shattering breaks down as load rises", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    V = {a: np.array([[D["per_network"][a][str(s)]["alpha_vc"] for s in SEEDS]], float) for a in ARCHS}
    S.paired_bars(ax, ["$\\alpha_{VC}$"], V, [D["paired"]["alpha_vc"]])
    ax.axhline(D["cover_bound_general_position"], color=S.ACCENT, ls="--", lw=1.0, alpha=0.6)
    ax.text(0.01, D["cover_bound_general_position"], " Cover bound (general position)", fontsize=7.5,
            color=S.ACCENT, va="bottom", transform=ax.get_yaxis_transform())
    ax.set_ylabel(r"$\alpha_{VC} = P^{*}/N$")
    ax.set_title("Shattering capacity", fontsize=10)

    S.arch_legend(fig)
    fig.suptitle("VC (Cover) shattering capacity, a negative control (n = 20)", fontweight="bold", y=1.03)
    S.save(fig, FIGS / "F4_26_vc_shattering.png")


def main():
    fig_capacity("A_8cond", "F4_16b_capacity_corr_A_8cond.png", "8 conditions")
    fig_capacity("B_20cond", "F4_16b_capacity_corr_B_20cond.png", "20 conditions")
    fig_capacity("T_4class", "F4_16c_capacity_corr_T_4class.png", "4 trained task classes",
                 corrected=False)
    fig_shattering_paired()
    fig_vc()


if __name__ == "__main__":
    main()
