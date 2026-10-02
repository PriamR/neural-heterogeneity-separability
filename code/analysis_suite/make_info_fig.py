"""The three information figures for sec:info and sec:division.

The section makes two claims and the old figure carried only the second, so the material is now three figures in
the order the argument runs.

1. F4_5_minmax_oinfo_clean.png, the kind of information: (a) the most redundant subset (max O) and (b) the most
   synergistic subset (min O) per network against subset size k, each on its own scale so the confidence bands are
   visible for both. k = 2 is omitted because the O-information of two variables is identically zero. From the
   O-information records via build_oinfo_figs.
2. F4_5c_info_amount.png, the amount of information: the heterogeneous neural code carries more class information
   at every scale of the pairwise decomposition (single unit, unit pair, redundant part, synergistic part). From
   data_cache/supervisor_extension/rsi_n20.json.
3. F4_26_class_resolution.png, how the class information is resolved in time (sec:division): language and parity
   decoded from the hidden state at each step, the homogeneous network in sequence and the heterogeneous one
   together. From data_cache/analysis_suite/instantaneous_decode_n20.json.

Run: python code/analysis_suite/make_info_fig.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
from build_oinfo_figs import build_records, minmax

REPO = HERE.parent.parent
RSI = REPO / "data_cache" / "supervisor_extension" / "rsi_n20.json"
ID = REPO / "data_cache" / "analysis_suite" / "instantaneous_decode_n20.json"
FIGS = REPO / "docs" / "analysis_suite" / "figs"

COL_LANG = "#1b7837"
COL_PAR = "#762a83"
RK = [4, 6, 8, 10]
BAND_ALPHA = 0.20

ROWS = [
    ("I_marginal", "Single unit\n$I(X_i;Y)$"),
    ("I_joint", "Unit pair\n$I(X_i,X_j;Y)$"),
    ("redundancy", "Redundant part\n$\min(I_i,I_j)$"),
    ("synergy", "Synergistic part\n$I_{ij}-\max(I_i,I_j)$"),
]


def fig_oinfo():
    agg, paired, seeds_by_k = minmax(build_records("clean"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fig.subplots_adjust(wspace=0.28)
    for ax, which, letter, title, marker in ((axes[0], "max", "a", "Most redundant subset, max $O$", "^"),
                                             (axes[1], "min", "b", "Most synergistic subset, min $O$", "v")):
        for a in ("het", "hom"):
            m = np.array([agg[a][which][k].mean() for k in RK])
            h = np.array([S.ci95_mean(agg[a][which][k]) for k in RK])
            S.band(ax, RK, m, h, S.COL[a], marker=marker, ms=5, label=S.LABEL[a], alpha=BAND_ALPHA)
        ax.axhline(0, color=S.CHANCE, lw=1.0)
        ax.set_xticks(RK)
        ax.set_xlabel("Subset size, $k$")
        ax.set_ylabel("O-information (bits)")
        S.panel_title(ax, letter, title)
    axes[0].set_ylim(0, 1.0)
    axes[1].set_ylim(-0.08, 0.01)
    S.legend(axes[0], loc="upper left")
    S.save(fig, FIGS / "F4_5_minmax_oinfo_clean.png")
    for k in (8, 10):
        for which in ("max", "min"):
            r = paired_triple(agg["het"][which][k], agg["hom"][which][k], f"{which} k={k}")
            print(f"  oinfo {which} O at k={k}: het {r['het_mean']:+.3f} hom {r['hom_mean']:+.3f} "
                  f"t_p {r['t_p']:.3f} wins {r['wins']}/{r['n']}")


def fig_amount():
    d = json.load(open(RSI))
    pn = d["per_network"]
    vals = {a: np.array([[pn[f"{s}_{a}"][k] for s in d["seeds"]] for k, _ in ROWS]) for a in ("het", "hom")}
    stats = [paired_triple(vals["het"][i], vals["hom"][i], k) for i, (k, _) in enumerate(ROWS)]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ticks = [f"{lbl}\n$p$ = {S.fmt_p(r['t_p'])}\n{r['wins']}/{r['n']} seeds" for (_, lbl), r in zip(ROWS, stats)]
    S.paired_bars(ax, ticks, vals, stats, width=0.34, test="t", show_stats=False)
    ax.set_ylabel("Class information (bits)")
    ax.set_ylim(0, 0.7)
    ax.tick_params(axis="x", labelsize=8.2)
    S.panel_title(ax, None, "Class information carried by units and pairs")
    S.legend(ax, loc="upper right")
    S.save(fig, FIGS / "F4_5c_info_amount.png")
    for (k, _), r in zip(ROWS, stats):
        print(f"  amount {k:12s} het {r['het_mean']:.3f} hom {r['hom_mean']:.3f} dz {r['dz']:+.2f} "
              f"t_p {r['t_p']:.2g} wins {r['wins']}/{r['n']}")


def fig_class_resolution():
    idc = json.load(open(ID))
    xd = np.array(idc["x_fraction"])
    late0 = int(idc["late_window_frac"] * len(xd))
    late_pct = int(round(100 * (1 - idc["late_window_frac"])))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, arch, letter in ((axes[0], "het", "a"), (axes[1], "hom", "b")):
        ax.axvspan(xd[late0], xd[-1], color=S.CHANCE, alpha=0.08, lw=0)
        for task, c, lab in (("lang", COL_LANG, "Language"), ("par", COL_PAR, "Parity")):
            m = np.array(idc["mean_curves"][arch][task])
            sem = np.array(idc["sem_curves"][arch][task])
            S.band(ax, xd, m, 1.96 * sem, c, label=lab)
        S.chance_line(ax, 0.5, label="Chance")
        ax.set_ylim(0.45, 0.9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Time (fraction of trial)")
        S.panel_title(ax, letter, f"{S.LABEL[arch]}: language and parity decoding")
    axes[0].set_ylabel("Decoding accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(color=S.CHANCE, alpha=0.18, label=f"Late window, last {late_pct}% of the trial"))
    axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    S.save(fig, FIGS / "F4_26_class_resolution.png")


def main():
    S.apply()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_oinfo()
    fig_amount()
    fig_class_resolution()


if __name__ == "__main__":
    main()
