"""Cross-seed figures for the extension of the supervisor's information analyses.

Each figure is the n = 20 analogue of one of the seed-1001 figures in the O-Info Review Package. Three things
change in the translation, and all three are the point of the exercise.

  The unit is the network, not the pair or the time bin. A violin over unit pairs, or a Mann-Whitney over
  them, describes one network's internal spread; it cannot speak about architectures, because the pairs
  inside a network are not independent. Every claim here rests on twenty within-seed differences.

  Shaded bands are 95% confidence intervals on the across-seed mean, not standard deviations over seeds. The
  question these figures answer is where each arm's mean sits and whether the two means are separated, which
  is the quantity the paired tests are about. An SD band is wider and invites a different, wrong reading.

  Every bar carries its own paired statistics: the permutation p, and how many of the twenty seeds moved in
  the direction of the mean. The bar shows the size of an effect and the win count shows whether the seeds
  agree about it. Those are different claims and the figure should carry both.

Chart types follow the package rather than this repository's earlier habit. The package compares groups with
bars and error bars, sweeps a parameter with `-o` lines, and shows structure with matrices and spring
layouts; it never draws connected paired slopes. Twenty crossing lines are also simply hard to read, and the
question they answer, whether the seeds agree, is answered better by the win count printed on the bar. So
every group comparison here is a grouped bar with a 95% CI whisker, with the per-seed values as faint dots.

Colours follow the paper's house style, red for heterogeneous and blue for homogeneous, rather than the
package's blue and orange, so these figures sit beside the rest of the thesis without a palette clash.

Run: python code/supervisor_extension/make_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC
import style as S
import matplotlib.pyplot as plt

S.apply()
FIGS = SC.FIGS
ARCHS = ("het", "hom")


def load(name):
    p = SC.DATA / name
    return json.load(open(p)) if p.exists() else None


def _annotate(ax, r, fmt="{:+.3f}", loc=(0.5, 0.02), ha="center"):
    """The paired result, printed on the panel, so a figure lifted into a talk carries its own statistics."""
    ax.text(loc[0], loc[1], f"$\\Delta$ {fmt.format(r['delta'])}   $d_z$ {r['dz']:+.2f}\n"
                            f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}   wins {r['wins']}/{r['n']}",
            ha=ha, va="bottom", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92))


def _seedvals(per, key, path=None):
    out = {}
    for arch in ARCHS:
        v = []
        for s in SC.SEEDS:
            r = per[f"{s}_{arch}"]
            for p in (path or []):
                r = r[p]
            v.append(r[key])
        out[arch] = np.array(v, float)
    return out


def _sweep(per, weighting, field):
    """(n_seeds, n_densities) for one weighting, from each network's own sweep records."""
    out = {}
    for arch in ARCHS:
        rows = []
        for s in SC.SEEDS:
            recs = [r for r in per[f"{s}_{arch}"]["sweep"] if r["weighting"] == weighting]
            recs.sort(key=lambda r: r["target_density"])
            rows.append([r[field] for r in recs])
        out[arch] = np.array(rows, float)
    return out


def _sweep_panel(ax, per, weighting, field, ylabel, title, densities, unit_line=None):
    """One panel of the package's density-sweep figure, with a 95% CI band over the twenty seeds."""
    V = _sweep(per, weighting, field)
    for arch in ARCHS:
        m, h = SC.ci95(V[arch])
        ax.plot(densities, m, "-o", color=S.COL[arch], ms=4, lw=2, label=S.LABEL[arch])
        ax.fill_between(densities, m - h, m + h, color=S.COL[arch], alpha=0.18, lw=0)
    if unit_line is not None:
        ax.axhline(unit_line, color=S.CHANCE, ls=":", lw=1.2)
    ax.set_xlabel("matched edge density")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    return V


# ---------------------------------------------------------------- modularity
def fig_modularity(M):
    """The package's own six-panel density sweep, redrawn as twenty-seed means with 95% CI bands.

    Panel for panel this is `network_analysis.ipynb` cell 19: the same six quantities against the same six
    matched densities, in the same order. The only change is that each line is now a mean over twenty
    networks with a confidence band, instead of one network's value.
    """
    per = M["per_network"]
    d = M["densities"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))

    _sweep_panel(axes[0, 0], per, "weighted", "Q", "modularity $Q$",
                 "(a) $Q$, weighted", d)
    _sweep_panel(axes[0, 1], per, "weighted", "Qnorm", "$Q/Q_{rand}$",
                 "(b) Normalised $Q$, weighted", d, unit_line=1.0)
    _sweep_panel(axes[0, 2], per, "weighted", "ncomm", "number of communities",
                 "(c) Communities, weighted", d)
    _sweep_panel(axes[1, 0], per, "weighted", "sigma", r"$\sigma=(C/C_{rand})/(L/L_{rand})$",
                 "(d) Small-worldness, weighted", d, unit_line=1.0)
    _sweep_panel(axes[1, 1], per, "binary", "Qnorm", "$Q/Q_{rand}$",
                 "(e) Normalised $Q$, binary", d, unit_line=1.0)
    _sweep_panel(axes[1, 2], per, "binary", "sigma", r"$\sigma$",
                 "(f) Small-worldness, binary", d, unit_line=1.0)
    axes[0, 0].legend(loc="best")

    fig.suptitle("Density-matched graph statistics against matched edge density "
                 "(mean of 20 paired seeds, shaded 95% CI)", fontweight="bold", y=1.01)
    S.save(fig, FIGS / "S1_modularity_n20.png")


def fig_modularity_summary(M):
    """The unthresholded graphs, where every network keeps every edge, as grouped bars.

    The sweep above matches density by discarding edges. These are the two graphs the package also reports
    without any thresholding at all: the binned functional graph and the cumulative-count graph.
    """
    per, pa = M["per_network"], M["paired"]
    graphs = ["binned full graph", "cumulative-count graph"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    def two(field):
        return {a: np.array([[per[f"{s}_{a}"][g][field] for s in SC.SEEDS]
                             for g in ("full", "cumulative")], float) for a in ARCHS}

    ax = axes[0]
    S.paired_bars(ax, graphs, two("Qnorm"), [pa["full_Qnorm"], pa["cum_Qnorm"]])
    ax.axhline(1.0, color=S.CHANCE, ls=":", lw=1.2)
    ax.text(0.01, 1.0, " random-null level", fontsize=7.5, color=S.CHANCE, va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_ylabel("$Q/Q_{rand}$")
    ax.set_title("(a) Normalised modularity", loc="left")

    ax = axes[1]
    S.paired_bars(ax, graphs, two("sigma"), [pa["full_sigma"], pa["cum_sigma"]])
    ax.axhline(1.0, color=S.CHANCE, ls=":", lw=1.2)
    ax.set_ylabel(r"$\sigma$")
    ax.set_title("(b) Small-worldness", loc="left")

    ax = axes[2]
    # the cumulative community count has no stored paired row, so it is computed here rather than left
    # unlabelled: a visible bar difference with no p invites the reader to assume one
    from stats_core import paired_triple
    nc = two("ncomm")
    cum_nc = paired_triple(nc["het"][1], nc["hom"][1], "communities, cumulative graph")
    S.paired_bars(ax, graphs, nc, [pa["full_ncomm"], cum_nc])
    ax.set_ylabel("number of communities")
    ax.set_title("(c) Communities found", loc="left")

    fig.suptitle("Unthresholded graphs, where no edge is discarded "
                 "(bars are means of 20 seeds, whiskers 95% CI, dots individual networks)",
                 fontweight="bold", y=1.02)
    S.arch_legend(fig)
    S.save(fig, FIGS / "S1b_modularity_full_n20.png")


# ---------------------------------------------------------------- RSI
def fig_rsi(M):
    """Six paired comparisons in two panels: the raw decomposition, and the same normalised by joint MI."""
    per, pa = M["per_network"], M["paired"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4))  # narrower canvas: same content, larger text once scaled to \linewidth

    raw_keys = ["redundancy", "synergy", "RSI"]
    names = ["redundancy\n$\\min(I_i,I_j)$", "synergy\n$I_{joint}-\\max(I_i,I_j)$",
             "RSI\n$I_{joint}-I_i-I_j$"]

    ax = axes[0]
    V = {a: np.array([_seedvals(per, k)[a] for k in raw_keys], float) for a in ARCHS}
    S.paired_bars(ax, names, V, [pa[k] for k in raw_keys])
    ax.axhline(0, color=S.CHANCE, ls=":", lw=1.2)
    ax.set_ylabel("bits")
    ax.set_title("(a) Raw decomposition", loc="left")

    ax = axes[1]
    nkeys = [k + "_n" for k in raw_keys]
    V = {a: np.array([_seedvals(per, k)[a] for k in nkeys], float) for a in ARCHS}
    S.paired_bars(ax, [n.split("\n")[0] for n in names], V, [pa[k] for k in nkeys])
    ax.axhline(0, color=S.CHANCE, ls=":", lw=1.2)
    ax.set_ylabel("fraction of $I_{joint}$")
    ax.set_title("(b) Normalised by joint MI", loc="left")

    fig.suptitle("Pairwise redundancy and synergy of the class information "
                 "(bars are means of 20 seeds, whiskers 95% CI, dots individual networks)",
                 fontweight="bold", y=1.02)
    S.arch_legend(fig)
    S.save(fig, FIGS / "S2_rsi_n20.png")


def fig_rsi_plane(M):
    """Where pairs sit in the synergy-redundancy plane, pooled over seeds. Descriptive: it shows the shape of
    the pair distribution that the network means summarise, and makes the scale difference visible."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ax = axes[0]
    for arch in ARCHS:
        R, Sy = [], []
        for s in SC.SEEDS:
            A = np.array(M["pair_level"][f"{s}_{arch}"], float)
            R.append(A[:, 3])
            Sy.append(A[:, 4])
        ax.scatter(np.concatenate(R), np.concatenate(Sy), s=5, color=S.COL[arch], alpha=0.20, lw=0,
                   label=S.LABEL[arch])
    ax.axhline(0, color=S.CHANCE, lw=1, ls="--")
    ax.axvline(0, color=S.CHANCE, lw=1, ls="--")
    ax.set_xlabel("redundancy (bits)")
    ax.set_ylabel("synergy (bits)")
    ax.set_title("(a) All pairs, all twenty seeds", loc="left")
    leg = ax.legend(markerscale=3)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    ax = axes[1]
    for arch in ARCHS:
        r = np.array([M["per_network"][f"{s}_{arch}"]["redundancy"] for s in SC.SEEDS], float)
        sy = np.array([M["per_network"][f"{s}_{arch}"]["synergy"] for s in SC.SEEDS], float)
        ax.scatter(r, sy, s=42, color=S.COL[arch], label=S.LABEL[arch], zorder=3, edgecolors="white", lw=0.6)
        # the arm's centroid with a 95% CI cross in both coordinates, in place of per-seed connectors
        ax.errorbar(r.mean(), sy.mean(), xerr=S.ci95_mean(r), yerr=S.ci95_mean(sy),
                    color=S.COL[arch], marker="s", ms=10, mec="black", mew=1.0, lw=2.0,
                    capsize=4, zorder=5)
    ax.set_xlabel("redundancy (bits)")
    ax.set_ylabel("synergy (bits)")
    ax.set_title("(b) Network means, squares mark the arm mean $\\pm$ 95% CI", loc="left")
    ax.legend()
    fig.suptitle("The synergy-redundancy plane", fontweight="bold", y=1.02)
    S.save(fig, FIGS / "S3_rsi_plane_n20.png")


# ---------------------------------------------------------------- temporal
def fig_temporal(M):
    per, pa, prof = M["per_network"], M["paired"], M["profiles"]
    tv = np.arange(SC.T_STEPS) * SC.DT_MS
    gms = np.array(M["acc_grid_ms"], float)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))

    # (a) instantaneous 4-class information, each arm's mean profile with a CI band
    ax = axes[0, 0]
    for arch in ARCHS:
        P = np.array([SC.smooth(prof[f"{s}_{arch}"]["inst"]["4-class"]) for s in SC.SEEDS], float)
        m, h = SC.ci95(P)
        ax.plot(tv, m, color=S.COL[arch], label=S.LABEL[arch])
        ax.fill_between(tv, m - h, m + h, color=S.COL[arch], alpha=0.20, lw=0)
    ax.set_xlabel("time in trial (ms)")
    ax.set_ylabel("mean pairwise JMI (bits)")
    ax.set_title("(a) Instantaneous 4-class information", loc="left")
    ax.legend()

    # (b) the same profiles normalised, so the comparison is timing not amplitude
    ax = axes[0, 1]
    for arch in ARCHS:
        P = np.array([SC.smooth(prof[f"{s}_{arch}"]["inst"]["4-class"]) for s in SC.SEEDS], float)
        P = P / P.max(1, keepdims=True)
        m, h = SC.ci95(P)
        ax.plot(tv, m, color=S.COL[arch], label=S.LABEL[arch])
        ax.fill_between(tv, m - h, m + h, color=S.COL[arch], alpha=0.20, lw=0)
    ax.set_xlabel("time in trial (ms)")
    ax.set_ylabel("JMI (fraction of own max)")
    ax.set_title("(b) Normalised shape: timing and sustainedness", loc="left")
    ax.legend()

    # (c) centre of mass per target
    ax = axes[0, 2]
    names = ["4-class", "language", "parity"]
    x = np.arange(3)
    w = 0.36
    for i, arch in enumerate(ARCHS):
        vals = np.array([[per[f"{s}_{arch}"][f"inst_{n}_com_ms"] for n in names] for s in SC.SEEDS], float)
        m, h = SC.ci95(vals)
        ax.bar(x + (i - 0.5) * w, m, w, yerr=h, color=S.COL[arch], label=S.LABEL[arch], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("centre of mass (ms)")
    ax.set_title("(c) Information timescale", loc="left")

    # (d,e) the accumulation curves that the lock-in times come from
    for c, arch in enumerate(ARCHS):
        ax = axes[1, c]
        for name, ls, col in [("language", "-", "#7b3fa0"), ("parity", "--", "#1b9e77")]:
            P = np.array([prof[f"{s}_{arch}"]["acc"][name] for s in SC.SEEDS], float)
            P = P / P[:, -3:].mean(1, keepdims=True)
            m, h = SC.ci95(P)
            ax.plot(gms, m, ls, color=col, label=name)
            ax.fill_between(gms, m - h, m + h, color=col, alpha=0.18, lw=0)
        ax.axhline(0.9, color=S.CHANCE, ls=":", lw=1.0)
        ax.text(0.02, 0.9, " lock-in threshold", fontsize=7.5, color=S.CHANCE, va="bottom",
                transform=ax.get_yaxis_transform())
        ax.set_xlabel("time in trial (ms)")
        ax.set_ylabel("accumulated JMI (fraction of asymptote)" if c == 0 else "")
        ax.set_title(f"({'de'[c]}) {S.LABEL[arch]}: language against parity", loc="left")
        ax.legend()

    # (f) the ordering question itself, one bar per arm
    ax = axes[1, 2]
    v = {a: np.array([[per[f"{s}_{a}"]["acc_gap_ms"] for s in SC.SEEDS]], float) for a in ARCHS}
    S.paired_bars(ax, ["accumulated lock-in"], v, [pa["acc_gap_ms"]])
    ax.axhline(0, color=S.CHANCE, ls=":", lw=1.2)
    ax.set_ylabel("parity lock-in $-$ language lock-in (ms)")
    ax.set_title("(f) Above zero means language resolves first", loc="left")

    fig.suptitle("When each factor of the task resolves (n = 20 paired seeds)", fontweight="bold", y=1.01)
    S.save(fig, FIGS / "S4_temporal_n20.png")


def fig_lockin_detail(M):
    """The ordering claim on its own, because it is the one the package's reading turns on.

    The comparison that matters here is within an arm, language against parity, not between arms. So the
    het-vs-hom triple is shown where it applies (panel a, per factor) and the within-arm sign test is
    printed where it applies (panel b).
    """
    per, pa, within = M["per_network"], M["paired"], M["within_arm_gap"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # (a) when each factor locks in, by arm
    ax = axes[0]
    V = {a: np.array([[per[f"{s}_{a}"][f"acc_{f}_lockin_ms"] for s in SC.SEEDS]
                      for f in ("language", "parity")], float) for a in ARCHS}
    S.paired_bars(ax, ["language", "parity"], V,
                  [pa["acc_language_lockin_ms"], pa["acc_parity_lockin_ms"]])
    ax.set_ylabel("accumulated lock-in time (ms)")
    ax.set_title("(a) When each factor locks in", loc="left", fontsize=10)

    # (b) the within-arm gap on both clocks, against zero
    ax = axes[1]
    V = {a: np.array([[per[f"{s}_{a}"][k] for s in SC.SEEDS]
                      for k in ("inst_gap_ms", "acc_gap_ms")], float) for a in ARCHS}
    x = S.paired_bars(ax, ["instantaneous\n(centre of mass)", "accumulated\n(lock-in)"], V, None)
    ax.axhline(0, color=S.CHANCE, ls=":", lw=1.2)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.26)
    for j, key in enumerate(("inst_gap_ms", "acc_gap_ms")):
        for i, arch in enumerate(ARCHS):
            w = within[f"{arch}_{key}"]
            ax.text(x[j] + (i - 0.5) * 0.36, hi + (hi - lo) * 0.02,
                    f"{w['n_language_first']}/{w['n']}\n{S.fmt_p(w['p'], w['floor'])}",
                    ha="center", va="bottom", fontsize=7.4, color=S.COL[arch], linespacing=1.25)
    ax.set_ylabel("parity $-$ language (ms)")
    ax.set_title("(b) Language-first count and within-arm sign test", loc="left", fontsize=10)

    # (c) tally of orderings
    ax = axes[2]
    cats = ["language-first", "concurrent", "parity-first"]
    x = np.arange(3)
    w = 0.36
    for i, arch in enumerate(ARCHS):
        cc = M["order_counts"][arch]
        ax.bar(x + (i - 0.5) * w, [cc.get(c, 0) for c in cats], w, color=S.COL[arch], label=S.LABEL[arch])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("number of seeds")
    ax.set_title(f"(c) Ordering tally (tolerance {M['tolerance_ms']:.0f} ms)", loc="left", fontsize=10)

    fig.suptitle("Does either network resolve language before parity? "
                 "(whiskers 95% CI, dots individual networks)", fontweight="bold", y=1.03)
    S.arch_legend(fig)
    S.save(fig, FIGS / "S5_lockin_order_n20.png")


def fig_division_of_labour(M):
    """The keystone claim: module identity predicts what a unit codes for in one arm and not the other."""
    per, pa = M["per_network"], M["paired"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # (a) the effect, raw and above each network's own permutation null, since eta2 rises with module count
    ax = axes[0]
    V = {a: np.array([_seedvals(per, k)[a] for k in ("eta2_SI_module", "eta2_excess")], float)
         for a in ARCHS}
    S.paired_bars(ax, ["raw $\\eta^2$", "above own\npermutation null"], V,
                  [pa["eta2_SI_module"], pa["eta2_excess"]])
    ax.axhline(0, color=S.CHANCE, ls=":", lw=1.2)
    ax.set_ylabel(r"$\eta^2$(selectivity $\sim$ module)")
    ax.set_title("(a) Does module identity predict factor preference?", loc="left", fontsize=10)

    # (b) how many individual networks are organised at all
    ax = axes[1]
    sig = M["n_significant_within_network"]
    n = len(SC.SEEDS)
    x = np.arange(2)
    ax.bar(x, [sig[a] for a in ARCHS], 0.5, color=[S.COL[a] for a in ARCHS])
    ax.axhline(0.05 * n, color=S.CHANCE, ls=":", lw=1.2)
    ax.text(0.02, 0.05 * n, " expected by chance", fontsize=7.5, color=S.CHANCE, va="bottom",
            transform=ax.get_yaxis_transform())
    for i, a in enumerate(ARCHS):
        ax.text(i, sig[a] + 0.3, f"{sig[a]}/{n}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([S.LABEL[a] for a in ARCHS])
    ax.set_ylabel("networks with p < 0.05")
    ax.set_ylim(0, n)
    ax.set_title("(b) Networks organised by module", loc="left", fontsize=10)

    fig.suptitle("Division of labour: do language and parity live in different modules?",
                 fontweight="bold", y=1.02)
    S.arch_legend(fig)
    S.save(fig, FIGS / "S6_division_of_labour_n20.png")


def main():
    M = load("modularity_n20.json")
    if M:
        fig_modularity(M)
        fig_modularity_summary(M)
    R = load("rsi_n20.json")
    if R:
        fig_rsi(R)
        fig_rsi_plane(R)
    T = load("temporal_lockin_n20.json")
    if T:
        fig_temporal(T)
        fig_lockin_detail(T)
    D = load("division_of_labour_n20.json")
    if D:
        fig_division_of_labour(D)
    if not any((M, R, T, D)):
        print("no result JSON found; run the run_*_n20.py scripts first")


if __name__ == "__main__":
    main()
