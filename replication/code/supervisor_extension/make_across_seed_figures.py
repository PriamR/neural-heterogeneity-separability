"""Across-seed replacements for the structural figures that were drawn from a single network.

The replication exists to move these analyses from n = 1 to n = 20, so a figure in it that shows one
network's matrices is answering the wrong question, however faithful it is to the original. Three figures had
that problem: the edge-statistic heatmaps, and the two community-by-class tuning grids. Each is rebuilt here
so that every panel is a statement about twenty paired seeds.

The single-network versions still exist in make_structure_figures.py, because reproducing the original figure
on the original seed is what establishes the pipeline is faithful. They belong in an appendix, not in the
argument.

Run: python code/supervisor_extension/make_across_seed_figures.py
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
from stats_core import paired_triple

S.apply()
FIGS = SC.FIGS
ARCHS = ("het", "hom")
LANG_C, PAR_C = "#7b3fa0", "#1b9e77"
Z = np.load(SC.DATA / "structure_matrices_n20.npz")
SCAL = json.load(open(SC.DATA / "structure_scalars_n20.json"))
MOD = json.load(open(SC.DATA / "modularity_n20.json"))


def g(seed, arch, field):
    return Z[f"{seed}_{arch}__{field}"]


def tun(seed, arch, kind, field):
    return Z[f"{seed}_{arch}__tun_{kind}_{field}"]


def _annot(ax, r, fmt="{:+.4f}"):
    ax.text(0.5, 0.03, f"$\\Delta$ {fmt.format(r['delta'])}   $d_z$ {r['dz']:+.2f}\n"
                       f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}   wins {r['wins']}/{r['n']}",
            ha="center", va="bottom", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92))


def _paired(ax, vals, title, ylabel, fmt="{:+.4f}", zero=False, legend=False, name=""):
    """One measure as a het-hom bar pair with a 95% CI whisker, the idiom the package uses throughout."""
    r = paired_triple(vals["het"], vals["hom"])
    S.paired_bars(ax, [name], {a: np.array([vals[a]], float) for a in ARCHS}, [r])
    if zero:
        ax.axhline(0, color=S.CHANCE, ls=":", lw=1.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    return r


# ---------------------------------------------------------------- S10: edge statistics, all twenty seeds
def fig_edge_stats():
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.4))

    # (a) the weight distribution itself, every edge of every network
    ax = axes[0]
    for arch in ARCHS:
        v = np.concatenate([g(s, arch, "W")[np.triu_indices(SCAL[f"{s}_{arch}"]["n"], 1)] for s in SC.SEEDS])
        ax.hist(v, bins=90, histtype="step", lw=1.8, color=S.COL[arch], density=True, label=S.LABEL[arch])
    ax.set_yscale("log")
    ax.set_xlabel("null-subtracted MI (bits)")
    ax.set_ylabel("density")
    ax.set_title("(a) Every edge, all twenty seeds", loc="left", fontsize=10)
    ax.legend(fontsize=8)

    # (b) mean edge weight per network
    v = {a: np.array([MOD["per_network"][f"{s}_{a}"]["edge_mi_mean"] for s in SC.SEEDS], float) for a in ARCHS}
    _paired(axes[1], v, "(b) Mean edge MI", "bits", name="mean edge MI")

    # (c) how much of the observed MI the surrogate null accounts for
    v = {a: np.array([MOD["per_network"][f"{s}_{a}"]["null_floor_frac"] for s in SC.SEEDS], float) for a in ARCHS}
    _paired(axes[2], v, "(c) Null floor as a fraction of observed", "fraction", name="null floor")

    # (d) how many pairs survive the surrogate test at all
    v = {}
    for arch in ARCHS:
        out = []
        for s in SC.SEEDS:
            sig = g(s, arch, "sig")
            n = sig.shape[0]
            out.append(sig[np.triu_indices(n, 1)].mean())
        v[arch] = np.array(out, float)
    _paired(axes[3], v, "(d) Significant-edge density", "fraction of pairs", name="significant edges")

    fig.suptitle("Edge statistics across the twenty paired seeds "
                 "(whiskers 95% CI, dots individual networks)", fontweight="bold", y=1.03)
    S.save(fig, FIGS / "S10_edge_stats_n20.png")


# ---------------------------------------------------------------- S12: community tuning, all twenty seeds
def fig_tuning_across_seeds():
    """Community-by-class tuning as twenty paired numbers per row, on both graphs.

    The heatmaps this replaces showed one network's communities against the four classes, which cannot say
    whether an arm encodes a factor. Each panel here is the arm's mean over its own communities, one value
    per network, which can.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    for r, (kind, lbl) in enumerate([("binned", "binned functional graph"),
                                     ("cumulative", "cumulative-count graph")]):
        vals = {f: {a: np.array([tun(s, a, kind, f).mean() for s in SC.SEEDS], float) for a in ARCHS}
                for f in ("eta2_lang", "eta2_par", "eta2_4c", "eta2_4c_frac")}

        # which factor does each arm's communities encode
        ax = axes[r, 0]
        keys = ("eta2_lang", "eta2_par")
        V = {a: np.array([vals[f][a] for f in keys], float) for a in ARCHS}
        S.paired_bars(ax, ["language", "parity"], V,
                      [paired_triple(vals[f]["het"], vals[f]["hom"]) for f in keys])
        ax.set_ylabel(r"mean community $\eta^2$")
        ax.set_title(f"({'ac'[r]}) Which factor? {lbl}", loc="left", fontsize=10)

        # is the class tuning just a global-activity effect
        ax = axes[r, 1]
        keys = ("eta2_4c", "eta2_4c_frac")
        V = {a: np.array([vals[f][a] for f in keys], float) for a in ARCHS}
        S.paired_bars(ax, ["raw $\\eta^2$", "fraction of network"], V,
                      [paired_triple(vals[f]["het"], vals[f]["hom"]) for f in keys])
        ax.axhline(3 / (SC.N_TRIALS - 1), color=S.CHANCE, ls=":", lw=1.2)
        ax.text(0.01, 3 / (SC.N_TRIALS - 1), " chance", fontsize=7.5, color=S.CHANCE, va="bottom",
                transform=ax.get_yaxis_transform())
        ax.set_ylabel(r"mean 4-class $\eta^2$")
        ax.set_title(f"({'bd'[r]}) Not a global-activity artefact, {lbl}", loc="left", fontsize=10)

    S.arch_legend(fig)
    fig.suptitle("Community class tuning across the twenty paired seeds: language separates the arms, "
                 "parity does not", fontweight="bold", y=1.02)
    S.save(fig, FIGS / "S12_tuning_n20.png")


def main():
    fig_edge_stats()
    fig_tuning_across_seeds()

    # the numbers the meeting document quotes for these panels
    print("\nparity and language tuning, both graphs:")
    for kind in ("binned", "cumulative"):
        for f in ("eta2_lang", "eta2_par", "eta2_4c", "eta2_4c_frac"):
            v = {a: np.array([tun(s, a, kind, f).mean() for s in SC.SEEDS], float) for a in ARCHS}
            r = paired_triple(v["het"], v["hom"])
            print(f"  {kind:>11} {f:>14}: het {r['het_mean']:.4f} hom {r['hom_mean']:.4f} "
                  f"d {r['delta']:+.4f} dz {r['dz']:+.2f} p {S.fmt_p(r['perm_p'], r['perm_floor'])} "
                  f"wins {r['wins']}/{r['n']}")
    print("\nedge statistics:")
    for key, lbl in (("edge_mi_mean", "mean edge MI"), ("null_floor_frac", "null floor fraction")):
        v = {a: np.array([MOD["per_network"][f"{s}_{a}"][key] for s in SC.SEEDS], float) for a in ARCHS}
        r = paired_triple(v["het"], v["hom"])
        print(f"  {lbl:>20}: het {r['het_mean']:.4f} hom {r['hom_mean']:.4f} d {r['delta']:+.4f} "
              f"dz {r['dz']:+.2f} p {S.fmt_p(r['perm_p'], r['perm_floor'])} wins {r['wins']}/{r['n']}")
    v = {}
    for arch in ARCHS:
        v[arch] = np.array([g(s, arch, "sig")[np.triu_indices(SCAL[f'{s}_{arch}']['n'], 1)].mean()
                            for s in SC.SEEDS], float)
    r = paired_triple(v["het"], v["hom"])
    print(f"  {'significant-edge density':>20}: het {r['het_mean']:.4f} hom {r['hom_mean']:.4f} "
          f"d {r['delta']:+.4f} dz {r['dz']:+.2f} p {S.fmt_p(r['perm_p'], r['perm_floor'])} "
          f"wins {r['wins']}/{r['n']}")


if __name__ == "__main__":
    main()
