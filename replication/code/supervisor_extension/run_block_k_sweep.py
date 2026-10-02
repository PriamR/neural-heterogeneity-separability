"""Does the block-concentration result depend on how many communities you force the partition into?

The aggregate figure cuts every network into exactly three communities so the block boundaries land in
comparable places and an average across networks means something. That is a real assumption and it deserves
testing, because Louvain finds slightly more communities in the heterogeneous arm (3.50 against 3.10, though
the difference is not significant at p = 0.136).

Two ways it could bite, pointing opposite ways:

  Forcing K below an arm's natural count merges communities that arm would have kept apart. Merging moves
  pairs from the between-community set into the within-community set, so it raises the merged arm's
  within-community mean and lowers its between-community mean. Since the heterogeneous arm is the one with
  more natural communities, forcing K = 3 biases *against* the reported result that heterogeneous networks
  carry more between-community weight. The finding survives the bias rather than being produced by it.

  Forcing K above an arm's natural count splits a genuine community, which does the reverse.

Rather than argue about which dominates, the statistic is recomputed at every K from 2 to 6 and the paired
comparison is reported at each. A conclusion that holds across the sweep does not depend on the choice.

Giving each arm its own K, which is the natural thing to try, is worse than it looks: with different K the
two arms have different numbers of within-pairs and between-pairs, so the within and between means are not
comparable quantities and the aggregate pictures cannot be read side by side. The sweep answers the same
question without that problem.

A K-free check is included as well: the profile of edge weight against distance from the diagonal, computed
on each network's own Louvain partition, whatever size that is.

Writes data_cache/supervisor_extension/block_k_sweep.json.

Run: python code/supervisor_extension/run_block_k_sweep.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC
import style as S
import matplotlib.pyplot as plt
from stats_core import paired_triple
from make_aggregate_block_figure import spectral_partition

warnings.filterwarnings("ignore")
S.apply()
FIGS = SC.FIGS
ARCHS = ("het", "hom")
K_RANGE = [2, 3, 4, 5, 6]
Z = np.load(SC.DATA / "structure_matrices_n20.npz")
SCAL = json.load(open(SC.DATA / "structure_scalars_n20.json"))


def within_between(W, lab):
    """Mean edge weight inside communities and between them, each scaled by the network's own mean."""
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(W.shape[0], 1)
    w, sm = np.asarray(W, float)[iu], same[iu]
    scale = w.mean()
    if scale <= 0 or not sm.any() or not (~sm).any():
        return np.nan, np.nan
    return float(w[sm].mean() / scale), float(w[~sm].mean() / scale)


def louvain_profile(W, modid, nbin=24):
    """K-free view: weight against normalised distance from the diagonal, on the network's own partition."""
    order = np.concatenate([np.where(modid == c)[0] for c in range(modid.max() + 1)])
    A = np.asarray(W, float)[np.ix_(order, order)]
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    scale = A[iu].mean()
    d = np.abs(iu[0] - iu[1]) / n
    w = A[iu] / scale if scale > 0 else A[iu]
    edges = np.linspace(0, 1, nbin + 1)
    b = np.clip(np.digitize(d, edges[1:-1]), 0, nbin - 1)
    return np.array([w[b == i].mean() if (b == i).any() else np.nan for i in range(nbin)])


def main():
    ncomm = {a: np.array([SCAL[f"{s}_{a}"]["ncomm"] for s in SC.SEEDS], float) for a in ARCHS}
    r = paired_triple(ncomm["het"], ncomm["hom"])
    print(f"Louvain community count: het {r['het_mean']:.2f}  hom {r['hom_mean']:.2f}  "
          f"d {r['delta']:+.2f}  p {S.fmt_p(r['perm_p'], r['perm_floor'])}  het more on {r['wins']}/20\n")

    res = {"ncomm": {a: ncomm[a].tolist() for a in ARCHS}, "by_k": {}}
    print(f"{'K':>3} | {'within het':>10} {'within hom':>10} {'p':>9} | "
          f"{'between het':>11} {'between hom':>11} {'p':>9} | {'ratio het':>9} {'ratio hom':>9} {'p':>9}")
    for K in K_RANGE:
        wb = {a: [] for a in ARCHS}
        for a in ARCHS:
            for s in SC.SEEDS:
                W = Z[f"{s}_{a}__W"]
                wb[a].append(within_between(W, spectral_partition(W, K, s)))
        arr = {a: np.array(wb[a], float) for a in ARCHS}
        rw = paired_triple(arr["het"][:, 0], arr["hom"][:, 0])
        rb = paired_triple(arr["het"][:, 1], arr["hom"][:, 1])
        rr = paired_triple(arr["het"][:, 0] / arr["het"][:, 1], arr["hom"][:, 0] / arr["hom"][:, 1])
        res["by_k"][str(K)] = {"within": rw, "between": rb, "ratio": rr,
                               "per_network": {a: arr[a].tolist() for a in ARCHS}}
        print(f"{K:>3} | {rw['het_mean']:>10.3f} {rw['hom_mean']:>10.3f} {rw['perm_p']:>9.3f} | "
              f"{rb['het_mean']:>11.3f} {rb['hom_mean']:>11.3f} {rb['perm_p']:>9.4f} | "
              f"{rr['het_mean']:>9.2f} {rr['hom_mean']:>9.2f} {rr['perm_p']:>9.4f}")

    prof = {a: np.array([louvain_profile(Z[f"{s}_{a}__W"], Z[f"{s}_{a}__modid"]) for s in SC.SEEDS])
            for a in ARCHS}

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.5))
    fig.subplots_adjust(wspace=0.34)

    for i, (field, title, ylab) in enumerate([
            ("within", "(a) Within-community weight", "weight / network mean"),
            ("between", "(b) Between-community weight", ""),
            ("ratio", "(c) Within / between ratio", "")]):
        ax = axes[i]
        for a in ARCHS:
            m = [res["by_k"][str(K)][field][f"{a}_mean"] for K in K_RANGE]
            sd = [res["by_k"][str(K)][field][f"{a}_sd"] for K in K_RANGE]
            h = 1.96 * np.array(sd) / np.sqrt(len(SC.SEEDS))
            ax.plot(K_RANGE, m, "-o", color=S.COL[a], ms=4, label=S.LABEL[a])
            ax.fill_between(K_RANGE, np.array(m) - h, np.array(m) + h, color=S.COL[a], alpha=0.18, lw=0)
        for j, K in enumerate(K_RANGE):
            p = res["by_k"][str(K)][field]["perm_p"]
            ax.annotate("*" if p < 0.05 else "", (K, ax.get_ylim()[1]), ha="center", va="top",
                        fontsize=13, color=S.ACCENT)
        ax.set_xticks(K_RANGE)
        ax.set_xlabel("communities forced, K")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)
            ax.axvline(3, color=S.CHANCE, ls=":", lw=1.2)
            ax.text(3.05, ax.get_ylim()[0], " K used in the\n aggregate figure", fontsize=7.5,
                    color=S.CHANCE, va="bottom")

    ax = axes[3]
    centres = (np.linspace(0, 1, 25)[:-1] + np.linspace(0, 1, 25)[1:]) / 2
    for a in ARCHS:
        m, h = SC.ci95(prof[a])
        ax.plot(centres, m, color=S.COL[a], label=S.LABEL[a])
        ax.fill_between(centres, m - h, m + h, color=S.COL[a], alpha=0.20, lw=0)
    ax.axhline(1.0, color=S.CHANCE, ls=":", lw=1.2)
    ax.set_xlabel("normalised distance from the diagonal")
    ax.set_ylabel("weight / network mean")
    ax.set_title("(d) K-free: each network's own Louvain partition", loc="left", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Does the block result depend on forcing three communities? Stars mark a paired "
                 "difference at p < 0.05\nthe between-community gap holds at every K, and on each "
                 "network's own partition", fontweight="bold", y=1.05, fontsize=11)
    S.save(fig, FIGS / "S18_block_k_sweep.png")

    json.dump(res, open(SC.DATA / "block_k_sweep.json", "w"), indent=1)
    print(f"\nwrote {SC.DATA / 'block_k_sweep.json'}")


if __name__ == "__main__":
    main()
