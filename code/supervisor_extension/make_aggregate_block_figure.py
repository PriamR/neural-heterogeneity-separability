"""One block matrix per arm, aggregated over twenty networks, to sit beside the forty-panel montage.

Averaging the forty matrices as they stand would be meaningless: unit index is arbitrary, so unit 7 of one
network has nothing to do with unit 7 of another, and averaging would wash every block into a flat field.
Three things have to be fixed before an average means anything.

  Ordering. Units are sorted by community, largest community first, so position along the axis carries
  community membership rather than an arbitrary label.

  Community count. Louvain finds 3 to 5 communities depending on the network, so block boundaries land in
  different places and averaging blurs them. Every network is therefore partitioned into exactly three
  communities by spectral clustering on its own weight matrix, which puts the boundaries at comparable
  positions. This is the same fixed-K partition used for the block-concentration statistic, where it exists
  to stop the community count from driving the comparison.

  Scale. The two arms differ in mean edge weight, so a raw average would show the arm with larger weights as
  "more structured" regardless of how that weight is arranged. Each network's matrix is divided by its own
  mean off-diagonal weight before averaging, which makes every network contribute its pattern and none
  contribute its scale.

Sizes still differ (34 to 58 active units), so each matrix is resampled onto a common grid by proportional
position before averaging. The result is an honest answer to "what does the typical network in this arm look
like", and it is a summary of the same matrices the montage shows one by one.

Run: python code/supervisor_extension/make_aggregate_block_figure.py
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

warnings.filterwarnings("ignore")
S.apply()
FIGS = SC.FIGS
ARCHS = ("het", "hom")
K_FIXED = 3
GRID = 96                      # common resampling grid
Z = np.load(SC.DATA / "structure_matrices_n20.npz")


def spectral_partition(W, K=K_FIXED, seed=0):
    from sklearn.cluster import SpectralClustering
    A = np.array(W, float)
    np.fill_diagonal(A, 0.0)
    A = (A + A.T) / 2.0
    return SpectralClustering(n_clusters=K, affinity="precomputed", random_state=seed,
                              assign_labels="kmeans").fit_predict(A)


def canonical(W, seed, K=K_FIXED, grid=GRID):
    """Community-ordered, scale-normalised, resampled onto a common grid.

    Communities are ordered by size so that the largest block always sits top-left, which is what makes the
    same grid cell mean the same thing in every network.
    """
    lab = spectral_partition(W, K, seed)
    order = np.concatenate([np.where(lab == c)[0]
                            for c in sorted(range(K), key=lambda c: -(lab == c).sum())])
    sizes = [int((lab == c).sum()) for c in sorted(range(K), key=lambda c: -(lab == c).sum())]
    A = np.array(W, float)[np.ix_(order, order)]
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    scale = A[iu].mean()
    if scale > 0:
        A = A / scale                                  # pattern, not magnitude
    # proportional resampling: unit i covers [i/n, (i+1)/n)
    idx = np.floor(np.arange(n) / n * grid).astype(int)
    out = np.zeros((grid, grid))
    cnt = np.zeros((grid, grid))
    np.add.at(out, (idx[:, None].repeat(n, 1), idx[None, :].repeat(n, 0)), A)
    np.add.at(cnt, (idx[:, None].repeat(n, 1), idx[None, :].repeat(n, 0)), 1.0)
    out = np.divide(out, cnt, out=np.zeros_like(out), where=cnt > 0)
    bounds = np.cumsum(sizes)[:-1] / n * grid
    return out, bounds, lab, order


def block_means(W, seed, K=K_FIXED):
    """Mean weight in each of the K-by-K blocks, scale-normalised, communities ordered by size."""
    lab = spectral_partition(W, K, seed)
    order = sorted(range(K), key=lambda c: -(lab == c).sum())
    A = np.array(W, float)
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    scale = A[iu].mean()
    M = np.zeros((K, K))
    for a, ca in enumerate(order):
        for b, cb in enumerate(order):
            ia, ib = np.where(lab == ca)[0], np.where(lab == cb)[0]
            sub = A[np.ix_(ia, ib)]
            if a == b:
                m = np.triu_indices(len(ia), 1)
                M[a, b] = sub[m].mean() if len(ia) > 1 else np.nan
            else:
                M[a, b] = sub.mean()
    return M / scale if scale > 0 else M


def diag_profile(W, seed, K=K_FIXED, nbin=24):
    """Mean weight against normalised distance from the community-block diagonal.

    A modular graph puts its weight near the diagonal and little elsewhere, so this collapses the picture to
    one curve per network and makes twenty of them comparable.
    """
    lab = spectral_partition(W, K, seed)
    order = np.concatenate([np.where(lab == c)[0]
                            for c in sorted(range(K), key=lambda c: -(lab == c).sum())])
    A = np.array(W, float)[np.ix_(order, order)]
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    scale = A[iu].mean()
    d = np.abs(iu[0] - iu[1]) / n
    w = A[iu] / scale if scale > 0 else A[iu]
    edges = np.linspace(0, 1, nbin + 1)
    b = np.clip(np.digitize(d, edges[1:-1]), 0, nbin - 1)
    return np.array([w[b == i].mean() if (b == i).any() else np.nan for i in range(nbin)]), edges


def main():
    agg, bnds, blk, prof = {}, {}, {}, {}
    for arch in ARCHS:
        mats, bs, bms, ps = [], [], [], []
        for s in SC.SEEDS:
            W = Z[f"{s}_{arch}__W"]
            m, b, _, _ = canonical(W, s)
            mats.append(m)
            bs.append(b)
            bms.append(block_means(W, s))
            p, edges = diag_profile(W, s)
            ps.append(p)
        agg[arch] = np.mean(mats, 0)
        bnds[arch] = np.mean(bs, 0)
        blk[arch] = np.nanmean(bms, 0)
        prof[arch] = np.array(ps)
    _, edges = diag_profile(Z[f"{SC.SEEDS[0]}_het__W"], SC.SEEDS[0])
    centres = (edges[:-1] + edges[1:]) / 2

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.7))
    fig.subplots_adjust(wspace=0.42)
    vmax = max(np.percentile(agg[a], 99) for a in ARCHS)

    for i, arch in enumerate(ARCHS):
        ax = axes[i]
        im = ax.imshow(agg[arch], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        for b in bnds[arch]:
            ax.axhline(b - 0.5, color="#00e5ff", lw=1.0)
            ax.axvline(b - 0.5, color="#00e5ff", lw=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({'ab'[i]}) {S.LABEL[arch]}, mean of 20 networks", loc="left", fontsize=10.5,
                     color=S.COL[arch])
        plt.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("units, ordered by community")

    # the same information as nine numbers
    ax = axes[2]
    diff = blk["hom"] - blk["het"]
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    for a in range(K_FIXED):
        for b in range(K_FIXED):
            ax.text(b, a, f"{blk['het'][a,b]:.2f}\n{blk['hom'][a,b]:.2f}", ha="center", va="center",
                    fontsize=9, color="k" if abs(diff[a, b]) < np.abs(diff).max() * 0.6 else "w")
    ax.set_xticks(range(K_FIXED))
    ax.set_yticks(range(K_FIXED))
    ax.set_xticklabels([f"c{i+1}" for i in range(K_FIXED)])
    ax.set_yticklabels([f"c{i+1}" for i in range(K_FIXED)])
    ax.set_title("(c) Block means (het above, hom below)", loc="left", fontsize=10.5)
    # Label on the bar rather than above it: as a colourbar title it collided with the panel title,
    # which is set with loc="left" and runs the full width of the axis.
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("hom $-$ het", fontsize=8)

    # collapse to one curve per network so twenty of them can carry a confidence band
    ax = axes[3]
    for arch in ARCHS:
        m, h = SC.ci95(prof[arch])
        ax.plot(centres, m, color=S.COL[arch], label=S.LABEL[arch])
        ax.fill_between(centres, m - h, m + h, color=S.COL[arch], alpha=0.20, lw=0)
    ax.axhline(1.0, color=S.CHANCE, ls=":", lw=1.2)
    ax.text(0.98, 1.0, " network mean", fontsize=7.5, color=S.CHANCE, va="bottom", ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("normalised distance from the diagonal")
    ax.set_ylabel("edge weight / network mean")
    ax.set_title("(d) Weight against distance from diagonal", loc="left", fontsize=10.5)
    ax.legend(fontsize=8)

    fig.suptitle("The forty block matrices aggregated: every network partitioned into three communities, "
                 "ordered by size,\nscaled by its own mean edge weight, and resampled onto a common grid",
                 fontweight="bold", y=1.06, fontsize=11)
    S.save(fig, FIGS / "S17_aggregate_blocks.png")

    # ------------------------------------------------------------------ split version
    # The same four panels as two figures of two, so each is legible at full text width
    # in the thesis. (a),(b) are the aggregate matrices; (c),(d) quantify them.
    figA, axA = plt.subplots(1, 2, figsize=(9.6, 4.9))
    figA.subplots_adjust(wspace=0.30)
    for i, arch in enumerate(ARCHS):
        ax = axA[i]
        im = ax.imshow(agg[arch], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        for b in bnds[arch]:
            ax.axhline(b - 0.5, color="#00e5ff", lw=1.0)
            ax.axvline(b - 0.5, color="#00e5ff", lw=1.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        S.panel_title(ax, "ab"[i], f"{S.LABEL[arch]}, mean of twenty networks")
        cb = plt.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("Edge weight / network mean", fontsize=8)
        cb.outline.set_visible(False)
        ax.set_xlabel("Units, ordered by community")
    axA[0].set_ylabel("Units, ordered by community")
    S.save(figA, FIGS / "S17a_aggregate_matrices.png")

    figB, axB = plt.subplots(1, 2, figsize=(10.2, 4.6))
    figB.subplots_adjust(wspace=0.34)
    ax = axB[0]
    diff = blk["hom"] - blk["het"]
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    for a in range(K_FIXED):
        for b in range(K_FIXED):
            ax.text(b, a, f"{blk['het'][a,b]:.2f}\n{blk['hom'][a,b]:.2f}", ha="center", va="center",
                    fontsize=9, color="k" if abs(diff[a, b]) < np.abs(diff).max() * 0.6 else "w")
    ax.set_xticks(range(K_FIXED)); ax.set_yticks(range(K_FIXED))
    ax.set_xticklabels([f"Community {i+1}" for i in range(K_FIXED)], fontsize=8)
    ax.set_yticklabels([f"Community {i+1}" for i in range(K_FIXED)], fontsize=8)
    ax.tick_params(length=0)
    ax.grid(False)
    S.panel_title(ax, "a", "Block means, heterogeneous over homogeneous")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Homogeneous $-$ heterogeneous", fontsize=8)
    cb.outline.set_visible(False)

    ax = axB[1]
    for arch in ARCHS:
        m, h = SC.ci95(prof[arch])
        S.band(ax, centres, m, h, S.COL[arch], label=S.LABEL[arch])
    S.ref_line(ax, y=1.0, label="Network mean")
    ax.set_xlabel("Normalised distance from the diagonal")
    ax.set_ylabel("Edge weight / network mean")
    S.panel_title(ax, "b", "Edge weight against distance from the diagonal")
    S.legend(ax, loc="upper right")
    S.save(figB, FIGS / "S17b_aggregate_quantified.png")


    # the statistics behind panels (c) and (d)
    print("\nwithin-community block mean (diagonal of panel c), per network:")
    v = {}
    for arch in ARCHS:
        v[arch] = np.array([np.nanmean(np.diag(block_means(Z[f"{s}_{arch}__W"], s))) for s in SC.SEEDS])
    r = paired_triple(v["het"], v["hom"])
    print(f"  het {r['het_mean']:.3f}  hom {r['hom_mean']:.3f}  d {r['delta']:+.3f}  dz {r['dz']:+.2f}  "
          f"p {S.fmt_p(r['perm_p'], r['perm_floor'])}  hom higher {r['n']-r['wins']}/{r['n']}")
    print("\noff-diagonal block mean:")
    v = {}
    for arch in ARCHS:
        out = []
        for s in SC.SEEDS:
            M = block_means(Z[f"{s}_{arch}__W"], s)
            out.append(np.nanmean(M[~np.eye(K_FIXED, dtype=bool)]))
        v[arch] = np.array(out)
    r = paired_triple(v["het"], v["hom"])
    print(f"  het {r['het_mean']:.3f}  hom {r['hom_mean']:.3f}  d {r['delta']:+.3f}  dz {r['dz']:+.2f}  "
          f"p {S.fmt_p(r['perm_p'], r['perm_floor'])}  hom higher {r['n']-r['wins']}/{r['n']}")
    print("\nnearest-diagonal bin of the profile (panel d, first bin):")
    r = paired_triple(prof["het"][:, 0], prof["hom"][:, 0])
    print(f"  het {r['het_mean']:.3f}  hom {r['hom_mean']:.3f}  d {r['delta']:+.3f}  dz {r['dz']:+.2f}  "
          f"p {S.fmt_p(r['perm_p'], r['perm_floor'])}  hom higher {r['n']-r['wins']}/{r['n']}")


if __name__ == "__main__":
    main()
