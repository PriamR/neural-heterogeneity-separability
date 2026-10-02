"""One illustrative pair of block matrices for the main text, beside the K-sweep curves of fig:blockconc.

The K-sweep and the aggregate matrices in the appendix are averages over the forty graphs. This figure shows one
seed's two networks, partitioned into exactly three communities by the same spectral clustering used for the
block-concentration statistic and the aggregate, community-ordered (largest first) and divided by the network's
own mean off-diagonal weight so that the colour scale carries pattern rather than magnitude. The two panels share
one colour scale. The within/between weight ratio in each title is the stored K = 3 block-concentration statistic for that network.

Run: python code/supervisor_extension/make_block_example_fig.py [seed]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "analysis_suite"))
import json
import style as S
from make_aggregate_block_figure import spectral_partition, Z, K_FIXED

REPO = HERE.parent.parent
FIGS = REPO / "docs" / "analysis_suite" / "figs"
CONC = json.load(open(REPO / "data_cache" / "supervisor_extension" / "block_concentration_n20.json"))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1008


def ordered(W, seed):
    lab = spectral_partition(W, K_FIXED, seed)
    comm = sorted(range(K_FIXED), key=lambda c: -(lab == c).sum())
    order = np.concatenate([np.where(lab == c)[0] for c in comm])
    A = np.array(W, float)[np.ix_(order, order)]
    iu = np.triu_indices(A.shape[0], 1)
    A = A / A[iu].mean()
    bounds = np.cumsum([(lab == c).sum() for c in comm])[:-1]
    return A, bounds


def ratio(arch, seed):
    """The K = 3 within/between weight ratio as stored for the block-concentration statistic (fig:blockconc)."""
    return float(CONC["per_network"][arch][str(seed)]["fixedK"])


def main():
    S.apply()
    panels = {}
    for arch in ("het", "hom"):
        W = Z[f"{SEED}_{arch}__W"]
        A, bounds = ordered(W, SEED)
        panels[arch] = (A, bounds, ratio(arch, SEED))
    vmax = max(np.percentile(panels[a][0], 99) for a in panels)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    fig.subplots_adjust(wspace=0.12, right=0.88)
    for ax, arch, letter in ((axes[0], "het", "a"), (axes[1], "hom", "b")):
        A, bounds, r = panels[arch]
        im = ax.imshow(A, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        for b in bounds:
            ax.axhline(b - 0.5, color="#00e5ff", lw=0.9)
            ax.axvline(b - 0.5, color="#00e5ff", lw=0.9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(S.COL[arch]); sp.set_linewidth(1.8)
        S.panel_title(ax, letter, f"{S.LABEL[arch]}, $w_{{\mathrm{{in}}}}/w_{{\mathrm{{out}}}}$ = {r:.1f}")
        ax.set_xlabel("Units, ordered by community")
    axes[0].set_ylabel("Units, ordered by community")
    cax = fig.add_axes([0.90, 0.16, 0.018, 0.68])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Edge weight / network mean", fontsize=9)
    cb.outline.set_visible(False)
    S.save(fig, FIGS / f"F4_27_block_example_seed{SEED}.png")
    print(f"  seed {SEED}: het ratio {panels['het'][2]:.2f}, hom ratio {panels['hom'][2]:.2f}")


if __name__ == "__main__":
    main()
